# Bubble-Aware Co-Scheduling (design, not implemented or validated)

Status: algorithm design for the paper. Nothing here is implemented or measured yet; the current scheduler
(`results/OPTIMIZATIONS.md` O1, O2) is the special case described in §6.

## 1. Why the current scheduler is not "bubble-aware"

The current DeltaServe scheduler is myopic: each step it admits as many FT samples as the step-time predictor says
fit under the SLO, the backward fires whenever the activation buffer is full, and it is paused reactively (big
prefill, slow step). It never asks *how long the GPU will stay free*. Two consequences, both measured:

- A backward cycle that starts just before a burst time-slices against it (RESULTS §8: rounds overlapping traffic
  stretch the cycle 2.4 -> 3.5-3.9 s, and the guard has to step in).
- When there is no bubble at all (WebArena: 6 agents decoding continuously), nothing raises training's share, and
  rounds take 800-1070 s instead of ~150 s.

Agentic traffic is bursty *with structure*: a session alternates LLM call -> tool execution -> LLM call ... ->
final answer -> user think time -> next task. In TraceLab, tool latency has median 0.3 s while user think time is
92.3 % of session wall clock (median 1.4 min, p90 20.6 min). So the server can predict bubbles from which state
each session is in. This is the signal the algorithm uses.

## 2. Problem

Online, at every engine step: choose (a) the FT tokens admitted to the step, (b) whether to start a backward cycle
and on how many buffered tokens, (c) whether the backward runs or is paused, to

    maximize   trained tokens per second  (equivalently: minimize RSI round turnaround)
    subject to P(TTFT > SLO_ttft) <= eps,  P(token gap > SLO_tbt) <= eps,
               staleness of the oldest untrained round <= D   (soft)

Decisions (a)-(c) are what DeltaServe already controls; the new ingredients are a bubble forecast (§3), a
bubble-sized backward (§4.1), a slack controller that trades SLO headroom for progress only when training falls
behind (§4.3), and chunked FT samples (§4.4).

## 3. Bubble forecast from session state

**Sessions.** A session is identified by an optional `X-Session-Id` header, or else by the prompt prefix (requests of
one agent share their history; the prefix-cache block hash of the first k blocks is the key). For each active
session i the server keeps: whether a request is in flight, and if not, the time e_i since its last response ended
and the kind of that response: `tool` (the output ends in a tool call / action) or `final` (anything else). With a
tool/reasoning parser the kind is exact; otherwise use the mixture in "Gap model" below.

**Gap model.** Two empirical gap distributions, learned online from the session's own history (and seeded from the
trace): F_tool (gap after a tool call: seconds) and F_user (gap after a final answer: minutes). Without the response
kind, F = w·F_tool + (1−w)·F_user with the posterior weight w_i(e) updated by Bayes as e grows (a session silent for
30 s is almost surely in user think time).

**Arrival survival.** For a waiting session, the probability it has *not* sent its next request within τ is the
conditional survival

    S_i(τ) = (1 − F_i(e_i + τ)) / (1 − F_i(e_i)),

and new sessions arrive at a background rate λ0 (EWMA of first-request arrivals). The probability that no new
request arrives in the next τ seconds is

    P0(τ) = exp(−λ0·τ) · Π_{i waiting} S_i(τ).

**Busy horizon.** Requests in flight keep the GPU busy for R̂ = max_i (predicted remaining decode tokens of i) ×
t_decode(b), with the remaining length from the session's past output lengths (EWMA) — no length predictor model is
needed for this to be useful, a rough mean suffices.

**Bubble length.** The idle bubble that starts when the in-flight requests finish, at confidence q (e.g. 0.8):

    L_q = sup { τ : P0(R̂ + τ) >= q }     (bubble length after the in-flight requests finish; 0 if the queue is non-empty)

Cheap to compute: one pass over active sessions per update (every ~50 ms or on each arrival/completion), and
P0 is monotone so L_q is a binary search over τ.

Also exported: the expected inference work in the next H seconds, Ŵ(H) = Σ_i (1 − S_i(H))·ĉ_i, where ĉ_i is the
predicted GPU time of session i's next request (its prefill delta ≈ its last tool output, plus its mean output
length of decode). Used by the controller in §4.3.

## 4. Decisions

### 4.1 Bubble-sized backward cycles

Model the backward cycle time on B buffered tokens as T_b(B) = a + b·B when alone, and κ(b_d)·T_b(B) when sharing
the GPU with b_d decoding requests (κ fitted online from the backward log, like the step-time estimator; κ ≈ 1.8 on
WebArena, from 2.6 -> 4.6 s).

When the backward is idle and the buffer holds B_buf tokens:

```
L = L_q (forecast)                        # free GPU time ahead, at confidence q
if queue is empty and L > 0:              # an idle bubble is opening
    B* = max { B <= B_buf : T_b(B) <= L } # the batch the bubble can finish
    if B* >= B_min: fire(B*)              # train a prefix of the buffer; the rest stays for the next cycle
    elif lag() > 0: fire(B_buf)           # behind schedule: take the bubble anyway, guard protects inference
    else: hold                            # bubble too short to be worth a cycle; keep filling
else:                                     # inference is running (decode-only or mixed)
    if B_buf == capacity or lag() > 0: fire(B_buf)   # today's rule; contention handled by pause + guard
    else: hold
```

Effect: a cycle is started when it is likely to finish before the next burst, and its size is cut to the bubble
instead of always being "the whole buffer". This needs the backward to accept a partial batch (train the first B*
tokens of the buffer, keep the rest), which the long-context path (`backward_batch_tokens`) already does for big
buffers.

### 4.2 Bubble-aligned forward fill

FT forward admission (the existing greedy per-step algorithm) gets one extra rule: if a bubble is forecast to open
within the next step-time and the buffer is not full, prefer filling it *in the bubble* (one FT-only eager forward
fills the buffer at no inference cost) over riding the current inference step with its stall cost. Concretely, the
stall cost of a mixed step is charged against the budget ρ of §4.3; the FT-only fill is free. So with bubbles
available, FT never rides inference steps; without bubbles, it rides them as much as the budget allows.

### 4.3 Slack controller: spend SLO headroom only when training falls behind

The current average-TBT gate has a fixed budget ρ (decode stall allowed per window, 0.5 s per 2 s). Make it a dual
variable, updated once per window w:

    lag_w   = max(0, age of the oldest untrained round − D)          # training behind the target turnaround
    viol_w  = fraction of requests in the window that violated an SLO
    ρ_{w+1} = clip( ρ_w + η1·lag_w/D − η2·(viol_w − eps)/eps ,  0, ρ_max )

and use the same multiplier to scale the max-TBT and TTFT margins used by admission (between the configured SLO and
a hard ceiling). Reading: with bubbles, training keeps up, lag stays 0, and ρ decays to 0 — FT only uses bubbles
and inference sees no stall at all. Without bubbles (WebArena), lag grows, ρ rises until either training keeps up
or the observed violation rate reaches eps. It is the standard primal-dual treatment of "maximize progress subject
to an average constraint", and it turns today's hand-set ρ into the price of SLO headroom.

The forecast enters as a lookahead: if Ŵ(H) predicts that inference work in the next H seconds is low (a bubble is
coming that can absorb the lag), the update uses lag_w − (GPU time of the forecast bubble), so the controller does
not spend SLO headroom now for work a bubble will do for free.

### 4.4 Chunked FT samples

A sample of n tokens costs (β + γ)·n of step time when it rides a step (the estimator's coefficients), so a
7K-token WebArena sample adds 0.4-0.8 s to its step and fits no TBT budget. Split it into chunks of

    c = floor( slack_step / (β + γ) ),   slack_step = SLO_tbt·m − t̂_step

tokens and feed one chunk per step using chunked prefill: the FT request keeps its KV across ⌈n/c⌉ steps (freed after
the last chunk), and each chunk's activations land at its offset in the buffer. The backward sees the same rows as
for an unchunked sample. This relaxes invariant 2 of CLAUDE.md (FT is single-step prefill) to "prefill-only, may
span steps, never decodes"; cost: the FT request holds n tokens of KV for a few steps.

### 4.5 Reactive safety net (unchanged)

The slow-step guard (O2) stays: the forecast and the estimator are predictions, and time-slicing interference is not
predictable. Any inference step issued during a backward that is still unfinished after the threshold pauses the
backward on every rank.

## 5. One step, all together

```
on every engine step:
    update session states (arrivals, completions, response kinds)       # §3, O(changed sessions)
    every 50 ms: recompute L_q, Ŵ(H)                                     # §3
    every window: update ρ (slack controller)                           # §4.3
    if backward idle: maybe fire a bubble-sized cycle                    # §4.1
    admit FT to this step (existing greedy), with
        - per-step TBT/TTFT margins scaled by the controller             # §4.3
        - stall charged to ρ only for mixed steps; FT-only fill free     # §4.2
        - long samples split into chunks of c tokens                     # §4.4
    pause the backward if: big prefill | slow step (guard) | all ranks agree   # O1, O2
```

Overhead: the forecast is a pass over active sessions every 50 ms (hundreds of sessions = microseconds); the rest
is O(1) per step on top of today's admission loop.

## 6. Relation to the current scheduler

Setting q = 0 (no forecast: L_q = ∞ when idle, 0 otherwise), η1 = η2 = 0 (fixed ρ) and c = ∞ (no chunking) gives
exactly today's scheduler: fire on a full buffer, greedy SLO admission, fixed stall budget, prefill/guard pauses.
So each component can be ablated on its own: +forecast (§4.1/4.2), +controller (§4.3), +chunking (§4.4).

## 7. What the experiments would have to show

- Bursty replay (RESULTS §8): fewer cycles overlapping bursts, guard triggers down, round wait closer to seq
  (today 55 vs 40 s at r1) at the same tail latency.
- WebArena (no bubbles): the controller and chunking raise training throughput from ~28 % of standalone, with the
  SLO violation rate at eps rather than ~0.
- Forecast quality on the TraceLab trace offline (no GPU needed): calibration of P0(τ) and the error of L_q, with
  and without the response kind.

## 8. Risks / open choices

- Session identification by prefix can merge two sessions with an identical system prompt; the first k blocks must
  include the first user message.
- The forecast is only as good as the gap model; the slack controller makes the algorithm degrade to "today's
  scheduler with an adaptive budget" when the forecast is useless.
- Partial-batch backward changes the effective batch size per optimizer step; either keep the optimizer step per
  full buffer (accumulate gradients across partial cycles) or accept variable batch sizes.
