# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Chat SFT rows -> (token ids, loss spans), matching RSI-system's trainers.

RSI-system (github.com/Jianshu-She/RSI-system) turns agent traces into chat
JSONL (``{"messages": [...]}`` per line, ``rsi/finetune/DATA_FORMAT.md``) and
trains on the assistant tokens only. The co-served trainer must see exactly the
tokens and targets those offline trainers do, so this module ports their
``encode`` functions:

* ``"last"`` / ``"all"`` — ``rsi/finetune/train_lora.py`` (``--loss_on``,
  ``--thinking``).
* ``"ace"`` — ``benchmarks/ace/.../train_lora.py`` (thinking off; skips the
  AppWorld few-shot demo dialogue; equals ``"all"`` + thinking off on
  single-turn rows except for the prompt rendering of the first turn).

A span ``(s, e)`` marks target token positions ``s .. e-1``: the loss is the
next-token cross-entropy of predicting ``ids[j]`` from position ``j - 1`` for
every ``j`` in a span.
"""

THINK_OFF = "<think>\n\n</think>\n\n"
LOSS_ON = ("last", "all", "ace")


def encode_chat(tok, messages: list[dict], loss_on: str = "all",
                thinking: str = "off") -> tuple[list[int], list[tuple[int, int]]]:
    """Return ``(ids, spans)``; ``spans`` is empty when the row has no usable
    target (the caller drops it, as the offline trainers do)."""
    if loss_on not in LOSS_ON:
        raise ValueError(f"chat_loss_on must be one of {LOSS_ON}, got {loss_on!r}")
    if thinking not in ("on", "off"):
        raise ValueError(f"chat_thinking must be 'on' or 'off', got {thinking!r}")
    if loss_on == "ace":
        return _encode_ace(tok, messages)
    tmpl = {} if thinking == "on" else {"enable_thinking": False}

    if loss_on == "last":
        # One target turn: prompt = the template's generation prompt for the
        # history, exactly as at inference.
        pre = _ids(tok.apply_chat_template(messages[:-1], tokenize=True,
                                           add_generation_prompt=True, **tmpl))
        full = _ids(tok.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=False, **tmpl))
        if full[:len(pre)] == pre and len(full) > len(pre):
            return full, [(len(pre), len(full))]
        return full, []

    # Every assistant turn, built turn by turn with the prefix the model was
    # conditioned on when it generated that turn.
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    nl = tok.encode("\n", add_special_tokens=False)
    for m in messages:
        role = m["role"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"chat_loss_on='all' supports system/user/assistant, got {role}")
        if role == "assistant":
            ids += tok.encode("<|im_start|>assistant\n"
                              + (THINK_OFF if thinking == "off" else ""),
                              add_special_tokens=False)
            body = tok.encode(m["content"] + "<|im_end|>", add_special_tokens=False)
            spans.append((len(ids), len(ids) + len(body)))
            ids += body + nl
        else:
            ids += tok.encode(f"<|im_start|>{role}\n{m['content']}<|im_end|>\n",
                              add_special_tokens=False)
    return ids, spans


def _encode_ace(tok, messages: list[dict]):
    """The ACE trainer's encode: the official template (thinking off) up to the
    first assistant turn after the task start, then every later turn appended
    by hand; loss on each assistant content + ``<|im_end|>``. AppWorld prompts
    carry a few-shot demo dialogue — the task starts at the last user turn
    saying "solve the actual task", and nothing before it is a target."""
    start = max([i for i, m in enumerate(messages)
                 if m["role"] == "user" and "solve the actual task" in m["content"]]
                or [0])
    first = next((i for i, m in enumerate(messages)
                  if m["role"] == "assistant" and i > start), None)
    if first is None:
        return [], []
    ids = tok.encode(tok.apply_chat_template(messages[:first], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False),
                     add_special_tokens=False)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    nl = tok.encode("\n", add_special_tokens=False)
    spans: list[tuple[int, int]] = []
    for i in range(first, len(messages)):
        m = messages[i]
        if m["role"] == "assistant":
            if i != first:
                ids += tok.encode("<|im_start|>assistant\n" + THINK_OFF,
                                  add_special_tokens=False)
            t = tok.encode(m["content"], add_special_tokens=False) + [end]
            spans.append((len(ids), len(ids) + len(t)))
            ids += t + nl
        else:
            ids += tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n",
                              add_special_tokens=False)
    return ids, spans


def _ids(x) -> list[int]:
    """``apply_chat_template(tokenize=True)`` returns a list on transformers 4.x
    and a BatchEncoding-like mapping on 5.x."""
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    return list(x)
