"""Dynamic Cheatsheet, cumulative mode (DC-CU), on the ACE finance tasks (online setting).

Faithful to DC's DynamicCheatsheet_Cumulative loop (dynamic-cheatsheet @ 5cfe3c3, language_model.advanced_generate):
for each test sample in order: generator(prompt with current cheatsheet) -> answer; curator(question, model answer,
previous cheatsheet) -> new cheatsheet. Uses DC's own prompts and extract_answer / extract_cheatsheet.
Same test order as ACE online mode (file order), scored with ACE's DataProcessor.answer_is_correct.

Choices (recorded in EXPERIMENTS.md): max_num_rounds=1, code execution ON as in DC (its generator prompt requires it;
up to 3 rounds, DC executor; note ACE has no tool), temperature 0,
generator max_tokens 4096 and curator 2x (DC convention), thinking off. "+GT" variant: DC has no native label mode,
so the ground-truth answer is appended to the curator's [[MODEL_ANSWER]] block.

Usage (ace/.venv python, cwd = $ACE_UPSTREAM_DIR; or scripts/run_dc.sh <task> <gt|nogt> <port>):
  LOCAL_LLM_URL=http://127.0.0.1:8104/v1 $ACE_PY $ACE_ROOT/baselines/run_dc_finance.py <finer|formula> <gt|nogt> <out_dir>
"""
import json
import os
import sys

import openai

sys.path.insert(0, ".")
_DC_DIR = os.environ.get("DC_DIR") or os.path.join(
    os.environ.get("ACE_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")),
    "dynamic-cheatsheet")  # clone of github.com/suzgunmirac/dynamic-cheatsheet @ 5cfe3c3 (unmodified)
sys.path.insert(0, _DC_DIR)
from dynamic_cheatsheet.utils.execute_code import extract_and_run_python_code
from dynamic_cheatsheet.utils.extractor import extract_answer, extract_cheatsheet
from eval.finance.data_processor import DataProcessor

DC = f"{_DC_DIR}/prompts"
task, label_mode, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
gen_tpl = open(f"{DC}/generator_prompt.txt").read()
cur_tpl = open(f"{DC}/curator_prompt_for_dc_cumulative.txt").read()

cfg = json.load(open("./eval/finance/data/sample_config.json"))[task]
proc = DataProcessor(task)
samples = proc.process_task_data([json.loads(l) for l in open(cfg["test_data"])])
if os.environ.get("DC_LIMIT"):
    samples = samples[: int(os.environ["DC_LIMIT"])]
client = openai.OpenAI(api_key="EMPTY", base_url=os.environ["LOCAL_LLM_URL"], timeout=1800)


def chat(prompt, max_tokens, messages=None):
    for _ in range(20):
        try:
            r = client.chat.completions.create(
                model="qwen3-32b", messages=messages or [{"role": "user", "content": prompt}], temperature=0.0,
                max_tokens=max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
        except openai.BadRequestError as e:
            if "maximum context length" in str(e) and max_tokens > 512:
                max_tokens //= 2
                continue
            raise
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            continue
    raise RuntimeError("LLM call failed")


FLAG = "EXECUTE CODE!"


def generate_with_code(prompt, max_tokens, max_depth=3):
    """DC LanguageModel.generate: run a ```python block followed by FLAG, feed output back, up to max_depth rounds."""
    history, final_output = [{"role": "user", "content": prompt}], ""
    for depth in range(1, max_depth + 2):
        output = chat(None, max_tokens, messages=history) or "(No response generated)"
        pre = output.split(FLAG)[0].strip() if FLAG in output else ""
        if FLAG in output and len(pre) >= 3 and pre.endswith("```"):
            executed = (extract_and_run_python_code(pre) or "(No code block found to execute)").strip()
            current = f"{pre}\n{FLAG}\n\n{executed}"
            final_output = f"{final_output}\n\n{current}".strip()
            if depth <= max_depth:
                warn = (" (This is the last round. No more code execution will be allowed. Please present your final "
                        "solution now.)") if depth == max_depth else ""
                history += [{"role": "assistant", "content": current},
                            {"role": "user", "content": "Proceed with any additional steps required and provide the "
                             "completed solution. If everything is already complete, type FINAL ANSWER and submit it in "
                             f"the expected format. If you are stuck, please try alternative methods to solve the problem "
                             f"and provide the final solution.{warn}"}]
                continue
            return final_output
        return f"{final_output}\n\n{output}".strip()
    return final_output


cheatsheet = "(empty)"
log = open(f"{out_dir}/steps.jsonl", "a")
done = sum(1 for _ in open(f"{out_dir}/steps.jsonl"))  # resume support
if done:
    last = [json.loads(l) for l in open(f"{out_dir}/steps.jsonl")][-1]
    cheatsheet = last["new_cheatsheet"]
correct = sum(json.loads(l)["correct"] for l in open(f"{out_dir}/steps.jsonl"))
for i, s in enumerate(samples):
    if i < done:
        continue
    input_txt = s["question"] + (f"\n\n{s['context']}" if s["context"].strip() else "")
    gen_out = generate_with_code(gen_tpl.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", cheatsheet), 4096)
    answer = extract_answer(gen_out)
    ok = bool(proc.answer_is_correct(answer, s["target"]))
    model_answer = gen_out if label_mode == "nogt" else f"{gen_out}\n\nGROUND TRUTH ANSWER: {s['target']}"
    cur_out = chat(cur_tpl.replace("[[QUESTION]]", input_txt).replace("[[MODEL_ANSWER]]", model_answer)
                   .replace("[[PREVIOUS_CHEATSHEET]]", cheatsheet), 8192)
    new_cs = extract_cheatsheet(cur_out, cheatsheet)
    correct += ok
    log.write(json.dumps({"i": i, "answer": answer, "target": s["target"], "correct": ok,
                          "cheatsheet_chars": len(new_cs), "new_cheatsheet": new_cs,
                          "generator_output": gen_out, "curator_output": cur_out}) + "\n")
    log.flush()
    cheatsheet = new_cs
    print(f"[{i + 1}/{len(samples)}] correct={ok} cumulative_acc={correct / (i + 1):.4f} cheatsheet_chars={len(new_cs)}", flush=True)

# Official metric = ACE's evaluate_accuracy (per-tag for FiNER, per-sample for Formula), same as ACE's test passes.
rows = [json.loads(l) for l in open(f"{out_dir}/steps.jsonl")]
acc = proc.evaluate_accuracy([r["answer"] for r in rows], [r["target"] for r in rows])
acc = acc[0] if isinstance(acc, tuple) else acc
json.dump({"task": task, "label_mode": label_mode, "accuracy": acc, "samples_correct": correct,
           "total": len(samples)}, open(f"{out_dir}/final.json", "w"), indent=1)
print(f"RESULT DC-CU {task} {label_mode}: accuracy={acc:.4f} (sample-level {correct}/{len(samples)})")
