"""torchtune 0.6.1 components for the vLLM + torchtune baseline of rsi_mix.sh (multi-round RSI FT).

  rounds_dataset  the rsi_mix_prep.py rounds (<dir>/r<k>.tok.jsonl, already rendered + masked like DeltaServe
                  trains them) in order; a sample of round k blocks until <dir>/go<k> exists, i.e. until the
                  traffic driver releases that round. Every round's size is a multiple of the FSDP world size,
                  so a step never mixes two rounds.
  ProgressLogger  after every optimizer step (rank 0) appends "<unix time> <cumulative input tokens> <loss>" to
                  <log_dir>/progress.txt: the driver's "round k trained" signal (tokens counted across ranks by
                  lms_tune.data.distributed_collate).
Model / checkpointer / collate come from DeltaServe-Baselines' lms_tune (on PYTHONPATH next to this package).
"""
import json
import os
import time
from typing import Any, Dict

from torch.utils.data import Dataset
from torchtune.training.metric_logging import MetricLoggerInterface

from lms_tune import data as lms_data


class RoundsDataset(Dataset):

    def __init__(self, path: str, rounds: int):
        self._path = path
        self._samples, self._round = [], []
        for k in range(rounds):
            for line in open(f"{path}/r{k}.tok.jsonl"):
                r = json.loads(line)
                self._samples.append((r["tokens"], r["labels"]))
                self._round.append(k)
        self._released = set()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        k = self._round[index]
        while k not in self._released:
            if os.path.exists(f"{self._path}/go{k}"):
                self._released.add(k)
            else:
                if not os.path.exists(f"{self._path}/waiting"):  # model loaded, blocked on a round: the driver's ready signal
                    open(f"{self._path}/waiting", "w").close()
                time.sleep(0.05)
        tokens, labels = self._samples[index]
        return {"tokens": list(tokens), "labels": list(labels)}


def rounds_dataset(tokenizer, path: str, rounds: int, packed: bool = False) -> RoundsDataset:
    if packed:
        raise ValueError("packed=True is not supported")
    return RoundsDataset(path, rounds)


class ProgressLogger(MetricLoggerInterface):

    def __init__(self, log_dir: str, **kwargs):
        os.makedirs(log_dir, exist_ok=True)
        self._f = open(f"{log_dir}/progress.txt", "a", buffering=1)

    def log(self, name, data, step) -> None:
        self.log_dict({name: data}, step)

    def log_config(self, config) -> None:
        pass

    def log_dict(self, payload, step) -> None:
        self._f.write(f"{time.time():.3f} {lms_data.TOKENS_SERVED} {float(payload.get('loss', 0.0)):.4f}\n")

    def close(self) -> None:
        self._f.close()
