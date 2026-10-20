#!/bin/bash
# Per-GPU utilization at 1 Hz from DCGM, timestamped: <epoch> GPU <id> POWER GPUTL FBUSD SMACT SMOCC TENSO
#   SMACT = fraction of time >= 1 warp is resident on an SM (DCGM_FI_PROF_SM_ACTIVE), averaged over SMs: the
#   honest "is the GPU doing work" number; GPUTL (nvidia-smi utilization) only says "a kernel was running".
# Usage: gpu_monitor.sh <out.txt>   (runs until killed)
OUT=$1
stdbuf -oL dcgmi dmon -e 155,203,252,1002,1003,1004 -d 1000 | while IFS= read -r l; do
  case "$l" in GPU*) echo "$(date +%s.%N) $l" >> $OUT;; esac
done
