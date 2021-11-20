#!/usr/bin/env bash
NGPUS=$1
CFG=$2
PY_ARGS=${@:3}

# avoid too many threads
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -x
python -m torch.distributed.launch --nproc_per_node=${NGPUS} ../../tools/train_val.py \
    --config $CFG \
    ${PY_ARGS}

