#!/bin/zsh
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/dreamer/bin/python
STEPS=${STEPS:-50000}
SEEDS=(${@:-0 1 2})
(( $# )) || SEEDS=(0 1 2)
for SEED in $SEEDS; do
    RUN_DIR="checkpoint_v3arch_seed${SEED}"
    mkdir -p "$RUN_DIR"
    echo "=== seed $SEED -> $RUN_DIR ($(date '+%b %d %H:%M'))"
    SEED=$SEED RUN_DIR=$RUN_DIR STEPS=$STEPS PYTHONUNBUFFERED=1 $PY -W ignore train.py > "$RUN_DIR/train.log" 2>&1 || echo "seed $SEED failed, see $RUN_DIR/train.log"
done
$PY summarize_seeds.py checkpoint_v3arch_seed*
