#!/bin/bash
# Run benchmarks on Flab dataset splits.
# Usage: cd notebooks && bash run-flab.sh
# Override output and config dirs via environment variables:
#   LOG_DIR=/path/to/logs CONFIGS_DIR=/path/to/configs/flab bash run-flab.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$HOME/output/Flab/logs}"
CONFIGS_DIR="${CONFIGS_DIR:-$REPO_DIR/notebooks/configs/flab}"

mkdir -p "$LOG_DIR"

run_benchmark() {
    local log_file="$1"
    local config="$2"
    nohup poetry run python benchmark_all.py --config "$config" > "$LOG_DIR/$log_file" &
    wait $!
}

# Flab IID-Random
run_benchmark "CE-Random.log"        "$CONFIGS_DIR/config-ce-random.yaml"
run_benchmark "FT-CE-Random.log"     "$CONFIGS_DIR/config-ft-ce-random.yaml"
run_benchmark "TIM-Random.log"       "$CONFIGS_DIR/config-gini-tim-random.yaml"
run_benchmark "FT-TIM-Random.log"    "$CONFIGS_DIR/config-ft-gini-tim-random.yaml"

# Flab 1H

run_benchmark "GINI-TIM-1H-3H.log"        "$CONFIGS_DIR/config-gini-tim-1H-3H.yaml"
run_benchmark "FT-GINI-TIM-1H-3H.log"     "$CONFIGS_DIR/config-ft-gini-tim-1H-3H.yaml"

# Flab 3H
run_benchmark "GINI-TIM-3H-1H.log"        "$CONFIGS_DIR/config-gini-tim-3H-1H.yaml"
run_benchmark "FT-GINI-TIM-3H-1H.log"     "$CONFIGS_DIR/config-ft-gini-tim-3H-1H.yaml"
