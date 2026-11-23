#!/bin/bash
# Usage: cd notebooks && bash run-poly.sh
# Override output and config dirs via environment variables:
#   LOG_DIR=/path/to/logs CONFIGS_DIR=/path/to/configs bash run-poly.sh


REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$HOME/output/poly/logs}"
CONFIGS_DIR="${CONFIGS_DIR:-$REPO_DIR/notebooks/configs/}"

mkdir -p "$LOG_DIR"

run_benchmark() {
    local log_file="$1"
    local config="$2"
    poetry run python benchmark_all.py --config "$config" > "$LOG_DIR/$log_file" &
    wait $!
}
run_benchmark "lora-ce-splitA.log"    "$CONFIGS_DIR/lora-poly/config-poly-lora-ce-splitA.yaml"
run_benchmark "lora-FT-CE-splitA.log"    "$CONFIGS_DIR/lora-poly/config-ft-poly-lora-CE-splitA.yaml"
run_benchmark "lora-GINI-TIM-splitA.log"    "$CONFIGS_DIR/lora-poly/config-poly-lora-gini-tim-splitA.yaml"
run_benchmark "lora-FT-GINI-TIM-splitA.log"    "$CONFIGS_DIR/lora-poly/config-ft-poly-lora-gini-tim-splitA.yaml"

run_benchmark "CE-splitA.log"           "$CONFIGS_DIR/poly/config-poly-CE-splitA.yaml"
run_benchmark "FT-CE-splitA.log"        "$CONFIGS_DIR/poly/config-ft-poly-CE-splitA.yaml"
run_benchmark "GINI-TIM-splitA.log"     "$CONFIGS_DIR/poly/config-poly-gini-tim-splitA.yaml"
run_benchmark "FT-GINI-TIM-splitA.log"  "$CONFIGS_DIR/poly/config-ft-poly-gini-tim-splitA.yaml"

run_benchmark "lora-ce-splitB.log"          "$CONFIGS_DIR/lora-poly/config-poly-lora-ce-splitB.yaml"
run_benchmark "lora-FT-CE-splitB.log"       "$CONFIGS_DIR/lora-poly/config-ft-poly-lora-CE-splitB.yaml"
run_benchmark "lora-GINI-TIM-splitB.log"    "$CONFIGS_DIR/lora-poly/config-poly-lora-gini-tim-splitB.yaml"
run_benchmark "lora-FT-GINI-TIM-splitB.log" "$CONFIGS_DIR/lora-poly/config-ft-poly-lora-gini-tim-splitB.yaml"

run_benchmark "CE-splitB.log"           "$CONFIGS_DIR/poly/config-poly-CE-splitB.yaml"
run_benchmark "FT-CE-splitB.log"        "$CONFIGS_DIR/poly/config-ft-poly-CE-splitB.yaml"
run_benchmark "GINI-TIM-splitB.log"     "$CONFIGS_DIR/poly/config-poly-gini-tim-splitB.yaml"
run_benchmark "FT-GINI-TIM-splitB.log"  "$CONFIGS_DIR/poly/config-ft-poly-gini-tim-splitB.yaml"


