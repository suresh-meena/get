#!/bin/bash

# Real world data experiments
TASKS_GPU0=(
  "stage2_csl"
  "stage3_zinc"
  "stage3_peptides_struct_probe"
  "stage4_tu_classification"
  "stage4_amazon_anomaly"
)

TASKS_GPU1=(
  "stage3_molhiv"
  "stage3_peptides_func_probe"
  "stage4_yelpchi_anomaly"
  "stage4_tfinance_anomaly"
  "stage4_tsocial_anomaly"
)

MODELS=("fullget" "pairwiseget" "quadratic_only")

# Function to run tasks on a specific GPU
run_tasks() {
  local gpu_id=$1
  shift
  local tasks=("$@")
  for task in "${tasks[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "Running $task with $model on GPU $gpu_id..."
      OUTPUT_FILE="outputs/protocol/${task}_${model}.json"
      CUDA_VISIBLE_DEVICES=$gpu_id conda run -n get python experiments/run_protocol.py \
        --task $task \
        --model_name $model \
        --batch_size 256 \
        --eval_batch_size 256 \
        --inference_mode_train fixed \
        --inference_mode_eval armijo \
        --armijo_max_backtracks 5 \
        --output $OUTPUT_FILE
      echo "Completed $task with $model on GPU $gpu_id"
    done
  done
}

run_tasks 0 "${TASKS_GPU0[@]}" &
run_tasks 1 "${TASKS_GPU1[@]}" &

wait
echo "All real-world data experiments completed."
