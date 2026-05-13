#!/bin/bash
# Refactored benchmark suite for GET and baselines
# Now supports 10-fold cross-validation for TU datasets across all models.

ANOMALY_TASKS=("stage4_amazon_anomaly")
TU_TASKS=("stage4_tu_proteins" "stage4_tu_nci1" "stage4_tu_nci109" "stage4_tu_enzymes" "stage4_tu_mutagenicity")
MOLECULAR_TASKS=("stage3_zinc" "stage3_molhiv" "stage3_molpcba" "stage3_peptides_struct_probe" "stage3_peptides_func_probe")
EXPRESSIVITY_TASKS=("stage2_csl")

# Focus default: sparse local higher-order vs pairwise on molecule-heavy tasks.
MODELS=("pairwiseget" "fullget")
RUN_MODE="${RUN_MODE:-molecular}"  # molecular | full

# Define GPUs to use
GPUS=(0 1)

run_task() {
  local gpu_id=$1
  local task=$2
  local model=$3
  local train_ratio=$4
  local cv_folds=$5
  local suffix=$6

  echo "[GPU $gpu_id] Starting $task | Model: $model | CV: $cv_folds..."
  mkdir -p outputs/protocol
  OUTPUT_FILE="outputs/protocol/${task}_${model}${suffix}.json"
  
  EXTRA_ARGS=()
  if [[ "$model" == "fullget" ]]; then
    EXTRA_ARGS+=(--lambda_m 1.0)
  fi
  
  # Standardize epochs and patience for benchmarking
  # TU datasets usually converge fast but need high statistical rigor
  if [[ $cv_folds -gt 1 ]]; then
    EPOCHS=50
    PATIENCE=20
  else
    EPOCHS=100
    PATIENCE=30
  fi

  CUDA_VISIBLE_DEVICES=$gpu_id conda run -n get python experiments/run_protocol.py \
    --task "$task" \
    --model_name "$model" \
    --batch_size 64 \
    --train_ratio "$train_ratio" \
    --cv_folds "$cv_folds" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    "${EXTRA_ARGS[@]}" \
    --output_file "$OUTPUT_FILE"
}

# Generate all task combinations
ALL_TASKS=()
for model in "${MODELS[@]}"; do
  if [[ "$RUN_MODE" == "full" ]]; then
    # Anomaly tasks with 1% and 40% splits (standard for PyGOD/anomaly benchmarks)
    for task in "${ANOMALY_TASKS[@]}"; do
      ALL_TASKS+=("$task $model 0.01 1 _split1")
      ALL_TASKS+=("$task $model 0.40 1 _split40")
    done
    
    # TU tasks with 10-fold CV (standard research protocol)
    for task in "${TU_TASKS[@]}"; do
      ALL_TASKS+=("$task $model 0.70 10 _cv10")
    done
  fi

  # Molecule-centric stage.
  for task in "${MOLECULAR_TASKS[@]}"; do
    ALL_TASKS+=("$task $model 0.70 1 _none")
  done

  # Optional broader protocol.
  if [[ "$RUN_MODE" == "full" ]]; then
    for task in "${EXPRESSIVITY_TASKS[@]}"; do
      ALL_TASKS+=("$task $model 0.70 1 _none")
    done
  fi
done

# Task distributor logic
total_tasks=${#ALL_TASKS[@]}
num_gpus=${#GPUS[@]}

echo "Distributing $total_tasks tasks across $num_gpus GPUs..."

for ((i=0; i<num_gpus; i++)); do
  (
    gpu_id=${GPUS[$i]}
    for ((j=i; j<total_tasks; j+=num_gpus)); do
      task_info=(${ALL_TASKS[$j]})
      task=${task_info[0]}
      model=${task_info[1]}
      ratio=${task_info[2]}
      folds=${task_info[3]}
      suffix=${task_info[4]}
      if [[ "$suffix" == "_none" ]]; then suffix=""; fi
      
      run_task "$gpu_id" "$task" "$model" "$ratio" "$folds" "$suffix"
    done
  ) &
done

wait
echo "All experiments completed. Summarizing results..."
# Optional: run a summary script here if available
