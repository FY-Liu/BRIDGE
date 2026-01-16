#!/bin/bash
# Run the full IRT pipeline after updating data/data_by_challenges
set -e  # Exit on error

# Parse command line arguments
EXCLUDE_NO_SUCCESS=""
USE_SUBTASK=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --exclude-no-success)
            EXCLUDE_NO_SUCCESS="--exclude-no-success"
            shift
            ;;
        --use-subtask)
            USE_SUBTASK="--use-subtask"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exclude-no-success  Exclude Cybench tasks where no model succeeded"
            echo "  --use-subtask         Use subtask-guided results instead of unguided"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Step 1: Parse Cybench logs"
if [ -n "$EXCLUDE_NO_SUCCESS" ]; then
    echo "        (excluding tasks with no successes)"
fi
if [ -n "$USE_SUBTASK" ]; then
    echo "        (using subtask-guided results)"
fi
echo "=========================================="
.venv/bin/python parse_cybench_logs.py $EXCLUDE_NO_SUCCESS $USE_SUBTASK

echo ""
echo "=========================================="
echo "Step 2: Prepare sparse py-IRT data"
echo "=========================================="
.venv/bin/python prepare_sparse_pyirt.py \
  --model-mapping data/model_run_mapping.json \
  --pyirt-input data/swe_a_pyirt.jsonl \
  --runs-input data/all_runs.jsonl \
  --gdpval-input data/gdpval_normalized_results.jsonl \
  --mlebench-input data/mlebench_normalized_results.jsonl \
  --cybench-input data/cybench_normalized_results.jsonl \
  --output data/all_a_pyirt.jsonl \
  --print-subject-counts \
  --keep-unmapped-pyirt-subjects

echo ""
echo "=========================================="
echo "Step 3: Fit IRT model"
echo "=========================================="
.venv/bin/python fit_irt.py --input_path data/all_a_pyirt.jsonl

echo ""
echo "=========================================="
echo "Step 4: Combine human minutes files"
echo "=========================================="
cat data/human_minutes_by_task.jsonl data/cybench_human_minutes_by_task.jsonl > data/combined_human_minutes.jsonl
echo "Created data/combined_human_minutes.jsonl"

echo ""
echo "=========================================="
echo "Step 5: Merge human minutes into params"
echo "=========================================="
.venv/bin/python merge_human_minutes.py --csv params/all_a_pyirt.csv --jsonl data/combined_human_minutes.jsonl

echo ""
echo "=========================================="
echo "Step 6: Run Cybench analysis"
echo "=========================================="
.venv/bin/python run_cybench_analysis.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
