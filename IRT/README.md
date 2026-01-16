.venv/bin/python prepare_sparse_pyirt.py   --model-mapping /home/lfy/BRIDGE/IRT/data/model_run_mapping.json  --pyirt-input /home/lfy/BRIDGE/IRT/data/swe_a_pyirt.jsonl --runs-input data/all_runs.jsonl   --output data/swe_metr_s_pyirt.jsonl --print-subject-counts

.venv/bin/python prepare_sparse_pyirt.py   --model-mapping /home/lfy/BRIDGE/IRT/data/model_run_mapping.json  --pyirt-input /home/lfy/BRIDGE/IRT/data/swe_a_pyirt.jsonl --runs-input data/all_runs.jsonl --gdpval-input /home/lfy/BRIDGE/IRT/data/gdpval_normalized_results.jsonl --mlebench-input /home/lfy/BRIDGE/IRT/data/mlebench_normalized_results.jsonl  --output data/all_a_pyirt.jsonl --print-subject-counts --keep-unmapped-pyirt-subjects

.venv/bin/python /home/lfy/BRIDGE/IRT/fit_irt.py --input_path /home/lfy/BRIDGE/IRT/data/all_a_pyirt.jsonl

.venv/bin/python merge_human_minutes.py --csv params/all_a_pyirt.csv
