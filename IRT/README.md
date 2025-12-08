# IRT workflow

This directory now supports fitting a single sparse 2PL model that spans the
SWE-Bench runs and the multi-benchmark runs stored in `data/all_runs.jsonl`.

## Update model identifiers

The correspondence file `data/model_run_mapping.csv` contains the columns
`run_id, alias, model_key, display_name, plot`.  Add or edit rows here whenever
you see a new run identifier or alias in any of the source files.  The
`model_key` column defines the canonical `subject_id` that will appear in the
final py-IRT dataset, so set it to a stable slug (for example `claude-3-opus`).

## Build a combined py-IRT dataset

Use `prepare_sparse_pyirt.py` to merge any number of py-IRT JSONL files with any
number of run logs.  The command below rebuilds the combined dataset that feeds
the current analysis:

```bash
.venv/bin/python prepare_sparse_pyirt.py \
  --model-mapping data/model_run_mapping.csv \
  --pyirt-input data/swebench_all_pyirt.jsonl \
  --runs-input data/all_runs.jsonl \
  --output data/swebench_plus_all_runs_pyirt.jsonl
```

The script canonically renames subject identifiers (using the mapping file),
deduplicates multiple runs of the same agent by taking the max observed score,
and reports how many rows had to fall back to raw identifiers.

## Fit the sparse 2PL model

Run the standard trainer on the new JSONL file.  The example below uses the
default CPU configuration and writes the parameters to
`params/swebench_plus_all_runs_pyirt.csv`:

```bash
.venv/bin/python fit_irt.py \
  --input_path data/swebench_plus_all_runs_pyirt.jsonl \
  --device cpu
```

Use the resulting CSV to analyze joint difficulty across the unified question
set.  Re-run the `prepare_sparse_pyirt.py` step whenever you edit the mapping
file or add new run logs.





.venv/bin/python prepare_sparse_pyirt.py   --model-mapping /home/lfy/BRIDGE/IRT/data/model_run_mapping.json  --pyirt-input /home/lfy/BRIDGE/IRT/data/swe_a_pyirt.jsonl --runs-input data/all_runs.jsonl   --output data/swe_metr_s_pyirt.jsonl --print-subject-counts

.venv/bin/python prepare_sparse_pyirt.py   --model-mapping /home/lfy/BRIDGE/IRT/data/model_run_mapping.json  --pyirt-input /home/lfy/BRIDGE/IRT/data/swe_a_pyirt.jsonl --runs-input data/all_runs.jsonl   --output data/swe_metr_a_pyirt.jsonl --print-subject-counts --keep-unmapped-pyirt-subjects

.venv/bin/python /home/lfy/BRIDGE/IRT/fit_irt.py --input_path /home/lfy/BRIDGE/IRT/data/swe_metr_a_pyirt.jsonl

.venv/bin/python merge_human_minutes.py --csv params/swe_metr_a_pyirt.csv