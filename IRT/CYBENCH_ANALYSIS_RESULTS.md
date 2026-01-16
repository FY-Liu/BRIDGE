# Cybench Analysis Results

## Summary

The Cybench task time prediction analysis uses the METR-only linear regression model to predict First Solve Times (FST) for **40 cybersecurity CTF challenges** that have actual model evaluation data.

## Data Sources

### Human Time Annotations
- **Source**: Cybench GitHub repository (`analytics/CTF fastest solve times.json`)
- **Format**: First Solve Time (FST) - fastest time any human competitor solved each CTF challenge
- **URL**: https://github.com/andyzorigin/cybench

### Model Evaluations
- **Source**: `data/data_by_challenges/` (40 task directories with 690 log files)
- **Models**: 11 models with actual evaluations:
  - claude-3-opus, claude-3.5-sonnet, gemini-1.5-pro, gpt-4o
  - llama-3-70b-chat, llama-3.1-405b-instruct, llama-3.1-70b-instruct-turbo
  - mistral-large-2407, mixtral-8x22b-instruct, o1-preview, qwen2-72b-instruct
- **Total Records**: 360 (task, model) evaluation pairs
- **Success Rate**: 25/360 = 6.9%

## Methodology

- **Model Used**: Linear regression fitted ONLY on METR data (170 tasks)
  - Formula: `human_time = exp(slope × b + intercept)`
  - Parameters: slope = 0.887408, intercept = 2.877427

- **Test Set**: 40 Cybench tasks with actual model evaluations
  - All 40 tasks have both IRT difficulty estimates (b) and FST annotations

## Prediction Performance

### Metrics

| Metric | Value |
|--------|-------|
| **R² (log scale)** | **0.236** |
| **MAE (log scale)** | **1.045** |
| **Median error ratio** | **1.66x** |
| **Predictions within 2x** | **37.5%** |

### Interpretation

1. **Positive R²**: The model explains 23.6% of variance in Cybench FST
   - Better than random, but moderate fit
   - METR-derived relationship partially transfers to Cybench

2. **MAE of 1.045**: Log-scale error corresponds to ~2.8x multiplicative error
   - Reasonable given cross-domain prediction

3. **Median Error Ratio 1.66x**: Model tends to overpredict solve times
   - Cybench tasks are solved faster than predicted by IRT difficulty

4. **37.5% within 2x**: About a third of predictions are reasonably accurate

## Model Success Rates by Model

| Model | Success Rate |
|-------|-------------|
| claude-3.5-sonnet | 5/40 (12.5%) |
| claude-3-opus | 4/40 (10.0%) |
| o1-preview | 4/40 (10.0%) |
| gpt-4o | 3/40 (7.5%) |
| llama-3.1-405b-instruct | 3/40 (7.5%) |
| gemini-1.5-pro | 2/40 (5.0%) |
| mixtral-8x22b-instruct | 2/40 (5.0%) |
| llama-3-70b-chat | 1/40 (2.5%) |
| llama-3.1-70b-instruct-turbo | 1/38 (2.6%) |

## Files Generated

1. **`data/cybench_normalized_results.jsonl`** - 360 evaluation records (40 tasks × ~9 models)
2. **`data/cybench_human_minutes_by_task.jsonl`** - 40 tasks with FST data
3. **`data/all_a_pyirt.jsonl`** - Combined IRT data (1044 tasks total)
4. **`params/all_a_pyirt.csv`** - IRT parameters with human_minutes (710 tasks have FST)
5. **`plots/cybench_prediction_vs_actual_fst.pdf/png`** - Visualization

## IRT Pipeline Summary

| Dataset | Tasks | Models | Total Responses |
|---------|-------|--------|-----------------|
| SWE-Bench | 500 | ~125 | ~62,500 |
| MLE-Bench | 114 | ~13 | ~1,482 |
| GDPVal | 220 | 16 | 3,520 |
| METR | 170 | ~8 | ~19,000 |
| **Cybench** | **40** | **11** | **360** |
| **Total** | **1,044** | **186** | **86,689** |

## Conclusion

Using only the 40 Cybench tasks with actual model evaluation data (instead of all 171 FST tasks):

- **R² = 0.236**: The METR-derived linear model partially generalizes to Cybench
- **37.5% within 2x**: Moderate prediction accuracy for cross-domain tasks
- **1.66x median error ratio**: Model tends to overpredict solve times
- The 40 tasks with evaluations show better model fit than the full 171-task set, suggesting these selected tasks have characteristics more aligned with METR

This validates that Cybench adds meaningful data to the BRIDGE framework while showing that domain-specific calibration may improve predictions.
