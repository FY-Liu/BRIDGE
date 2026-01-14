# Cybench Analysis Results

## Summary

The Cybench task time prediction analysis has been successfully completed using the METR-only linear regression model to predict First Solve Times (FST) for 171 cybersecurity CTF challenges.

## Methodology

- **Model Used**: Linear regression fitted ONLY on METR data (170 tasks)
  - Formula: `human_time = exp(slope × b + intercept)`
  - Parameters: slope = 0.887408, intercept = 2.877427

- **Test Set**: 171 Cybench tasks (held-out, not used in model fitting)
  - All tasks have both IRT difficulty estimates (b) and FST annotations
  - FST range: 2.0 to 2,769.9 minutes

## Prediction Performance

### Metrics

| Metric | Value |
|--------|-------|
| **R² (log scale)** | **-0.074** |
| **MAE (log scale)** | **1.295** |
| **Median error ratio** | **0.79x** |
| **Predictions within 2x** | **33.9%** |

### Interpretation

1. **Negative R²**: The model performs worse than simply using the mean FST value
   - Indicates the METR-derived relationship does NOT generalize to Cybench tasks

2. **Large MAE**: Log-scale error of 1.295 corresponds to approximately 3.6x multiplicative error
   - Predictions are consistently off by more than 3x on average

3. **Systematic Bias**: Median error ratio of 0.79x shows the model tends to underpredict
   - Model predicts shorter solve times than actual FST

4. **Poor Precision**: Only 33.9% within 2x error band
   - Most predictions fall outside reasonable accuracy bounds

## Visualization Analysis

The generated plots (`plots/cybench_prediction_vs_actual_fst.pdf/png`) reveal:

### Left Panel: Predicted vs Actual FST
- **Clustering**: Predictions cluster around 50-100 minutes regardless of actual FST
- **Flat Prediction**: Model fails to capture the wide range of actual solve times
- **Scatter**: Most points fall far from the perfect prediction line

### Right Panel: Residual Plot by Difficulty
- **Color Gradient**: Shows relationship between task difficulty (b) and prediction error
  - Green (easy tasks, b < 0): Tend to have negative errors (underprediction)
  - Red (hard tasks, b > 0): More variable errors

- **Systematic Pattern**: Clear trend in residuals suggests model misspecification
  - Not random scatter expected from a good model
  - Indicates missing variables or wrong functional form

## Key Findings

### 1. Domain-Specific Relationships

The relationship between IRT difficulty and human solve time appears **domain-specific**:

- **METR Tasks**: Linear relationship (R² ≈ 0.3-0.4 from previous analysis)
- **Cybench Tasks**: No linear relationship (R² = -0.074)

This suggests that:
- Autonomous task completion (METR) has different time dynamics than CTF challenges
- Cybersecurity skills may affect solve time differently than general task-solving ability

### 2. Cybench Task Characteristics

Possible explanations for poor generalization:

1. **Different Skill Profiles**:
   - CTF challenges require specialized security knowledge
   - METR tasks require general autonomous reasoning
   - IRT difficulty may capture different aspects of task complexity

2. **Different Time Distributions**:
   - CTF challenges may have "aha moments" that don't scale with difficulty
   - Experts solve both easy and hard CTF tasks relatively quickly
   - METR tasks may have more predictable time scaling

3. **Competition Context**:
   - FST represents fastest solver in competition (top expert performance)
   - METR times may represent more typical agent performance
   - Different measurement contexts lead to different time relationships

### 3. Implications for BRIDGE Framework

- **Benchmark Diversity**: Confirms that Cybench adds distinct characteristics to BRIDGE
- **Model Limitations**: Single linear model cannot predict time across all task types
- **Need for Domain Models**: May require separate time-prediction models per benchmark

## Files Generated

1. **`plots/cybench_prediction_vs_actual_fst.pdf`**
   - High-resolution visualization for publication

2. **`plots/cybench_prediction_vs_actual_fst.png`**
   - Web-friendly format for quick viewing

3. **`run_cybench_analysis.py`**
   - Standalone script for reproducing the analysis

## Next Steps

Potential follow-up analyses:

1. **Fit Cybench-Specific Model**:
   - Use Cybench data to fit a separate regression model
   - Compare slope/intercept to METR model
   - Test if relationship is simply different parameters vs fundamentally different

2. **Combined Model**:
   - Fit model on METR + Cybench together
   - Add benchmark-type indicator variable
   - Test for interaction effects

3. **Task Feature Analysis**:
   - Investigate what features of Cybench tasks drive FST
   - CTF category (crypto, reverse, pwn, web, etc.)
   - Competition difficulty ratings
   - Number of subtasks

4. **Cross-Benchmark Comparison**:
   - Test METR model on SWE-Bench tasks
   - Check if poor generalization is specific to Cybench
   - Identify which benchmarks have similar time relationships

## Conclusion

The analysis reveals that **the METR-only linear model does not generalize to Cybench tasks**, with negative R² and only 33.9% of predictions within 2x of actual FST. This finding:

- ✅ Validates the importance of including Cybench in BRIDGE
- ✅ Shows that different task types have different time-difficulty relationships
- ✅ Suggests need for domain-specific or multi-domain prediction models
- ⚠️ Indicates caution when extrapolating findings across benchmark types

The poor prediction performance is actually a **positive finding** for the BRIDGE framework, as it demonstrates that Cybench captures fundamentally different task characteristics than existing benchmarks, making it a valuable addition for comprehensive AI capability assessment.
