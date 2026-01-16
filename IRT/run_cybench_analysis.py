#!/usr/bin/env python3
"""
Run Cybench analysis section from analysis.ipynb
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
BASE_DIR = Path('/home/lfy/BRIDGE/IRT')
cybench_results_path = BASE_DIR / 'data' / 'cybench_normalized_results.jsonl'
params_path = BASE_DIR / 'params' / 'all_a_pyirt.csv'
plots_dir = BASE_DIR / 'plots'
plots_dir.mkdir(exist_ok=True)

# Load Cybench task IDs and difficulty labels
def load_jsonl_records(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records

cybench_records = load_jsonl_records(cybench_results_path)
cybench_task_ids = {record['task_id'] for record in cybench_records}

# Create task_id -> difficulty_label mapping
task_difficulty_map = {}
for record in cybench_records:
    task_id = record['task_id']
    if task_id not in task_difficulty_map:
        task_difficulty_map[task_id] = record.get('difficulty_label', 'unlabelled')

print(f"# of tasks in Cybench: {len(cybench_task_ids)}")

# Count by difficulty
difficulty_counts = {}
for label in task_difficulty_map.values():
    difficulty_counts[label] = difficulty_counts.get(label, 0) + 1
print(f"Difficulty distribution: {difficulty_counts}")

# Load IRT parameters
df = pd.read_csv(params_path)
# Handle unnamed first column for task_id
if df.columns[0] != 'task_id':
    df.rename(columns={df.columns[0]: 'task_id'}, inplace=True)

# Mark Cybench tasks
df['task_source'] = 'other'
df.loc[df['task_id'].isin(cybench_task_ids), 'task_source'] = 'cybench'

# Add difficulty_label column
df['difficulty_label'] = df['task_id'].map(task_difficulty_map).fillna('unlabelled')

# Define prediction function (from METR-only fit)
# These parameters are from fitting ONLY on METR data
slope = 0.887408
intercept = 2.877427

def predict_minutes_from_b(b_values):
    """Predict human minutes from IRT difficulty parameter b"""
    return np.exp(slope * b_values + intercept)

# Get Cybench predictions
cybench_predictions = df[(df['task_id'].isin(cybench_task_ids)) &
                         (df['task_source'] == 'cybench') &
                         (df['b'].notna())].copy()

print(f"\nCybench tasks with difficulty estimates: {len(cybench_predictions)}")

# Predict human time
cybench_predictions['predicted_minutes'] = predict_minutes_from_b(cybench_predictions['b'])

# Filter to tasks with actual human_minutes for evaluation
mask = cybench_predictions['human_minutes'].notna()
print(f"Cybench tasks with actual FST annotations: {mask.sum()}")

if mask.sum() > 0:
    # Calculate metrics
    actual = cybench_predictions.loc[mask, 'human_minutes'].values
    predicted = cybench_predictions.loc[mask, 'predicted_minutes'].values

    # R² on log scale
    y_actual = np.log(actual)
    y_pred = np.log(predicted)
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r_squared_cybench = 1 - (ss_res / ss_tot)

    # MAE on log scale
    mae_log = np.mean(np.abs(y_actual - y_pred))

    # Median error ratio
    error_ratio = predicted / actual
    median_error_ratio = np.median(error_ratio)

    # Within 2x
    within_2x = np.mean((error_ratio >= 0.5) & (error_ratio <= 2.0)) * 100

    print(f"\nCybench Prediction Metrics (using METR-only model):")
    print(f"  R² (log scale): {r_squared_cybench:.3f}")
    print(f"  MAE (log scale): {mae_log:.3f}")
    print(f"  Median error ratio: {median_error_ratio:.2f}x")
    print(f"  Predictions within 2x: {within_2x:.1f}%")

    # Create log error column for visualization
    cybench_predictions.loc[mask, 'log_error'] = y_pred - y_actual

    # Define colors for difficulty levels
    difficulty_colors = {
        'very_easy': '#2ecc71',  # Green
        'easy': '#3498db',       # Blue
        'medium': '#f39c12',     # Orange
        'hard': '#e74c3c',       # Red
        'unlabelled': '#95a5a6'  # Gray
    }

    difficulty_order = ['very_easy', 'easy', 'medium', 'hard', 'unlabelled']
    difficulty_labels_display = {
        'very_easy': 'Very Easy',
        'easy': 'Easy',
        'medium': 'Medium',
        'hard': 'Hard',
        'unlabelled': 'Unlabelled'
    }

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Predicted vs Actual, colored by difficulty
    for diff in difficulty_order:
        diff_mask = mask & (cybench_predictions['difficulty_label'] == diff)
        if diff_mask.sum() > 0:
            ax1.scatter(
                cybench_predictions.loc[diff_mask, 'human_minutes'],
                cybench_predictions.loc[diff_mask, 'predicted_minutes'],
                c=difficulty_colors[diff],
                label=f"{difficulty_labels_display[diff]} ({diff_mask.sum()})",
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--',
             linewidth=2, label='Perfect Prediction')

    # 2x error bands
    ax1.fill_between([min_val, max_val], [min_val/2, max_val/2],
                     [min_val*2, max_val*2], alpha=0.15, color='gray',
                     label='2x Error Band')

    ax1.set_xlabel('Actual FST (minutes)', fontsize=12)
    ax1.set_ylabel('Predicted Time (minutes)', fontsize=12)
    ax1.set_title(f'Cybench: Predicted vs Actual FST\n(R² = {r_squared_cybench:.3f}, n={mask.sum()})',
                  fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right plot: Residuals colored by difficulty
    for diff in difficulty_order:
        diff_mask = mask & (cybench_predictions['difficulty_label'] == diff)
        if diff_mask.sum() > 0:
            ax2.scatter(
                cybench_predictions.loc[diff_mask, 'human_minutes'],
                cybench_predictions.loc[diff_mask, 'log_error'],
                c=difficulty_colors[diff],
                label=difficulty_labels_display[diff],
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax2.axhline(y=np.log(2), color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=-np.log(2), color='gray', linestyle=':', alpha=0.5)

    # Add text annotations for the lines
    ax2.text(max_val * 0.7, np.log(2) + 0.15, '2x over', fontsize=9, color='gray')
    ax2.text(max_val * 0.7, -np.log(2) - 0.25, '2x under', fontsize=9, color='gray')

    ax2.set_xlabel('Actual FST (minutes)', fontsize=12)
    ax2.set_ylabel('Log Error (log(pred) - log(actual))', fontsize=12)
    ax2.set_title('Prediction Residuals by Difficulty Label',
                  fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = plots_dir / 'cybench_prediction_vs_actual_fst.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

    # Also save as PNG for easy viewing
    output_path_png = plots_dir / 'cybench_prediction_vs_actual_fst.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path_png}")

    plt.show()

else:
    print("\nNo Cybench tasks with both difficulty estimates and FST annotations found.")
