#!/usr/bin/env python3
"""Estimate SWE-Bench difficulty horizons using IRT beta parameters."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy import optimize as opt

from model_mapping import load_model_mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ANALYSIS_DIR = REPO_ROOT / "eval-analysis-public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit logistic curves relating SWE-Bench success to IRT beta values "
            "and plot the 50% difficulty horizon over time."
        )
    )
    parser.add_argument(
        "--runs-file",
        type=Path,
        default=EVAL_ANALYSIS_DIR / "data/external/swe_bench_runs.jsonl",
        help="Path to swe_bench_runs.jsonl from eval-analysis-public.",
    )
    parser.add_argument(
        "--model-mapping",
        type=Path,
        default=REPO_ROOT / "IRT/data/model_run_mapping.csv",
        help="JSON/CSV mapping of model keys to aliases and plot flags.",
    )
    parser.add_argument(
        "--pyirt-file",
        type=Path,
        default=REPO_ROOT / "IRT/data/swebench_all_pyirt.jsonl",
        help="JSONL with per-agent binary responses from py-irt fitting.",
    )
    parser.add_argument(
        "--beta-file",
        type=Path,
        default=REPO_ROOT / "IRT/params/swebench_all_pyirt.csv",
        help="CSV with alpha/beta parameters indexed by SWE-Bench task id.",
    )
    parser.add_argument(
        "--release-dates",
        type=Path,
        default=REPO_ROOT / "IRT/data/release_dates.yaml",
        help="YAML file mapping model names to calendar release dates.",
    )
    parser.add_argument(
        "--extra-release-dates",
        type=Path,
        default=REPO_ROOT / "IRT/data/extra_release_dates.yaml",
        help="Optional YAML with additional release dates for new models.",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=REPO_ROOT / "IRT/analysis/swebench_beta_time_horizon.csv",
        help="Where to write the per-model logistic fit summary.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=REPO_ROOT / "IRT/analysis/swebench_beta_time_horizon.png",
        help="Where to save the release-date vs beta difficulty plot.",
    )
    parser.add_argument(
        "--weight-column",
        default="invsqrt_task_weight",
        help="Which column to use for sample weighting.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.1,
        help="L2 regularization weight passed to sklearn LogisticRegression.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap draws per model for confidence intervals.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for the beta difficulty intervals.",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Plot with a linear y-axis limited to [-5, 5] rather than symlog.",
    )
    return parser.parse_args()


def load_release_dates(path: Path) -> dict[str, pd.Timestamp]:
    raw = yaml.safe_load(path.read_text())
    return {agent: pd.to_datetime(date) for agent, date in raw["date"].items()}


def load_run_metadata(path: Path) -> dict[str, RunAlias]:
    """Return mapping from model key -> alias metadata."""
    if not path.exists():
        return {}

    entries = load_model_mapping(path)
    metadata: dict[str, RunAlias] = {}
    for key, entry in entries.items():
        metadata[key] = RunAlias(display_name=entry.alias, plot=entry.plot)
    return metadata


def resolve_release_date(
    agent: str,
    release_dates: dict[str, pd.Timestamp],
    fallbacks: Sequence[str] | None = None,
) -> pd.Timestamp:
    candidates: list[str | None] = [agent]
    if fallbacks:
        candidates.extend(fallbacks)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in release_dates:
            return release_dates[candidate]
        match = re.match(r"^(\d{8})", candidate)
        if match:
            try:
                return pd.to_datetime(match.group(1))
            except ValueError:
                continue
    return pd.NaT


def get_bce_loss(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    model,
    weights: NDArray[np.float64],
) -> float:
    """Weighted binary cross-entropy, mirroring eval-analysis implementation."""
    y_pred = model.predict_proba(x)[:, 1]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    weights = weights / max(np.mean(weights), 1e-12)
    bce = -weights * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(np.mean(bce))


class SimpleLogisticModel:
    def __init__(self, coef: np.ndarray, intercept: float) -> None:
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.array([intercept])

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.intercept_[0] + np.dot(X, self.coef_.T).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def get_x_for_quantile(model: SimpleLogisticModel, quantile: float) -> float:
    logit = np.log(quantile / (1 - quantile))
    return float((logit - model.intercept_[0]) / model.coef_[0][0])


def logistic_regression(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    regularization: float,
) -> SimpleLogisticModel:
    assert np.all((y >= 0) & (y <= 1)), "y values must be in [0,1]"
    original_weight_sum = np.sum(sample_weight)
    original_average = np.average(y, weights=sample_weight)

    fractional_mask = (y > 0) & (y < 1)
    if np.any(fractional_mask):
        X_frac = X[fractional_mask]
        y_frac = y[fractional_mask]
        w_frac = sample_weight[fractional_mask]

        X_split = np.vstack([X_frac, X_frac])
        y_split = np.zeros(2 * len(y_frac))
        y_split[len(y_frac) :] = 1
        w_split = np.concatenate([(1 - y_frac) * w_frac, y_frac * w_frac])

        X = np.vstack([X[~fractional_mask], X_split])
        y = np.concatenate([y[~fractional_mask], y_split])
        sample_weight = np.concatenate([sample_weight[~fractional_mask], w_split])
        assert np.allclose(np.sum(sample_weight), original_weight_sum)
        assert np.allclose(np.average(y, weights=sample_weight), original_average)

    return _fit_weighted_logistic(X, y, sample_weight, regularization)


def _fit_weighted_logistic(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    regularization: float,
) -> SimpleLogisticModel:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    def loss(theta: NDArray[np.float64]) -> float:
        intercept = theta[0]
        coef = theta[1:]
        z = intercept + np.dot(X, coef)
        p = 1.0 / (1.0 + np.exp(-z))
        epsilon = 1e-12
        log_likelihood = np.sum(
            sample_weight
            * (y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
        )
        penalty = 0.5 * regularization * np.sum(coef**2)
        return -(log_likelihood - penalty)

    init = np.zeros(X.shape[1] + 1)
    result = opt.minimize(loss, init, method="L-BFGS-B")
    theta = result.x
    intercept = float(theta[0])
    coef = theta[1:]
    return SimpleLogisticModel(coef=coef, intercept=intercept)


def weighted_average(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def bootstrap_quantile(
    agent_df: pd.DataFrame,
    quantile: float,
    regularization: float,
    weight_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    confidence: float,
) -> tuple[float, float] | tuple[float, float, float]:
    """Return (q_low, q_high) or (low, mid, high) if quantile==0.5 for convenience."""
    draws: list[float] = []
    indices = np.arange(len(agent_df))
    low_q = max((1.0 - confidence) / 2.0, 0.0)
    high_q = 1.0 - low_q

    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sample = agent_df.iloc[sample_idx]
        y = sample["score_binarized"].to_numpy()
        if len(np.unique(y)) < 2:
            continue

        model = logistic_regression(
            sample[["beta"]].to_numpy(),
            y,
            sample[weight_column].to_numpy(),
            regularization,
        )
        draws.append(float(get_x_for_quantile(model, quantile)))

    if not draws:
        return float("nan"), float("nan")
    low = float(np.nanquantile(draws, low_q))
    high = float(np.nanquantile(draws, high_q))
    if quantile == 0.5:
        return low, float(np.nanmean(draws)), high
    return low, high


@dataclass
class RunAlias:
    display_name: str
    plot: bool


@dataclass
class AgentSummary:
    agent: str
    release_date: pd.Timestamp
    coefficient: float
    intercept: float
    bce_loss: float
    average_success: float
    beta_p50: float
    beta_p50_low: float
    beta_p50_high: float
    task_count: int
    plot: bool


def fit_agent(
    agent_name: str,
    agent_df: pd.DataFrame,
    release_lookup: dict[str, pd.Timestamp],
    regularization: float,
    weight_column: str,
    n_bootstrap: int,
    confidence: float,
    display_name: str | None = None,
    release_fallbacks: Sequence[str] | None = None,
    plot_flag: bool = True,
) -> AgentSummary | None:
    x = agent_df[["beta"]].to_numpy()
    y = agent_df["score_binarized"].to_numpy()
    weights = agent_df[weight_column].to_numpy()

    if len(np.unique(y)) < 2:
        return None

    model = logistic_regression(x, y, weights, regularization)
    beta_p50 = float(get_x_for_quantile(model, 0.5))

    rng = np.random.default_rng(1337)
    beta_p50_low, _, beta_p50_high = bootstrap_quantile(
        agent_df,
        quantile=0.5,
        regularization=regularization,
        weight_column=weight_column,
        n_bootstrap=n_bootstrap,
        rng=rng,
        confidence=confidence,
    )

    label = display_name or agent_name

    return AgentSummary(
        agent=label,
        release_date=resolve_release_date(label, release_lookup, release_fallbacks),
        coefficient=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        bce_loss=get_bce_loss(x, y, model, weights),
        average_success=weighted_average(y, weights),
        beta_p50=beta_p50,
        beta_p50_low=beta_p50_low,
        beta_p50_high=beta_p50_high,
        task_count=len(agent_df),
        plot=plot_flag,
    )


def plot_horizon(
    summaries: pd.DataFrame,
    output_path: Path,
    linear_scale: bool = False,
) -> None:
    # if "plot" in summaries.columns:
        # summaries = summaries[summaries["plot"]]
    summaries = summaries.dropna(subset=["release_date"]).sort_values("release_date")

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = {
        "Claude": "#8000FF",
        "GPT": "#0062FF",
        "Gemini": "#2E8B57",
        "Qwen": "#FF00E6",
    }

    for _, row in summaries.iterrows():
        color = (
            colors["Claude"]
            if "claude" in row.agent.lower()
            else colors["GPT"]
            if "gpt" in row.agent.lower() or "o1" in row.agent.lower()
            else colors["Gemini"]
            if "gemini" in row.agent.lower()
            else colors["Qwen"]
            if "qwen" in row.agent.lower()
            else "#555555"
        )
        ax.errorbar(
            row.release_date,
            row.beta_p50,
            yerr=np.array(
                [
                    row.beta_p50 - row.beta_p50_low,
                    row.beta_p50_high - row.beta_p50,
                ]
            ).reshape(2, 1),
            fmt="o",
            color=color,
            ecolor="#444444",
            capsize=3,
            label=row.agent if row.agent not in ax.get_legend_handles_labels()[1] else "",
        )
        # ax.text(
        #     row.release_date,
        #     row.beta_p50 + 0.05,
        #     row.agent,
        #     fontsize=5,
        #     rotation=-20,
        #     ha="left",
        # )

    # Trendline over time
    dates_numeric = summaries["release_date"].map(mdates.date2num).to_numpy()
    if len(dates_numeric) >= 2:
        slope, intercept = np.polyfit(dates_numeric, summaries["beta_p50"].to_numpy(), 1)

        date_range = pd.date_range(
            summaries["release_date"].min(),
            summaries["release_date"].max() + pd.Timedelta(days=60),
            freq="D",
        )
        date_nums = np.array([mdates.date2num(d) for d in date_range])
        predictions = slope * date_nums + intercept
        ax.plot(
            date_range,
            predictions,
            linestyle="--",
            color="#35607a",
            linewidth=2,
            alpha=0.7,
            label="Linear trend",
        )

    ax.set_title("SWE-Bench beta difficulty at 50% success", fontsize=16)
    ax.set_ylabel("IRT β difficulty (higher = harder)", fontsize=13)
    ax.set_xlabel("Model release date", fontsize=13)

    beta_values = summaries["beta_p50"].to_numpy()
    valid_beta = beta_values[~np.isnan(beta_values)]
    if linear_scale:
        ax.set_yscale("linear")
        ax.set_ylim(-5, 5)
    elif len(valid_beta):
        # Use a symmetrical log scale to keep both very large and very small betas readable.
        beta_abs_max = float(np.max(np.abs(valid_beta)))
        linthresh = max(0.1, 0.05 * beta_abs_max)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.7)
        margin = 0.1 * (np.max(valid_beta) - np.min(valid_beta))
        if np.isfinite(margin):
            lower = np.min(valid_beta) - margin
            upper = np.max(valid_beta) + margin
            ax.set_ylim(lower, upper)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=500)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    
    base_runs = pd.read_json(args.runs_file, lines=True)
    base_runs.rename(columns={"alias": "agent"}, inplace=True)
    base_cols = [
        "task_id",
        "human_minutes",
        "equal_task_weight",
        "invsqrt_task_weight",
        "task_source",
    ]
    task_meta = base_runs[base_cols].drop_duplicates("task_id")

    # records: list[pd.DataFrame] = [base_runs]
    records: list[pd.DataFrame] = []
    if args.pyirt_file.exists():
        with args.pyirt_file.open() as f:
            pyirt_records: list[dict[str, object]] = []
            for line in f:
                entry = json.loads(line)
                agent = entry["subject_id"]
                for task_id, score in entry["responses"].items():
                    pyirt_records.append(
                        {
                            "agent": agent,
                            "task_id": task_id,
                            "score_binarized": float(score),
                        }
                    )
        pyirt_df = pd.DataFrame(pyirt_records)
        pyirt_df = pyirt_df.merge(task_meta, on="task_id", how="left", validate="m:1")
        missing = pyirt_df["human_minutes"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} py-IRT rows missing task metadata; ensure task ids align"
            )
        records.append(pyirt_df)

    runs = pd.concat(records, ignore_index=True)
    beta_df = pd.read_csv(args.beta_file).rename(columns={"Unnamed: 0": "task_id"})
    beta_df.rename(columns={"a": "alpha", "b": "beta"}, inplace=True)
    runs = runs.merge(beta_df[["task_id", "beta"]], on="task_id", how="left")

    missing_beta = runs["beta"].isna().sum()
    if missing_beta:
        raise ValueError(f"{missing_beta} runs are missing beta parameters")

    run_metadata = load_run_metadata(args.model_mapping)

    release_dates = load_release_dates(args.release_dates)
    if args.extra_release_dates.exists():
        extra_dates = load_release_dates(args.extra_release_dates)
        release_dates.update(extra_dates)
    summaries: list[AgentSummary] = []
    for agent, agent_df in runs.groupby("agent"):
        metadata = run_metadata.get(agent)
        display_name = metadata.display_name if metadata else agent
        fallbacks: list[str] = []
        if display_name != agent:
            fallbacks.append(agent)
        plot_flag = metadata.plot if metadata else True
        summary = fit_agent(
            agent,
            agent_df,
            release_dates,
            args.regularization,
            args.weight_column,
            args.n_bootstrap,
            args.confidence,
            display_name=display_name,
            release_fallbacks=fallbacks,
            plot_flag=plot_flag,
        )
        if summary:
            summaries.append(summary)

    summary_df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values(
        "release_date"
    )
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_table, index=False)

    plot_horizon(summary_df, args.output_plot, linear_scale=args.linear_scale)
    print(f"Wrote summary to {args.output_table}")
    print(f"Wrote figure to {args.output_plot}")


if __name__ == "__main__":
    main()
