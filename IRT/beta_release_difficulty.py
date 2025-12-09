#!/usr/bin/env python3
"""Plot release dates vs. beta difficulty using the combined py-IRT outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize as opt

from model_mapping import load_model_mapping

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESPONSES = REPO_ROOT / "data/all_a_pyirt.jsonl"
DEFAULT_BETA = REPO_ROOT / "params/all_a_pyirt.csv"
DEFAULT_MAPPING = REPO_ROOT / "data/model_run_mapping.json"
DEFAULT_TABLE = REPO_ROOT / "analysis/beta_release_difficulty.csv"
DEFAULT_PLOT = REPO_ROOT / "analysis/beta_release_difficulty.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the beta difficulty each model solves at 50% success "
            "and plot it against the model release date."
        )
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES,
        help="py-IRT JSONL responses (default: %(default)s)",
    )
    parser.add_argument(
        "--beta-params",
        type=Path,
        default=DEFAULT_BETA,
        help="CSV with IRT task parameters (default: %(default)s)",
    )
    parser.add_argument(
        "--model-mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help="JSON mapping with release dates (default: %(default)s)",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Where to save the per-model summary table (default: %(default)s)",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="Where to write the release-date plot (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level for beta difficulty (default: %(default)s)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Bootstrap draws per model for confidence intervals",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.1,
        help="L2 penalty applied to the logistic slope (default: %(default)s)",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Plot with a linear y-axis instead of symmetric log scale",
    )
    return parser.parse_args()


def load_pyirt_responses(path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with path.open() as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}") from exc
            agent = payload.get("subject_id")
            responses = payload.get("responses") or {}
            if not isinstance(responses, dict):
                continue
            for task_id, score in responses.items():
                records.append(
                    {
                        "agent": str(agent),
                        "task_id": str(task_id),
                        "score": float(score),
                    }
                )
    if not records:
        raise ValueError(f"No responses were found in {path}")
    return pd.DataFrame(records)


def load_task_betas(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.rename(columns={"Unnamed: 0": "task_id"})
    frame = frame.rename(columns={"b": "beta"})
    expected_cols = {"task_id", "beta"}
    if not expected_cols.issubset(frame.columns):
        missing = ", ".join(sorted(expected_cols - set(frame.columns)))
        raise ValueError(f"Missing required beta columns: {missing}")
    return frame[["task_id", "beta"]].dropna()


def load_release_dates(path: Path) -> tuple[dict[str, pd.Timestamp], dict[str, str]]:
    mapping = load_model_mapping(path)
    releases: dict[str, pd.Timestamp] = {}
    aliases: dict[str, str] = {}
    for key, entry in mapping.items():
        label = entry.alias or key
        aliases[key] = label
        timestamp = (
            pd.to_datetime(entry.release_date)
            if entry.release_date
            else pd.NaT
        )
        releases[key] = timestamp
        releases[label] = timestamp
    return releases, aliases


def _fit_logistic(
    beta_values: np.ndarray,
    scores: np.ndarray,
    regularization: float,
) -> tuple[float, float]:
    beta_values = np.asarray(beta_values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if np.all(scores == scores[0]):
        raise ValueError("Cannot fit logistic regression with a single class.")

    def loss(theta: np.ndarray) -> float:
        intercept = theta[0]
        slope = theta[1]
        logits = intercept + slope * beta_values
        log_prob = (
            scores * -np.logaddexp(0, -logits)
            + (1.0 - scores) * -np.logaddexp(0, logits)
        )
        penalty = 0.5 * regularization * (slope**2)
        return -(np.sum(log_prob) - penalty)

    result = opt.minimize(loss, x0=np.zeros(2), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    intercept = float(result.x[0])
    slope = float(result.x[1])
    return intercept, slope


def _beta_at_quantile(intercept: float, slope: float, quantile: float) -> float:
    logit = np.log(quantile / (1.0 - quantile))
    if abs(slope) < 1e-8:
        return float("nan")
    return (logit - intercept) / slope


def _bootstrap_beta(
    beta_values: np.ndarray,
    scores: np.ndarray,
    n_draws: int,
    confidence: float,
    regularization: float,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(1337)
    n = len(beta_values)
    draws: list[float] = []
    for _ in range(n_draws):
        idx = rng.integers(0, n, size=n)
        sample_scores = scores[idx]
        if len(np.unique(sample_scores)) < 2:
            continue
        sample_betas = beta_values[idx]
        intercept, slope = _fit_logistic(sample_betas, sample_scores, regularization)
        draws.append(_beta_at_quantile(intercept, slope, 0.5))
    if not draws:
        return float("nan"), float("nan"), float("nan")
    low_q = max((1.0 - confidence) / 2.0, 0.0)
    high_q = 1.0 - low_q
    draws_arr = np.sort(np.asarray(draws))
    low = float(np.nanquantile(draws_arr, low_q))
    mid = float(np.nanmedian(draws_arr))
    high = float(np.nanquantile(draws_arr, high_q))
    return low, mid, high


@dataclass
class AgentSummary:
    agent_key: str
    label: str
    release_date: pd.Timestamp
    intercept: float
    slope: float
    beta_p50: float
    beta_low: float
    beta_high: float
    task_count: int


def summarize_agents(
    responses: pd.DataFrame,
    betas: pd.DataFrame,
    releases: dict[str, pd.Timestamp],
    aliases: dict[str, str],
    n_bootstrap: int,
    confidence: float,
    regularization: float,
) -> list[AgentSummary]:
    merged = responses.merge(betas, on="task_id", how="inner", validate="m:1")
    summaries: list[AgentSummary] = []
    for agent, agent_df in merged.groupby("agent"):
        scores = agent_df["score"].to_numpy()
        if len(np.unique(scores)) < 2:
            continue
        beta_values = agent_df["beta"].to_numpy()
        intercept, slope = _fit_logistic(beta_values, scores, regularization)
        beta_p50 = _beta_at_quantile(intercept, slope, 0.5)
        beta_low, _, beta_high = _bootstrap_beta(
            beta_values,
            scores,
            n_bootstrap,
            confidence,
            regularization,
        )
        release_date = releases.get(agent, pd.NaT)
        label = aliases.get(agent, agent)
        summaries.append(
            AgentSummary(
                agent_key=agent,
                label=label,
                release_date=release_date,
                intercept=intercept,
                slope=slope,
                beta_p50=beta_p50,
                beta_low=beta_low,
                beta_high=beta_high,
                task_count=len(agent_df),
            )
        )
    return summaries


def plot_release_curve(
    summary: pd.DataFrame,
    output_path: Path,
    linear_scale: bool,
) -> None:
    valid = summary.dropna(subset=["release_date", "beta_p50"])
    if valid.empty:
        raise ValueError("No models had both release dates and beta estimates.")
    valid = valid.sort_values("release_date")

    fig, ax = plt.subplots(figsize=(11, 7))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    label_colors = {
        label: next(color_cycle) for label in sorted(valid["label"].unique())
    }
    markers = {
        "claude": "o",
        "gpt": "^",
        "o": "^",
        "gemini": "s",
        "qwen": "D",
        "default": "P",
    }

    used_labels: set[str] = set()
    for _, row in valid.iterrows():
        name = row["label"]
        lower = row["beta_p50"] - row["beta_low"]
        upper = row["beta_high"] - row["beta_p50"]
        vendor = next(
            (key for key in markers if key != "default" and key in name.lower()),
            None,
        )
        color = label_colors.get(name, "#555555")
        marker = markers.get(vendor, markers["default"])
        label = name if name not in used_labels else ""
        ax.errorbar(
            row["release_date"],
            row["beta_p50"],
            yerr=np.array([[lower], [upper]]),
            fmt=marker,
            color=color,
            ecolor="#444444",
            elinewidth=1,
            capsize=3,
            label=label,
        )
        used_labels.add(name)

    ax.set_title("Beta difficulty solved at 50% success", fontsize=14)
    ax.set_xlabel("Release date")
    ax.set_ylabel("IRT β difficulty (higher is harder)")
    if linear_scale:
        ax.set_yscale("linear")
    else:
        betas = valid["beta_p50"].to_numpy()
        linthresh = max(0.1, 0.05 * np.nanmax(np.abs(betas)))
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    responses = load_pyirt_responses(args.responses)
    task_betas = load_task_betas(args.beta_params)
    releases, aliases = load_release_dates(args.model_mapping)

    summaries = summarize_agents(
        responses,
        task_betas,
        releases,
        aliases,
        args.n_bootstrap,
        args.confidence,
        args.regularization,
    )
    if not summaries:
        raise ValueError("No agents could be summarized from the provided files.")
    summary_df = pd.DataFrame([s.__dict__ for s in summaries])
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_table, index=False)

    plot_release_curve(summary_df, args.output_plot, args.linear_scale)
    print(f"Wrote summary to {args.output_table}")
    print(f"Wrote figure to {args.output_plot}")


if __name__ == "__main__":
    main()
