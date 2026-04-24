from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "nateraw/vit-base-patch16-224-cifar10"
DATASET_NAME = "uoft-cs/cifar10"
SPLIT = "test"
SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "instances"
BATCH_SIZE = 64

@dataclass
class ExampleRecord:
    # Minimal prediction record used before grouping and sampling instances.
    dataset_index: int
    true_label: int
    true_label_name: str
    pred_label: int
    pred_label_name: str
    confidence: float
    margin: float
    correct: bool
    confidence_group: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_processor(model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser


def infer_dataset(dataset, processor, model, device, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    id2label = model.config.id2label
    rows: List[ExampleRecord] = []
    batch_starts = range(0, len(dataset), batch_size)

    for start in tqdm(batch_starts, desc="Running CIFAR-10 inference", total=(len(dataset) + batch_size - 1) // batch_size):
        batch = dataset[start : start + batch_size]
        images = batch["img"]
        true_labels = batch["label"]

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        pred_labels = probs.argmax(dim=-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

        for i, (y_true, y_pred) in enumerate(zip(true_labels, pred_labels)):
            global_idx = start + i

            confidence = float(probs_np[i, y_pred])

            sorted_probs = np.sort(probs_np[i])[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])

            rows.append(
                ExampleRecord(
                    dataset_index=global_idx,
                    true_label=int(y_true),
                    true_label_name=id2label[int(y_true)],
                    pred_label=int(y_pred),
                    pred_label_name=id2label[int(y_pred)],
                    confidence=confidence,
                    margin=margin,
                    correct=bool(y_true == y_pred),
                ))

    return pd.DataFrame([asdict(r) for r in rows])


def assign_confidence_groups_within_class(df_correct: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for cls in tqdm(sorted(df_correct["true_label"].unique()), desc="Assigning confidence groups"):
        cls_df = df_correct[df_correct["true_label"] == cls].copy()
        cls_df = cls_df.sort_values("confidence", ascending=True).reset_index(drop=True)

        n = len(cls_df)
        if n < 3:
            raise ValueError(f"Class {cls} has too few correct examples: {n}")

        low_end = n // 3
        mid_end = 2 * n // 3

        groups = []
        for idx in range(n):
            if idx < low_end:
                groups.append("low")
            elif idx < mid_end:
                groups.append("medium")
            else:
                groups.append("high")

        cls_df["confidence_group"] = groups
        cls_df["confidence_rank_within_class"] = np.arange(len(cls_df))
        cls_df["confidence_percentile_within_class"] = cls_df["confidence_rank_within_class"] / max(len(cls_df) - 1, 1)

        parts.append(cls_df)

    return pd.concat(parts, ignore_index=True)


def stratified_sample(df_grouped: pd.DataFrame, n_per_group_per_class: int = 5, seed: int = 42) -> pd.DataFrame:
    sampled_parts = []
    group_pairs = [
        (cls, group)
        for cls in sorted(df_grouped["true_label"].unique())
        for group in ["low", "medium", "high"]
        ]

    for cls, group in tqdm(group_pairs, desc="Stratified random sampling"):
        subset = df_grouped[(df_grouped["true_label"] == cls) & (df_grouped["confidence_group"] == group)].copy()

        if len(subset) < n_per_group_per_class:
            raise ValueError(
                f"Class {cls}, group {group}: "
                f"only {len(subset)} examples are available, "
                f"but you requested {n_per_group_per_class}.")

        sampled = subset.sample(n=n_per_group_per_class, random_state=seed, replace=False)
        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sort_values(["true_label", "confidence_group", "dataset_index"]).reset_index(drop=True)
    return sampled_df


def representative_sample_one_per_group(df_grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Selects 1 instance per class x confidence group.

    Criteria:
    1) smallest distance to the group's median confidence
    2) largest margin
    3) smallest dataset_index
    """
    selected_parts = []
    group_pairs = [
        (cls, group)
        for cls in sorted(df_grouped["true_label"].unique())
        for group in ["low", "medium", "high"]
    ]

    for cls, group in tqdm(group_pairs, desc="Selecting 1 representative per stratum"):
        subset = df_grouped[(df_grouped["true_label"] == cls) & (df_grouped["confidence_group"] == group)].copy()

        if len(subset) == 0:
            raise ValueError(f"No instance found for class {cls}, group {group}.")

        median_conf = subset["confidence"].median()
        subset["group_confidence_median"] = median_conf
        subset["distance_to_group_median"] = (subset["confidence"] - median_conf).abs()

        subset = subset.sort_values(by=["distance_to_group_median", "margin", "dataset_index"], ascending=[True, False, True]).reset_index(drop=True)

        chosen = subset.iloc[[0]].copy()
        selected_parts.append(chosen)

    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df = selected_df.sort_values(["true_label", "confidence_group", "dataset_index"]).reset_index(drop=True)
    return selected_df


def main() -> None:
    # Generate the CSV files that the main experiment consumes in batch mode.
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, split=args.split)
    processor, model, device = load_model_and_processor(args.model_name)

    df = infer_dataset(dataset, processor, model, device, batch_size=args.batch_size)
    df_correct = df[df["correct"]].copy().reset_index(drop=True)
    df_grouped = assign_confidence_groups_within_class(df_correct)

    selected = stratified_sample(df_grouped, n_per_group_per_class=5, seed=args.seed)
    selected_representative_1 = representative_sample_one_per_group(df_grouped)

    all_predictions_path = output_dir / "cifar10_all_predictions.csv"
    correct_predictions_path = output_dir / "cifar10_correct_predictions.csv"
    grouped_path = output_dir / "cifar10_correct_grouped_by_confidence.csv"
    selected_path = output_dir / "cifar10_selected_instances_representative_5.csv"
    representative_path = output_dir / "cifar10_selected_instances_representative_1.csv"

    df.to_csv(all_predictions_path, index=False)
    df_correct.to_csv(correct_predictions_path, index=False)
    df_grouped.to_csv(grouped_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected_representative_1.to_csv(representative_path, index=False)

    print("Total test examples:", len(df))
    print("Correctly classified only:", len(df_correct))

    print("\nRandomly selected per class x confidence (5 per stratum):")
    print(selected.groupby(["true_label_name", "confidence_group"]).size().unstack(fill_value=0))

    print("\nRepresentative selections per class x confidence (1 per stratum):")
    print(selected_representative_1.groupby(["true_label_name", "confidence_group"]).size().unstack(fill_value=0))

    print("\nSaved files:")
    print(f"- {all_predictions_path}")
    print(f"- {correct_predictions_path}")
    print(f"- {grouped_path}")
    print(f"- {selected_path}")
    print(f"- {representative_path}")


if __name__ == "__main__":
    main()
