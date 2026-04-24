from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .config import AppConfig

try:
    import pandas as pd
except Exception:
    pd = None

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


@dataclass
class Individual:
    # Shared candidate representation across all search algorithms.
    individual_id: int
    z: torch.Tensor
    created_by: str
    parent_ids: str
    mutation_sigma: Optional[float]
    birth_generation: int


@dataclass
class RunContext:
    # Loaded runtime objects reused across all runs in one process.
    config: AppConfig
    device: torch.device
    vae: AutoencoderKL
    feature_extractor: AutoImageProcessor
    clf_model: AutoModelForImageClassification
    vae_scaling_factor: float
    sample_size: int
    to_tensor_01: Any
    to_pil_image: Any

def set_global_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def resolve_torch_device() -> torch.device:
    requested_device_raw = os.environ.get("LEI_CUDA_DEVICE", "0").strip()
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if requested_device_raw.lower() == "cpu":
        return torch.device("cpu")
    try:
        requested_index = int(requested_device_raw)
    except ValueError:
        requested_index = 0
    requested_index = min(max(requested_index, 0), torch.cuda.device_count() - 1)
    return torch.device(f"cuda:{requested_index}")


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_vae_with_cache_fallback(model_name: str, prefer_local_cache: bool) -> AutoencoderKL:
    if prefer_local_cache:
        try:
            return AutoencoderKL.from_pretrained(model_name, local_files_only=True)
        except Exception:
            pass
    return AutoencoderKL.from_pretrained(model_name)


def load_processor_with_cache_fallback(model_name: str, prefer_local_cache: bool, size: int = 224, use_fast: bool = True) -> AutoImageProcessor:
    if prefer_local_cache:
        try:
            return AutoImageProcessor.from_pretrained(
                model_name,
                size=size,
                use_fast=use_fast,
                local_files_only=True,
            )
        except Exception:
            pass
    return AutoImageProcessor.from_pretrained(model_name, size=size, use_fast=use_fast)


def load_classifier_with_cache_fallback(model_name: str, prefer_local_cache: bool) -> AutoModelForImageClassification:
    if prefer_local_cache:
        try:
            return AutoModelForImageClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception:
            pass
    return AutoModelForImageClassification.from_pretrained(model_name)


def build_run_context(config: AppConfig) -> RunContext:
    # Load the VAE and classifier once and keep them available for the whole execution.
    device = resolve_torch_device()
    vae = load_vae_with_cache_fallback(config.common.vae_model_name, config.common.prefer_local_hf_cache).to(device)
    vae.eval()
    feature_extractor = load_processor_with_cache_fallback(config.common.classifier_model_name, config.common.prefer_local_hf_cache, size=224, use_fast=True)
    clf_model = load_classifier_with_cache_fallback(config.common.classifier_model_name, config.common.prefer_local_hf_cache).to(device)
    clf_model.eval()
    return RunContext(
        config=config,
        device=device,
        vae=vae,
        feature_extractor=feature_extractor,
        clf_model=clf_model,
        vae_scaling_factor=float(getattr(vae.config, "scaling_factor", 1.0)),
        sample_size=int(vae.config.sample_size),
        to_tensor_01=transforms.ToTensor(),
        to_pil_image=transforms.ToPILImage())


def next_individual_id(counter: dict[str, int]) -> int:
    counter["value"] += 1
    return counter["value"]


def prepare_classifier_inputs(ctx: RunContext, images: Any) -> torch.Tensor:
    inputs = ctx.feature_extractor(images=images, return_tensors="pt")
    return inputs["pixel_values"].to(ctx.device)


def latent_batch_to_pil(ctx: RunContext, batch_z: torch.Tensor) -> list[Image.Image]:
    with torch.inference_mode():
        recon = ctx.vae.decode(batch_z / ctx.vae_scaling_factor).sample
    recon = (recon.clamp(-1, 1) + 1.0) / 2.0
    recon_cpu = recon.cpu()
    pil_list: list[Image.Image] = []
    for i in range(recon_cpu.shape[0]):
        pil_32 = ctx.to_pil_image(recon_cpu[i])
        pil_list.append(pil_32.resize((224, 224)))
    del recon, recon_cpu
    return pil_list


def encode_image_to_latent(ctx: RunContext, pil_img: Image.Image) -> torch.Tensor:
    pil_32 = pil_img.resize((ctx.sample_size, ctx.sample_size))
    x = ctx.to_tensor_01(pil_32).unsqueeze(0).to(ctx.device)
    x = 2.0 * x - 1.0
    with torch.inference_mode():
        posterior = ctx.vae.encode(x)
        z = posterior.latent_dist.mean * ctx.vae_scaling_factor
    return z.squeeze(0)


def get_latent_stats(z: torch.Tensor) -> tuple[tuple[int, ...], int, float]:
    latent_shape = tuple(int(dim) for dim in z.shape)
    latent_dim = int(z.numel())
    latent_dim_sqrt = float(np.sqrt(latent_dim))
    return latent_shape, latent_dim, latent_dim_sqrt


def get_classifier_logits_and_class(ctx: RunContext, pil_img: Image.Image) -> tuple[torch.Tensor, float, int]:
    x = prepare_classifier_inputs(ctx, pil_img)
    with torch.inference_mode():
        outputs = ctx.clf_model(x)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=1)
    logits0 = logits[0].detach().cpu()
    pred_class = int(torch.argmax(probs, dim=1)[0].item())
    p0 = float(probs[0, pred_class].item())
    del logits, probs, x
    maybe_empty_cuda_cache()
    return logits0, p0, pred_class


def evaluate_fitness_sensitivity(ctx: RunContext, population: list[Individual], z0: torch.Tensor, orig_class: int, latent_dim_sqrt: float) -> dict[str, np.ndarray]:
    # Shared LEI-Local scoring function used to compare all algorithms under the same metric.
    n = len(population)
    config = ctx.config.common
    metrics: dict[str, np.ndarray] = {
        "fitness_total": np.zeros(n, dtype=np.float32),
        "margin_logit": np.zeros(n, dtype=np.float32),
        "dist_norm": np.zeros(n, dtype=np.float32),
        "dist_l2": np.zeros(n, dtype=np.float32),
        "prob_original_class": np.zeros(n, dtype=np.float32),
        "prob_best_alt_class": np.zeros(n, dtype=np.float32),
        "pred_class": np.zeros(n, dtype=np.int32),
        "target_class_if_changed": np.full(n, -1, dtype=np.int32),
        "logit_original": np.zeros(n, dtype=np.float32),
        "logit_best_alt": np.zeros(n, dtype=np.float32),
        "fitness_margin_term": np.zeros(n, dtype=np.float32),
        "fitness_distance_penalty": np.zeros(n, dtype=np.float32),
        "fitness_constraint_penalty": np.zeros(n, dtype=np.float32),
        "constraint_violation": np.zeros(n, dtype=np.float32),
        "within_confidence_region": np.zeros(n, dtype=np.int32),
        "changed_class": np.zeros(n, dtype=np.int32),
    }
    z0 = z0.to(ctx.device)
    z0_flat = z0.view(-1)
    for start in range(0, n, config.batch_eval_size):
        batch_individuals = population[start:start + config.batch_eval_size]
        batch_z = torch.stack([ind.z for ind in batch_individuals], dim=0)
        pil_imgs = latent_batch_to_pil(ctx, batch_z)
        inputs = prepare_classifier_inputs(ctx, pil_imgs)
        with torch.inference_mode():
            outputs = ctx.clf_model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=1)
        logit_orig = logits[:, orig_class]
        logits_others = logits.clone()
        logits_others[:, orig_class] = -1e9
        logit_other_max, idx_other_max = torch.max(logits_others, dim=1)
        margin = logit_other_max - logit_orig
        p_orig = probs[:, orig_class]
        p_other_max = probs.gather(1, idx_other_max.unsqueeze(1)).squeeze(1)
        flat = batch_z.view(batch_z.size(0), -1)
        z0_batch = z0_flat.unsqueeze(0).expand_as(flat)
        diff = flat - z0_batch
        dist_l2 = torch.norm(diff, dim=1)
        dist_norm = dist_l2 / latent_dim_sqrt
        over_radius = torch.clamp(dist_norm - config.trust_region_radius, min=0.0)
        pred_classes = torch.argmax(probs, dim=1)
        changed_mask = (pred_classes != orig_class).float()
        same_mask = 1.0 - changed_mask
        margin_term_before = config.margin_alpha_before * margin
        margin_term_after = -config.margin_gamma_after * torch.relu(margin)
        margin_term = same_mask * margin_term_before + changed_mask * margin_term_after
        distance_penalty_before = config.dist_beta_before * dist_norm
        distance_penalty_after = config.dist_gamma_after * dist_norm
        distance_penalty = same_mask * distance_penalty_before + changed_mask * distance_penalty_after
        constraint_penalty = config.trust_region_penalty * over_radius
        batch_fitness = margin_term - distance_penalty - constraint_penalty
        bsz = batch_z.size(0)
        end = start + bsz
        pred_np = pred_classes.detach().cpu().numpy().astype(np.int32)
        changed_np = (pred_np != orig_class).astype(np.int32)
        alt_np = idx_other_max.detach().cpu().numpy().astype(np.int32)
        alt_np = np.where(changed_np == 1, alt_np, -1)
        metrics["fitness_total"][start:end] = batch_fitness.detach().cpu().numpy()
        metrics["margin_logit"][start:end] = margin.detach().cpu().numpy()
        metrics["dist_norm"][start:end] = dist_norm.detach().cpu().numpy()
        metrics["dist_l2"][start:end] = dist_l2.detach().cpu().numpy()
        metrics["prob_original_class"][start:end] = p_orig.detach().cpu().numpy()
        metrics["prob_best_alt_class"][start:end] = p_other_max.detach().cpu().numpy()
        metrics["pred_class"][start:end] = pred_np
        metrics["target_class_if_changed"][start:end] = alt_np
        metrics["logit_original"][start:end] = logit_orig.detach().cpu().numpy()
        metrics["logit_best_alt"][start:end] = logit_other_max.detach().cpu().numpy()
        metrics["fitness_margin_term"][start:end] = margin_term.detach().cpu().numpy()
        metrics["fitness_distance_penalty"][start:end] = distance_penalty.detach().cpu().numpy()
        metrics["fitness_constraint_penalty"][start:end] = constraint_penalty.detach().cpu().numpy()
        metrics["constraint_violation"][start:end] = over_radius.detach().cpu().numpy()
        metrics["within_confidence_region"][start:end] = (over_radius.detach().cpu().numpy() <= 0).astype(np.int32)
        metrics["changed_class"][start:end] = changed_np
        del batch_z, pil_imgs, inputs, outputs, logits, probs
        maybe_empty_cuda_cache()
    return metrics


def sanitize_for_path(raw: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
    return sanitized.strip("._") or "item"


def tensor_fingerprint(z: torch.Tensor) -> str:
    payload = z.detach().cpu().numpy().astype(np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def compact_z_payload(ctx: RunContext, z: torch.Tensor) -> tuple[float, float, float, str]:
    z_cpu = z.detach().cpu().view(-1).numpy().astype(np.float32)
    head = z_cpu[: ctx.config.common.z_vector_head_size].tolist()
    return (
        float(np.mean(z_cpu)),
        float(np.std(z_cpu)),
        float(np.linalg.norm(z_cpu)),
        json.dumps(head),
    )


def maybe_full_z_payload(ctx: RunContext, z: torch.Tensor) -> Optional[str]:
    if not ctx.config.common.save_full_z_vectors:
        return None
    z_cpu = z.detach().cpu().view(-1).numpy().astype(np.float32).tolist()
    return json.dumps(z_cpu)


def compute_population_diversity(population: list[Individual]) -> tuple[float, float]:
    if len(population) < 2:
        return 0.0, 0.0
    flat = np.stack([ind.z.detach().cpu().view(-1).numpy().astype(np.float32) for ind in population], axis=0)
    centroid = np.mean(flat, axis=0, keepdims=True)
    dist_centroid = np.linalg.norm(flat - centroid, axis=1)
    diffs = flat[:, None, :] - flat[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    upper = dists[np.triu_indices(len(population), k=1)]
    return float(np.mean(upper)), float(np.mean(dist_centroid))


def aggregate_generation_metrics(instance_id: str, run_id: str, generation: int, eval_metrics: dict[str, np.ndarray], population: list[Individual], mutation_sigma_reference: float, num_mutation_offspring: int, num_crossover_offspring: int, elite_count: int, cumulative_unique_flips: int) -> dict[str, Any]:
    fitness = eval_metrics["fitness_total"]
    margins = eval_metrics["margin_logit"]
    changed = eval_metrics["changed_class"]
    dist_norm = eval_metrics["dist_norm"]
    flips_mask = changed == 1
    flip_dists = dist_norm[flips_mask]
    valid_flip_targets = eval_metrics["target_class_if_changed"][flips_mask]
    valid_flip_targets = valid_flip_targets[valid_flip_targets >= 0]
    num_unique_valid_flips = int(np.unique(valid_flip_targets).size) if valid_flip_targets.size > 0 else 0
    pairwise_mean, centroid_mean = compute_population_diversity(population)
    return {
        "instance_id": instance_id,
        "run_id": run_id,
        "generation": generation,
        "population_size": int(len(population)),
        "best_fitness": float(np.max(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "median_fitness": float(np.median(fitness)),
        "std_fitness": float(np.std(fitness)),
        "best_margin": float(np.max(margins)),
        "mean_margin": float(np.mean(margins)),
        "fraction_margin_positive": float(np.mean(margins > 0)),
        "num_flips": int(np.sum(flips_mask)),
        "flip_rate": float(np.mean(flips_mask)),
        "num_unique_valid_flips": num_unique_valid_flips,
        "cumulative_unique_flips": int(cumulative_unique_flips),
        "best_flip_distance": float(np.min(flip_dists)) if flip_dists.size > 0 else float("nan"),
        "mean_flip_distance": float(np.mean(flip_dists)) if flip_dists.size > 0 else float("nan"),
        "mean_dist_norm": float(np.mean(dist_norm)),
        "median_dist_norm": float(np.median(dist_norm)),
        "std_dist_norm": float(np.std(dist_norm)),
        "fraction_outside_region": float(np.mean(eval_metrics["within_confidence_region"] == 0)),
        "mean_pairwise_latent_distance": pairwise_mean,
        "distance_to_centroid_mean": centroid_mean,
        "mutation_sigma": float(mutation_sigma_reference),
        "num_mutation_offspring": int(num_mutation_offspring),
        "num_crossover_offspring": int(num_crossover_offspring),
        "elite_count": int(elite_count),
    }


def compute_stagnation_length(best_fitness_per_gen: list[float]) -> int:
    if not best_fitness_per_gen:
        return 0
    best_so_far = -float("inf")
    last_improvement_idx = 0
    for idx, val in enumerate(best_fitness_per_gen):
        if val > best_so_far:
            best_so_far = val
            last_improvement_idx = idx
    return len(best_fitness_per_gen) - 1 - last_improvement_idx


def finalize_run_summary(instance_id: str, run_id: str, individual_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_evals = len(individual_rows)
    total_generations = len(generation_rows)
    found_flip = any(int(r["changed_class"]) == 1 for r in individual_rows)
    first_flip_row = None
    if found_flip:
        for row in individual_rows:
            if int(row["changed_class"]) == 1:
                first_flip_row = row
                break
    num_total_flips = sum(int(r["changed_class"]) == 1 for r in individual_rows)
    unique_target_classes = {int(r["target_class_if_changed"]) for r in individual_rows if int(r["changed_class"]) == 1 and int(r["target_class_if_changed"]) >= 0}
    best_fitness_row = max(individual_rows, key=lambda r: float(r["fitness_total"])) if individual_rows else None
    best_margin_row = max(individual_rows, key=lambda r: float(r["margin_logit"])) if individual_rows else None
    flip_rows = [r for r in individual_rows if int(r["changed_class"]) == 1]
    unique_flip_hashes = {str(r["z_hash"]) for r in flip_rows if r.get("z_hash")}
    num_unique_flips = int(len(unique_flip_hashes))
    fraction_unique_flips = float(num_unique_flips / num_total_flips) if num_total_flips > 0 else 0.0
    best_flip_row = max(flip_rows, key=lambda r: float(r["fitness_total"])) if flip_rows else None
    max_margin_flip_row = max(flip_rows, key=lambda r: float(r["margin_logit"])) if flip_rows else None
    diversity_start = float("nan")
    diversity_mid = float("nan")
    diversity_end = float("nan")
    if generation_rows:
        diversity_start = float(generation_rows[0]["mean_pairwise_latent_distance"])
        diversity_mid = float(generation_rows[len(generation_rows) // 2]["mean_pairwise_latent_distance"])
        diversity_end = float(generation_rows[-1]["mean_pairwise_latent_distance"])
    best_fitness_per_gen = [float(r["best_fitness"]) for r in generation_rows]
    stagnation_length = compute_stagnation_length(best_fitness_per_gen)
    improvement_last_10pct = float("nan")
    if generation_rows:
        split_idx = max(1, int(math.floor(0.9 * len(generation_rows))))
        first_slice = best_fitness_per_gen[:split_idx]
        last_slice = best_fitness_per_gen[split_idx:]
        if first_slice and last_slice:
            improvement_last_10pct = float(max(last_slice) - max(first_slice))
    return {
        "instance_id": instance_id,
        "run_id": run_id,
        "total_evals": int(total_evals),
        "total_generations": int(total_generations),
        "found_flip": int(bool(found_flip)),
        "first_flip_eval": int(first_flip_row["eval_id"]) if first_flip_row else None,
        "first_flip_generation": int(first_flip_row["generation"]) if first_flip_row else None,
        "evals_to_first_flip": int(first_flip_row["eval_id"]) if first_flip_row else None,
        "num_total_flips": int(num_total_flips),
        "num_unique_flips": int(num_unique_flips),
        "fraction_unique_flips": fraction_unique_flips,
        "num_unique_target_classes": int(len(unique_target_classes)),
        "best_fitness_ever": float(best_fitness_row["fitness_total"]) if best_fitness_row else float("nan"),
        "best_margin_ever": float(best_margin_row["margin_logit"]) if best_margin_row else float("nan"),
        "generation_of_best_fitness": int(best_fitness_row["generation"]) if best_fitness_row else None,
        "generation_of_best_margin": int(best_margin_row["generation"]) if best_margin_row else None,
        "best_flip_distance": float(best_flip_row["dist_norm"]) if best_flip_row else float("nan"),
        "best_flip_margin": float(best_flip_row["margin_logit"]) if best_flip_row else float("nan"),
        "best_flip_eval": int(best_flip_row["eval_id"]) if best_flip_row else None,
        "best_flip_generation": int(best_flip_row["generation"]) if best_flip_row else None,
        "best_flip_lpips": float(best_flip_row["lpips"]) if best_flip_row else float("nan"),
        "first_flip_distance": float(first_flip_row["dist_norm"]) if first_flip_row else float("nan"),
        "first_flip_margin": float(first_flip_row["margin_logit"]) if first_flip_row else float("nan"),
        "max_margin_flip_distance": float(max_margin_flip_row["dist_norm"]) if max_margin_flip_row else float("nan"),
        "diversity_start": diversity_start,
        "diversity_mid": diversity_mid,
        "diversity_end": diversity_end,
        "stagnation_length": int(stagnation_length),
        "fitness_improvement_last_10pct_budget": improvement_last_10pct,
    }


def append_individual_rows(ctx: RunContext, rows: list[dict[str, Any]], instance_id: str, run_id: str, generation: int, population: list[Individual], eval_metrics: dict[str, np.ndarray], eval_counter: dict[str, int], eval_stage: str) -> None:
    for idx, ind in enumerate(population):
        eval_counter["value"] += 1
        eval_id = eval_counter["value"]
        z_mean, z_std, z_norm, z_head_json = compact_z_payload(ctx, ind.z)
        z_full = maybe_full_z_payload(ctx, ind.z)
        rows.append({
                "instance_id": instance_id,
                "run_id": run_id,
                "generation": int(generation),
                "eval_stage": eval_stage,
                "eval_id": int(eval_id),
                "individual_id": int(ind.individual_id),
                "parent_ids": ind.parent_ids,
                "created_by": ind.created_by,
                "pred_class": int(eval_metrics["pred_class"][idx]),
                "changed_class": int(eval_metrics["changed_class"][idx]),
                "target_class_if_changed": int(eval_metrics["target_class_if_changed"][idx]),
                "prob_original_class": float(eval_metrics["prob_original_class"][idx]),
                "prob_best_alt_class": float(eval_metrics["prob_best_alt_class"][idx]),
                "logit_original": float(eval_metrics["logit_original"][idx]),
                "logit_best_alt": float(eval_metrics["logit_best_alt"][idx]),
                "margin_logit": float(eval_metrics["margin_logit"][idx]),
                "fitness_total": float(eval_metrics["fitness_total"][idx]),
                "fitness_margin_term": float(eval_metrics["fitness_margin_term"][idx]),
                "fitness_distance_penalty": float(eval_metrics["fitness_distance_penalty"][idx]),
                "fitness_constraint_penalty": float(eval_metrics["fitness_constraint_penalty"][idx]),
                "dist_l2": float(eval_metrics["dist_l2"][idx]),
                "dist_norm": float(eval_metrics["dist_norm"][idx]),
                "within_confidence_region": int(eval_metrics["within_confidence_region"][idx]),
                "constraint_violation": float(eval_metrics["constraint_violation"][idx]),
                "lpips": float("nan"),
                "z_hash": tensor_fingerprint(ind.z),
                "z_mean": z_mean,
                "z_std": z_std,
                "z_l2_norm": z_norm,
                "z_head": z_head_json,
                "z_vector": z_full,
                "mutation_sigma": float(ind.mutation_sigma) if ind.mutation_sigma is not None else float("nan"),
                "birth_generation": int(ind.birth_generation),
                "reconstruction_ref": None,
            })


def write_records_to_parquet_or_csv(records: list[dict[str, Any]], parquet_path: Path) -> tuple[Path, str]:
    if pd is None:
        csv_path = parquet_path.with_suffix(".csv")
        if records:
            keys = list(records[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
        return csv_path, "csv"
    df = pd.DataFrame.from_records(records)
    df.to_parquet(parquet_path, index=False)
    return parquet_path, "parquet"


def save_csv_dict_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_single_row_csv(row: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def export_legacy_metrics_csv(individual_rows: list[dict[str, Any]], out_path: Path) -> None:
    keys = [
        "generation",
        "index",
        "fitness",
        "margin_logit",
        "dist_norm",
        "dist_raw",
        "p_orig",
        "p_other_max",
        "pred_class",
        "changed_class",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in individual_rows:
            if row.get("eval_stage") != "in_loop":
                continue
            grouped.setdefault(int(row["generation"]), []).append(row)
        for generation in sorted(grouped.keys()):
            rows = grouped[generation]
            for idx, row in enumerate(rows):
                writer.writerow({
                        "generation": generation,
                        "index": idx,
                        "fitness": row["fitness_total"],
                        "margin_logit": row["margin_logit"],
                        "dist_norm": row["dist_norm"],
                        "dist_raw": row["dist_l2"],
                        "p_orig": row["prob_original_class"],
                        "p_other_max": row["prob_best_alt_class"],
                        "pred_class": row["pred_class"],
                        "changed_class": row["changed_class"],
                    })


def class_name(class_idx: int) -> str:
    return CIFAR10_CLASSES.get(class_idx, f"class_{class_idx}")


def load_instance_specs_from_csv(csv_path: str) -> list[dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def dataset_item_to_pil_image(dataset_item: Any) -> Image.Image:
    image = dataset_item["img"] if isinstance(dataset_item, dict) and "img" in dataset_item else dataset_item["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.array(image)).convert("RGB")


def make_run_dir(config: AppConfig, instance_id: str, target_class: int, run_seed: int, run_sequence_idx: int) -> tuple[Path, str, str]:
    timestamp = datetime.datetime.now().isoformat(timespec="microseconds").replace(":", "-")
    confidence_group = f"trr_{config.common.trust_region_radius:.2f}"
    run_id = f"{timestamp}_r{run_sequence_idx:03d}"
    run_dir = (
        Path(config.common.output_base)
        / f"instance={instance_id}"
        / f"class={target_class}"
        / f"confidence={confidence_group}"
        / f"seed={run_seed}"
        / f"run={run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id, confidence_group


def make_logger(instance_id: str, run_id: str, run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"evolution.{instance_id}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    file_handler = logging.FileHandler(run_dir / "run.log")
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def snapshot_generations(total_iterations: int, n_snapshots: int) -> list[int]:
    gens_lin = list(np.linspace(1, total_iterations, n_snapshots, dtype=int))
    return sorted(set(gens_lin + [total_iterations]))


def save_best_reconstruction(ctx: RunContext, run_dir: Path, filename: str, z_tensor: torch.Tensor) -> None:
    with torch.inference_mode():
        recon = ctx.vae.decode(z_tensor.unsqueeze(0) / ctx.vae_scaling_factor).sample
    recon = (recon.clamp(-1, 1) + 1.0) / 2.0
    recon_cpu = recon.cpu().squeeze(0)
    ctx.to_pil_image(recon_cpu).resize((224, 224)).save(run_dir / filename)
    del recon, recon_cpu
    maybe_empty_cuda_cache()


def save_snapshot_grid(ctx: RunContext, run_dir: Path, grid_frames: list[Image.Image], population: list[Individual], fitness: np.ndarray) -> None:
    sorted_idxs = np.argsort(-fitness)
    indices = np.linspace(0, len(sorted_idxs) - 1, ctx.config.common.k_grid, dtype=int)
    n_cols = int(np.ceil(np.sqrt(ctx.config.common.k_grid)))
    n_rows = int(np.ceil(ctx.config.common.k_grid / n_cols))
    width, height = 224, 224
    grid_img = Image.new("RGB", (n_cols * width, n_rows * height))
    selected = [population[idx].z for idx in sorted_idxs[indices]]
    batch_z = torch.stack(selected, dim=0).to(ctx.device)
    pil_imgs = latent_batch_to_pil(ctx, batch_z)
    for j, pil in enumerate(pil_imgs):
        row, col = divmod(j, n_cols)
        grid_img.paste(pil, (col * width, row * height))
    grid_frames.append(grid_img)
    del batch_z, pil_imgs, grid_img
    maybe_empty_cuda_cache()


def save_artifacts_manifest(run_dir: Path, individual_log_path: Path) -> None:
    manifest = {
        "where": str(run_dir),
        "files": {
            "individual_log": str(individual_log_path),
            "z0": str(run_dir / "z0.npy"),
            "generation_summary": str(run_dir / "generation_summary.csv"),
            "run_summary": str(run_dir / "run_summary.csv"),
            "instance_summary": str(run_dir / "instance_summary.csv"),
            "legacy_metrics": str(run_dir / "metrics_per_gen.csv"),
        }}
    with (run_dir / "artifacts_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

        
def load_execution_inputs(config: AppConfig) -> list[dict[str, Any]]:
    # Normalize batch-mode CSV inputs and single-image mode into one common job list.
    if config.common.use_dataset_index_batch:
        instance_specs = load_instance_specs_from_csv(config.common.instances_csv_path)
        if config.common.instance_limit is not None:
            instance_specs = instance_specs[: config.common.instance_limit]
        dataset = load_dataset(config.common.dataset_name, split=config.common.dataset_split)
        jobs: list[dict[str, Any]] = []
        for spec in instance_specs:
            dataset_index = int(spec["dataset_index"])
            x0_pil = dataset_item_to_pil_image(dataset[dataset_index])
            instance_id = sanitize_for_path(f"{config.common.dataset_name}_{config.common.dataset_split}_idx_{dataset_index:05d}")
            jobs.append({
                    "x0_pil": x0_pil,
                    "input_reference": f"{config.common.dataset_name}:{config.common.dataset_split}:{dataset_index}",
                    "instance_id": instance_id,
                    "dataset_index": dataset_index,
                })
        return jobs
    if not config.common.input_image_path:
        raise ValueError(
            "input_image_path must be set when use_dataset_index_batch is false."
        )
    input_path = Path(config.common.input_image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found at {input_path}")
    return [{
            "x0_pil": Image.open(input_path).convert("RGB"),
            "input_reference": str(input_path),
            "instance_id": sanitize_for_path(input_path.stem),
            "dataset_index": None,
        }]
