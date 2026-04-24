from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.cmaes import CMAESAlgorithm
from src.config import AppConfig, default_config_path, load_config, serialize_config
from src.genetic import GeneticAlgorithm
from src.hill import HillClimbingAlgorithm
from src.random_search import RandomSearchAlgorithm
from src.pipeline import (
    aggregate_generation_metrics,
    append_individual_rows,
    build_run_context,
    class_name,
    encode_image_to_latent,
    evaluate_fitness_sensitivity,
    export_legacy_metrics_csv,
    finalize_run_summary,
    get_classifier_logits_and_class,
    get_latent_stats,
    load_execution_inputs,
    make_logger,
    make_run_dir,
    maybe_empty_cuda_cache,
    save_artifacts_manifest,
    save_best_reconstruction,
    save_csv_dict_rows,
    save_single_row_csv,
    save_snapshot_grid,
    set_global_seed,
    snapshot_generations,
    write_records_to_parquet_or_csv,
)

ALGORITHM_REGISTRY = {
    "genetic": GeneticAlgorithm,
    "cmaes": CMAESAlgorithm,
    "hill": HillClimbingAlgorithm,
    "random_search": RandomSearchAlgorithm,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_config_path(Path(__file__).resolve().parent)))
    return parser


def resolve_algorithm(config: AppConfig):
    # Map the YAML algorithm name to the concrete search implementation.
    algorithm_name = config.common.algorithm
    if algorithm_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Options: {sorted(ALGORITHM_REGISTRY)}")
    return ALGORITHM_REGISTRY[algorithm_name]


def build_run_config_payload(config: AppConfig, optimizer_name: str, run_seed: int, run_sequence_idx: int, target_class: int, pred_class: int, p0: float, input_reference: str, dataset_index: int | None, instance_id: str, run_id: str, confidence_group: str, latent_shape: tuple[int, ...], latent_dim: int, algorithm_config: dict[str, Any]) -> dict[str, Any]:
    # Save both the user config and run-specific metadata for reproducibility.

    payload = serialize_config(config)
    payload["runtime"] = {
        "seed": run_seed,
        "run_sequence_idx": run_sequence_idx,
        "optimizer": optimizer_name,
        "predicted_class": pred_class,
        "predicted_class_name": class_name(pred_class),
        "orig_class": target_class,
        "orig_class_name": class_name(target_class),
        "p0": p0,
        "latent_shape": latent_shape,
        "latent_dim": latent_dim,
        "input_image_reference": input_reference,
        "dataset_index": dataset_index,
        "dataset_name": config.common.dataset_name if dataset_index is not None else None,
        "dataset_split": config.common.dataset_split if dataset_index is not None else None,
        "instance_id": instance_id,
        "run_id": run_id,
        "confidence_group": confidence_group,
    }
    payload["algorithm_runtime"] = algorithm_config
    return payload


def run_single_experiment( ctx, config: AppConfig, x0_pil, input_reference: str, instance_id: str, run_seed: int, run_sequence_idx: int, dataset_index: int | None) -> Path:
    # One run = one instance + one seed + one algorithm configuration.
    set_global_seed(run_seed)
    z0 = encode_image_to_latent(ctx, x0_pil)
    latent_shape, latent_dim, latent_dim_sqrt = get_latent_stats(z0)
    _, p0, pred_class = get_classifier_logits_and_class(ctx, x0_pil)
    target_class = pred_class if config.common.target_class < 0 else config.common.target_class
    print(f"Input image: {input_reference}")
    if dataset_index is not None:
        print(f"Dataset index: {dataset_index}")
    print(f"Predicted class: {class_name(pred_class)} (index {pred_class}), p0 = {p0:.4f}")
    print(f"Original class used in LEI-Local: {class_name(target_class)} (index {target_class})")
    run_dir, run_id, confidence_group = make_run_dir(config, instance_id, target_class, run_seed, run_sequence_idx)
    logger = make_logger(instance_id, run_id, run_dir)
    algorithm_cls = resolve_algorithm(config)
    algorithm = algorithm_cls(ctx, config, z0)
    id_counter = {"value": 0}
    algorithm_state = algorithm.initialize(latent_shape, id_counter)
    eval_counter = {"value": 0}
    run_config = build_run_config_payload(
        config=config,
        optimizer_name=algorithm.name,
        run_seed=run_seed,
        run_sequence_idx=run_sequence_idx,
        target_class=target_class,
        pred_class=pred_class,
        p0=p0,
        input_reference=input_reference,
        dataset_index=dataset_index,
        instance_id=instance_id,
        run_id=run_id,
        confidence_group=confidence_group,
        latent_shape=latent_shape,
        latent_dim=latent_dim,
        algorithm_config=algorithm.initial_config())
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    np.save(run_dir / "z0.npy", z0.detach().cpu().numpy().astype(np.float32))
    total_iterations = algorithm.total_iterations()
    snapshot_gens = snapshot_generations(total_iterations, config.common.n_snapshots)
    individual_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    grid_frames = []
    cumulative_valid_flip_targets: set[int] = set()
    consumed_evals = 0
    last_population = None
    last_eval_metrics = None
    last_generation = 0
    progress = tqdm(range(1, total_iterations + 1), desc="Iteracoes")
    for generation in progress:
        remaining_budget = None
        if algorithm.name == "hill":
            remaining_budget = config.hill.classifier_eval_budget_in_loop - consumed_evals
        elif algorithm.name == "random_search":
            remaining_budget = config.random_search.classifier_eval_budget_in_loop - consumed_evals
        if remaining_budget is not None and remaining_budget <= 0:
            break
        population, ask_info = algorithm.ask(algorithm_state, generation, latent_shape, id_counter, remaining_budget)
        eval_metrics = evaluate_fitness_sensitivity(ctx, population, z0, target_class, latent_dim_sqrt)
        fitness = eval_metrics["fitness_total"]
        mean_f = float(np.mean(fitness))
        best_f = float(np.max(fitness))
        std_f = float(np.std(fitness))
        frac_changed = float(np.mean(eval_metrics["changed_class"] == 1))
        logger.info(f"Iteration {generation:03d} - mean={mean_f:.4f}, best={best_f:.4f}, "
            f"std={std_f:.4f}, frac_changed={frac_changed:.3f}")
        append_individual_rows(
            ctx,
            rows=individual_rows,
            instance_id=instance_id,
            run_id=run_id,
            generation=generation,
            population=population,
            eval_metrics=eval_metrics,
            eval_counter=eval_counter,
            eval_stage="in_loop",
        )
        if generation % config.common.save_best_every == 0:
            best_idx_now = int(np.argmax(fitness))
            save_best_reconstruction(ctx, run_dir, f"best_gen_{generation}.png", population[best_idx_now].z)
        if generation in snapshot_gens:
            save_snapshot_grid(ctx, run_dir, grid_frames, population, fitness)
        flip_targets = eval_metrics["target_class_if_changed"][eval_metrics["changed_class"] == 1]
        flip_targets = flip_targets[flip_targets >= 0]
        if flip_targets.size > 0:
            cumulative_valid_flip_targets.update(np.unique(flip_targets).astype(int).tolist())
        step_info = algorithm.tell( algorithm_state, generation, population, fitness, ask_info, latent_shape, id_counter)
        gen_row = aggregate_generation_metrics(
            instance_id=instance_id,
            run_id=run_id,
            generation=generation,
            eval_metrics=eval_metrics,
            population=population,
            mutation_sigma_reference=float(step_info.get("mutation_sigma_reference", 0.0)),
            num_mutation_offspring=int(step_info.get("num_mutation_offspring", 0)),
            num_crossover_offspring=int(step_info.get("num_crossover_offspring", 0)),
            elite_count=int(step_info.get("elite_count", 0)),
            cumulative_unique_flips=len(cumulative_valid_flip_targets),
        )
        gen_row.update({k: v for k, v in step_info.items() if k not in gen_row})
        if algorithm.name in {"hill", "random_search"}:
            consumed_evals += int(step_info.get("batch_size", len(population)))
            gen_row["consumed_evals"] = int(consumed_evals)
        generation_rows.append(gen_row)
        last_population = population
        last_eval_metrics = eval_metrics
        last_generation = generation
        progress.set_postfix(mean=f"{mean_f:.3f}", best=f"{best_f:.3f}", frac_changed=f"{frac_changed:.2f}")
        maybe_empty_cuda_cache()
    if last_population is None or last_eval_metrics is None:
        raise RuntimeError("No evaluation was executed.")
    if algorithm.evaluate_final_population:
        final_population = algorithm.final_population(algorithm_state)
        final_metrics = evaluate_fitness_sensitivity(ctx, final_population, z0, target_class, latent_dim_sqrt)
        final_generation = total_iterations
    else:
        final_population = algorithm.final_population(algorithm_state)
        final_metrics = last_eval_metrics
        final_generation = last_generation
    append_individual_rows(ctx,
        rows=individual_rows,
        instance_id=instance_id,
        run_id=run_id,
        generation=final_generation,
        population=final_population,
        eval_metrics=final_metrics,
        eval_counter=eval_counter,
        eval_stage=algorithm.final_eval_stage)
    np.save(run_dir / "best_delta_z.npy", (final_population[int(np.argmax(final_metrics["fitness_total"]))].z - z0.to(ctx.device)).detach().cpu().numpy())
    save_best_reconstruction(ctx, run_dir, "best_final.png", final_population[int(np.argmax(final_metrics["fitness_total"]))].z)
    if grid_frames:
        grid_frames[0].save(run_dir / "gif_evolution.gif", save_all=True, append_images=grid_frames[1:], duration=500, loop=0)
    individual_log_path, individual_format = write_records_to_parquet_or_csv(individual_rows, run_dir / "individual_log.parquet")
    save_csv_dict_rows(generation_rows, run_dir / "generation_summary.csv")
    run_summary = finalize_run_summary(instance_id=instance_id, run_id=run_id, individual_rows=individual_rows, generation_rows=generation_rows)
    save_single_row_csv(run_summary, run_dir / "run_summary.csv")
    save_single_row_csv({
            "instance_id": instance_id,
            "run_id": run_id,
            "seed": run_seed,
            "run_sequence_idx": run_sequence_idx,
            "orig_class": target_class,
            "orig_class_name": class_name(target_class),
            "input_image_reference": input_reference,
            "dataset_index": dataset_index,
            "individual_log_path": str(individual_log_path),
            "individual_log_format": individual_format,
            "z0_path": str(run_dir / "z0.npy"),
            "generation_summary_path": str(run_dir / "generation_summary.csv"),
            "run_summary_path": str(run_dir / "run_summary.csv"),
        }, run_dir / "instance_summary.csv")
    export_legacy_metrics_csv(individual_rows, run_dir / "metrics_per_gen.csv")
    save_artifacts_manifest(run_dir, individual_log_path)
    return run_dir


def run_experiments(config: AppConfig) -> None:
    # The outer loop handles instances and repeated runs. Each algorithm owns only the search logic.
    ctx = build_run_context(config)
    print(f"Using device: {ctx.device}")
    jobs = load_execution_inputs(config)
    total_jobs = len(jobs) * config.common.runs_per_instance
    job_idx = 0
    for instance_pos, job in enumerate(jobs, start=1):
        for run_sequence_idx in range(1, config.common.runs_per_instance + 1):
            run_seed = config.common.base_seed + (job_idx * config.common.run_seed_increment)
            job_idx += 1
            print(f"\n[{job_idx}/{total_jobs}] Running instance {instance_pos}/{len(jobs)} "
                f"run={run_sequence_idx} seed={run_seed} algorithm={config.common.algorithm}")
            run_single_experiment(
                ctx=ctx,
                config=config,
                x0_pil=job["x0_pil"],
                input_reference=job["input_reference"],
                instance_id=job["instance_id"],
                run_seed=run_seed,
                run_sequence_idx=run_sequence_idx,
                dataset_index=job["dataset_index"])


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_experiments(config)


if __name__ == "__main__":
    main()
