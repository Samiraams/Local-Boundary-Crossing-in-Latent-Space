from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CommonConfig:
    algorithm: str = "genetic"
    base_seed: int = 420
    num_generations: int = 200
    population_size: int = 100
    batch_eval_size: int = 8
    pixel_perturb_std: float = 0.30
    sigma_init: float = 1.0
    sigma_local: float = 0.20
    sigma_min: float = 1e-6
    sigma_max: float = 3.0
    k_grid: int = 25
    n_snapshots: int = 25
    save_best_every: int = 5
    output_base: Optional[str] = None
    input_image_path: Optional[str] = None
    use_dataset_index_batch: bool = True
    instances_csv_path: str = "instances/cifar10_selected_instances_representative_1.csv"
    dataset_name: str = "uoft-cs/cifar10"
    dataset_split: str = "test"
    runs_per_instance: int = 5
    run_seed_increment: int = 1
    instance_limit: Optional[int] = None
    target_class: int = -1
    margin_alpha_before: float = 1.0
    dist_beta_before: float = 0.3
    dist_gamma_after: float = 2.0
    margin_gamma_after: float = 0.2
    trust_region_radius: float = 0.75
    trust_region_penalty: float = 2.0
    save_reconstruction_path_in_log: bool = True
    save_full_z_vectors: bool = True
    z_vector_head_size: int = 16
    prefer_local_hf_cache: bool = True
    vae_model_name: str = "stabilityai/sd-vae-ft-ema"
    classifier_model_name: str = "nateraw/vit-base-patch16-224-cifar10"


@dataclass
class GeneticConfig:
    elitism: int = 1
    prob_mutation: float = 0.95
    prob_crossover: float = 0.05


@dataclass
class CMAESConfig:
    pass


@dataclass
class HillConfig:
    sigma_up_factor: float = 1.05
    sigma_down_factor: float = 0.95
    classifier_eval_budget_in_loop: Optional[int] = None


@dataclass
class RandomSearchConfig:
    classifier_eval_budget_in_loop: Optional[int] = None


@dataclass
class AppConfig:
    common: CommonConfig
    genetic: GeneticConfig
    cmaes: CMAESConfig
    hill: HillConfig
    random_search: RandomSearchConfig

    def validate(self) -> None:
        if self.common.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.common.num_generations <= 0:
            raise ValueError("num_generations must be positive.")
        if self.common.batch_eval_size <= 0:
            raise ValueError("batch_eval_size must be positive.")
        if self.common.k_grid <= 0:
            raise ValueError("k_grid must be positive.")
        if self.common.n_snapshots <= 0:
            raise ValueError("n_snapshots must be positive.")
        if self.common.save_best_every <= 0:
            raise ValueError("save_best_every must be positive.")
        if self.common.runs_per_instance <= 0:
            raise ValueError("runs_per_instance must be positive.")
        if self.common.run_seed_increment <= 0:
            raise ValueError("run_seed_increment must be positive.")
        if self.genetic.elitism < 0 or self.genetic.elitism >= self.common.population_size:
            raise ValueError("elitism must be between 0 and population_size - 1.")
        if not 0.0 <= self.genetic.prob_mutation <= 1.0:
            raise ValueError("prob_mutation must be between 0 and 1.")
        if not 0.0 <= self.genetic.prob_crossover <= 1.0:
            raise ValueError("prob_crossover must be between 0 and 1.")
        if self.hill.classifier_eval_budget_in_loop is None:
            self.hill.classifier_eval_budget_in_loop = (
                self.common.num_generations * self.common.population_size
            )
        if self.random_search.classifier_eval_budget_in_loop is None:
            self.random_search.classifier_eval_budget_in_loop = (
                self.common.num_generations * self.common.population_size
            )


DEFAULT_OUTPUT_BASES = {
    "genetic": "outputs/genetic",
    "cmaes": "outputs/cmaes",
    "hill": "outputs/hill",
    "random_search": "outputs/random_search",
}


def resolve_default_output_base(algorithm: str) -> str:
    if algorithm not in DEFAULT_OUTPUT_BASES:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return DEFAULT_OUTPUT_BASES[algorithm]


def build_default_config() -> AppConfig:
    return AppConfig(
        common=CommonConfig(),
        genetic=GeneticConfig(),
        cmaes=CMAESConfig(),
        hill=HillConfig(),
        random_search=RandomSearchConfig(),
    )


def apply_overrides(dataclass_instance: Any, overrides: dict[str, Any]) -> None:
    valid_fields = {field.name for field in fields(dataclass_instance)}
    for key, value in overrides.items():
        if key in valid_fields:
            setattr(dataclass_instance, key, value)


def _project_root_from_config_path(config_path: Optional[str]) -> Path:
    if config_path is not None:
        return Path(config_path).resolve().parent
    return Path(__file__).resolve().parents[1]


def _resolve_project_path(project_root: Path, raw_path: Optional[str]) -> Optional[str]:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_config(config_path: Optional[str] = None) -> AppConfig:
    project_root = _project_root_from_config_path(config_path)
    config = build_default_config()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        apply_overrides(config.common, raw.get("common", {}))
        apply_overrides(config.genetic, raw.get("genetic", {}))
        apply_overrides(config.cmaes, raw.get("cmaes", {}))
        apply_overrides(config.hill, raw.get("hill", {}))
        apply_overrides(config.random_search, raw.get("random_search", {}))
    if config.common.output_base is None:
        config.common.output_base = resolve_default_output_base(config.common.algorithm)
    config.common.output_base = _resolve_project_path(project_root, config.common.output_base)
    config.common.input_image_path = _resolve_project_path(project_root, config.common.input_image_path)
    config.common.instances_csv_path = _resolve_project_path(project_root, config.common.instances_csv_path)
    config.validate()
    return config


def serialize_config(config: AppConfig) -> dict[str, Any]:
    return {
        "common": asdict(config.common),
        "genetic": asdict(config.genetic),
        "cmaes": asdict(config.cmaes),
        "hill": asdict(config.hill),
        "random_search": asdict(config.random_search),
    }


def default_config_path(project_root: Path) -> Path:
    return project_root / "config.yaml"
