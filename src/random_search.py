from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .pipeline import Individual, RunContext, next_individual_id
from .config import AppConfig


@dataclass
class RandomSearchState:
    center: np.ndarray
    sigma: float
    dim: int
    last_population: Optional[list[Individual]] = None


class RandomSearchAlgorithm:
    name = "random_search"
    final_eval_stage = "final_eval_reuse"
    evaluate_final_population = False

    def __init__(self, ctx: RunContext, config: AppConfig, z0: torch.Tensor):
        self.ctx = ctx
        self.config = config
        self.z0 = z0

    @staticmethod
    def flatten_latent(z: torch.Tensor) -> np.ndarray:
        return z.detach().cpu().view(-1).numpy().astype(np.float32)

    def vector_to_latent(self, v: np.ndarray, latent_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.from_numpy(v.reshape(latent_shape)).to(self.ctx.device).float()

    def total_iterations(self) -> int:
        budget = self.config.random_search.classifier_eval_budget_in_loop
        return max(1, int(np.ceil(budget / self.config.common.population_size)))

    def initial_config(self) -> dict[str, float]:
        return {"NUM_ITERACOES_RANDOM": self.total_iterations()}

    def initialize( self, latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> RandomSearchState:
        x0 = self.flatten_latent(self.z0).astype(np.float64)
        return RandomSearchState(center=x0, sigma=float(self.config.common.sigma_local), dim=int(x0.size))

    def ask(self, state: RandomSearchState, generation: int, latent_shape: tuple[int, ...], id_counter: dict[str, int], remaining_budget: Optional[int] = None) -> tuple[list[Individual], dict[str, float]]:
        batch_size = min(self.config.common.population_size, remaining_budget or self.config.common.population_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1 for Random Search.")
        noise = np.random.randn(batch_size, state.dim)
        x_samples = state.center[None, :] + state.sigma * noise
        population: list[Individual] = []
        for i in range(batch_size):
            population.append(Individual(individual_id=next_individual_id(id_counter), z=self.vector_to_latent(x_samples[i].astype(np.float32), latent_shape), created_by="random_sample", parent_ids="z0_center", mutation_sigma=float(state.sigma), birth_generation=int(generation)))
        state.last_population = population
        return population, {"sigma_before": float(state.sigma), "batch_size": batch_size}

    def tell(self, state: RandomSearchState, generation: int, population: list[Individual], fitness: np.ndarray, ask_info: dict[str, float], latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> dict[str, float]:
        best_idx = int(np.argmax(fitness))
        return {
            "mutation_sigma_reference": float(ask_info["sigma_before"]),
            "num_mutation_offspring": int(ask_info["batch_size"]),
            "num_crossover_offspring": 0,
            "elite_count": 0,
            "best_candidate_idx": int(best_idx),
            "sigma_before": float(ask_info["sigma_before"]),
            "sigma_after": float(ask_info["sigma_before"]),
            "batch_size": int(ask_info["batch_size"]),
        }

    def final_population(self, state: RandomSearchState) -> list[Individual]:
        if state.last_population is None:
            raise RuntimeError("Random Search has no final population available.")
        return state.last_population
