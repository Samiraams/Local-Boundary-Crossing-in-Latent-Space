from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .pipeline import Individual, RunContext, next_individual_id
from .config import AppConfig


@dataclass
class HillState:
    incumbent: np.ndarray
    sigma: float
    dim: int
    accepted_moves: int
    attempted_moves: int
    last_population: Optional[list[Individual]] = None


class HillClimbingAlgorithm:
    name = "hill"
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
        budget = self.config.hill.classifier_eval_budget_in_loop
        return max(1, int(np.ceil(budget / self.config.common.population_size)))

    def initial_config(self) -> dict[str, float]:
        return {"NUM_ITERACOES_HILL": self.total_iterations()}

    def initialize(self, latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> HillState:
        x0 = self.flatten_latent(self.z0).astype(np.float64)
        return HillState(incumbent=x0, sigma=float(self.config.common.sigma_local), dim=int(x0.size), accepted_moves=0, attempted_moves=0)

    def ask(self, state: HillState, generation: int, latent_shape: tuple[int, ...], id_counter: dict[str, int], remaining_budget: Optional[int] = None) -> tuple[list[Individual], dict[str, float]]:
        batch_size = min(self.config.common.population_size, remaining_budget or self.config.common.population_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1 for Hill Climbing.")
        x_samples = np.zeros((batch_size, state.dim), dtype=np.float64)
        x_samples[0] = state.incumbent.copy()
        if batch_size > 1:
            noise = np.random.randn(batch_size - 1, state.dim)
            x_samples[1:] = state.incumbent[None, :] + state.sigma * noise
        population: list[Individual] = []
        for i in range(batch_size):
            population.append(Individual(individual_id=next_individual_id(id_counter), z=self.vector_to_latent(x_samples[i].astype(np.float32), latent_shape), created_by="incumbent" if i == 0 else "neighbor", parent_ids="incumbent", mutation_sigma=float(state.sigma), birth_generation=int(generation)))
        state.last_population = population
        return population, {"x_samples": x_samples, "sigma_before": float(state.sigma), "batch_size": batch_size}

    def tell(self, state: HillState, generation: int, population: list[Individual], fitness: np.ndarray, ask_info: dict[str, float], latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> dict[str, float]:
        x_samples = ask_info["x_samples"]
        incumbent_fitness = float(fitness[0])
        best_idx = int(np.argmax(fitness))
        best_fitness = float(fitness[best_idx])
        accepted = best_fitness > incumbent_fitness and best_idx != 0
        state.attempted_moves += 1
        if accepted:
            state.incumbent = x_samples[best_idx].copy()
            state.accepted_moves += 1
            state.sigma = float(
                min(self.config.common.sigma_max, max(self.config.common.sigma_min, state.sigma * self.config.hill.sigma_up_factor)))
        else:
            state.sigma = float(min(self.config.common.sigma_max, max(self.config.common.sigma_min, state.sigma * self.config.hill.sigma_down_factor)))
        return {
            "mutation_sigma_reference": float(ask_info["sigma_before"]),
            "num_mutation_offspring": int(max(0, ask_info["batch_size"] - 1)),
            "num_crossover_offspring": 0,
            "elite_count": 1 if ask_info["batch_size"] > 0 else 0,
            "accepted_move": int(accepted),
            "best_candidate_idx": int(best_idx),
            "sigma_before": float(ask_info["sigma_before"]),
            "sigma_after": float(state.sigma),
            "batch_size": int(ask_info["batch_size"]),
        }

    def final_population(self, state: HillState) -> list[Individual]:
        if state.last_population is None:
            raise RuntimeError("Hill Climbing has no final population available.")
        return state.last_population
