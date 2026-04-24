from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .pipeline import Individual, RunContext, next_individual_id
from .config import AppConfig


@dataclass
class GeneticState:
    population: list[Individual]


class GeneticAlgorithm:
    name = "genetic"
    final_eval_stage = "final_eval"
    evaluate_final_population = True

    def __init__(self, ctx: RunContext, config: AppConfig, z0: torch.Tensor):
        self.ctx = ctx
        self.config = config
        self.z0 = z0

    def total_iterations(self) -> int:
        return self.config.common.num_generations

    def initial_config(self) -> dict[str, float]:
        return {}

    def initialize(self, latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> GeneticState:
        population: list[Individual] = []
        for _ in range(self.config.common.population_size):
            noise = torch.randn_like(self.z0) * self.config.common.sigma_local
            population.append(Individual(individual_id=next_individual_id(id_counter), z=(self.z0 + noise).to(self.ctx.device), created_by="init", parent_ids="", mutation_sigma=float(self.config.common.sigma_local), birth_generation=0))
        return GeneticState(population=population)

    def ask(self, state: GeneticState, generation: int, latent_shape: tuple[int, ...], id_counter: dict[str, int], remaining_budget: Optional[int] = None) -> tuple[list[Individual], dict[str, int]]:
        return state.population, {}

    def get_mutation_sigma(self, generation: int) -> float:
        factor = 0.05 + 0.95 * (1.0 - (generation - 1) / max(self.config.common.num_generations - 1, 1))
        max_std = self.config.common.pixel_perturb_std * factor
        return float(torch.rand(1, device=self.ctx.device).item() * max_std)

    @staticmethod
    def crossover_latent(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(1, device=z1.device)
        return alpha * z1 + (1.0 - alpha) * z2

    @staticmethod
    def tournament_selection_index(pop_size: int, fitness: np.ndarray, k: int = 2) -> int:
        candidates = np.random.choice(pop_size, size=k, replace=False)
        return int(candidates[np.argmax(fitness[candidates])])

    def tell(self, state: GeneticState, generation: int, population: list[Individual], fitness: np.ndarray, ask_info: dict[str, int], latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> dict[str, float]:
        elite_idxs = np.argsort(fitness)[-self.config.genetic.elitism :][::-1]
        new_population: list[Individual] = []
        for elite_idx in elite_idxs:
            parent = population[int(elite_idx)]
            new_population.append(Individual(individual_id=next_individual_id(id_counter), z=parent.z.clone(), created_by="elite", parent_ids=str(parent.individual_id), mutation_sigma=None, birth_generation=generation))
        num_mutation_offspring = 0
        num_crossover_offspring = 0
        mutation_sigmas_this_gen: list[float] = []
        while len(new_population) < self.config.common.population_size:
            p1_idx = self.tournament_selection_index(len(population), fitness)
            p2_idx = self.tournament_selection_index(len(population), fitness)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            if random.random() < self.config.genetic.prob_crossover:
                child_z = self.crossover_latent(p1.z, p2.z)
                created_by = "crossover"
                parent_ids = f"{p1.individual_id}|{p2.individual_id}"
                num_crossover_offspring += 1
            else:
                child_z = p1.z.clone()
                created_by = "clone"
                parent_ids = str(p1.individual_id)
            mutation_sigma: Optional[float] = None
            if random.random() < self.config.genetic.prob_mutation:
                mutation_sigma = self.get_mutation_sigma(generation)
                child_z = child_z + (torch.randn_like(child_z) * mutation_sigma)
                created_by = "crossover+mutation" if created_by == "crossover" else "mutation"
                num_mutation_offspring += 1
                mutation_sigmas_this_gen.append(float(mutation_sigma))
            new_population.append(Individual(individual_id=next_individual_id(id_counter), z=child_z, created_by=created_by, parent_ids=parent_ids, mutation_sigma=mutation_sigma, birth_generation=generation))
        state.population = new_population
        sigma_ref = float(np.mean(mutation_sigmas_this_gen)) if mutation_sigmas_this_gen else 0.0
        return {
            "mutation_sigma_reference": sigma_ref,
            "num_mutation_offspring": int(num_mutation_offspring),
            "num_crossover_offspring": int(num_crossover_offspring),
            "elite_count": int(self.config.genetic.elitism),
        }

    def final_population(self, state: GeneticState) -> list[Individual]:
        return state.population
