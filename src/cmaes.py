from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .pipeline import Individual, RunContext, next_individual_id
from .config import AppConfig


@dataclass
class CMAESState:
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    B: np.ndarray
    D: np.ndarray
    p_c: np.ndarray
    p_sigma: np.ndarray
    weights: np.ndarray
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c1: float
    c_mu: float
    chi_n: float
    dim: int
    lambda_: int
    mu: int
    last_population: Optional[list[Individual]] = None


class CMAESAlgorithm:
    name = "cmaes"
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
        return self.config.common.num_generations

    def initial_config(self) -> dict[str, float]:
        return {}

    def initialize(
        self,
        latent_shape: tuple[int, ...],
        id_counter: dict[str, int],
    ) -> CMAESState:
        x0 = self.flatten_latent(self.z0)
        dim = int(x0.size)
        lambda_ = self.config.common.population_size
        mu = lambda_ // 2
        if mu < 1:
            raise ValueError("CMA-ES requer population_size >= 2.")
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = float((np.sum(weights) ** 2) / np.sum(weights**2))
        c_sigma = float((mu_eff + 2.0) / (dim + mu_eff + 5.0))
        d_sigma = float(
            1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        )
        c_c = float((4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim))
        c1 = float(2.0 / ((dim + 1.3) ** 2 + mu_eff))
        c_mu = float(
            min(
                1.0 - c1,
                2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff),
            )
        )
        chi_n = float(np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim)))
        return CMAESState(
            mean=x0.astype(np.float64),
            sigma=float(self.config.common.sigma_local),
            C=np.eye(dim, dtype=np.float64),
            B=np.eye(dim, dtype=np.float64),
            D=np.ones(dim, dtype=np.float64),
            p_c=np.zeros(dim, dtype=np.float64),
            p_sigma=np.zeros(dim, dtype=np.float64),
            weights=weights.astype(np.float64),
            mu_eff=mu_eff,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            c_c=c_c,
            c1=c1,
            c_mu=c_mu,
            chi_n=chi_n,
            dim=dim,
            lambda_=lambda_,
            mu=mu,
        )

    def ask(self, state: CMAESState, generation: int, latent_shape: tuple[int, ...], id_counter: dict[str, int], remaining_budget: Optional[int] = None) -> tuple[list[Individual], dict[str, np.ndarray]]:
        z_samples = np.random.randn(state.lambda_, state.dim)
        y_samples = (z_samples * state.D) @ state.B.T
        x_samples = state.mean[None, :] + state.sigma * y_samples
        population: list[Individual] = []
        for i in range(state.lambda_):
            population.append(Individual(individual_id=next_individual_id(id_counter), z=self.vector_to_latent(x_samples[i].astype(np.float32), latent_shape), created_by="cmaes_sample", parent_ids="cma_mean", mutation_sigma=float(state.sigma), birth_generation=int(generation)))
        state.last_population = population
        return population, {"x_samples": x_samples}

    def tell(self, state: CMAESState, generation: int, population: list[Individual], fitness: np.ndarray, ask_info: dict[str, np.ndarray], latent_shape: tuple[int, ...], id_counter: dict[str, int]) -> dict[str, float]:
        x_samples = ask_info["x_samples"]
        order = np.argsort(-fitness.astype(np.float64))
        x_sorted = x_samples[order]
        x_mu = x_sorted[: state.mu]
        mean_old = state.mean.copy()
        mean_new = np.sum(x_mu * state.weights[:, None], axis=0)
        y_w = (mean_new - mean_old) / max(state.sigma, 1e-12)
        inv_sqrt_c_y = state.B @ ((state.B.T @ y_w) / state.D)
        state.p_sigma = (1.0 - state.c_sigma) * state.p_sigma + np.sqrt(
            state.c_sigma * (2.0 - state.c_sigma) * state.mu_eff
        ) * inv_sqrt_c_y
        norm_p_sigma = float(np.linalg.norm(state.p_sigma))
        denom = np.sqrt(max(1.0 - (1.0 - state.c_sigma) ** 2, 1e-12))
        h_sigma_cond = norm_p_sigma / (denom * state.chi_n)
        h_sigma = 1.0 if h_sigma_cond < (1.4 + 2.0 / (state.dim + 1.0)) else 0.0
        state.p_c = (1.0 - state.c_c) * state.p_c + h_sigma * np.sqrt(
            state.c_c * (2.0 - state.c_c) * state.mu_eff
        ) * y_w
        y_k = (x_mu - mean_old[None, :]) / max(state.sigma, 1e-12)
        rank_mu = np.zeros_like(state.C)
        for i in range(state.mu):
            y = y_k[i][:, None]
            rank_mu += state.weights[i] * (y @ y.T)
        delta_h_sigma = (1.0 - h_sigma) * state.c_c * (2.0 - state.c_c)
        state.C = (
            (1.0 - state.c1 - state.c_mu) * state.C
            + state.c1 * ((state.p_c[:, None] @ state.p_c[None, :]) + delta_h_sigma * state.C)
            + state.c_mu * rank_mu
        )
        state.C = 0.5 * (state.C + state.C.T)
        eigvals, eigvecs = np.linalg.eigh(state.C)
        eigvals = np.maximum(eigvals, 1e-20)
        state.D = np.sqrt(eigvals)
        state.B = eigvecs
        state.sigma = float(
            np.clip(
                state.sigma
                * np.exp((state.c_sigma / state.d_sigma) * ((norm_p_sigma / state.chi_n) - 1.0)),
                self.config.common.sigma_min,
                self.config.common.sigma_max,
            )
        )
        state.mean = mean_new
        return {
            "mutation_sigma_reference": float(state.sigma),
            "num_mutation_offspring": int(state.lambda_),
            "num_crossover_offspring": 0,
            "elite_count": 0,
            "sigma_after": float(state.sigma),
        }

    def final_population(self, state: CMAESState) -> list[Individual]:
        if state.last_population is None:
            raise RuntimeError("CMA-ES has no final population available.")
        return state.last_population
