"""
3PL item response theory via MLE (not Bayesian/NUTS): adds a FIXED guessing
floor c on top of the 2PL model (rasch_2pl.py),
P(correct) = c + (1-c) * sigmoid(a_item * (theta_person - b_item)).

All items here are 4-option multiple choice, so a candidate can get a hard
item right purely by chance (~25%) -- without a guessing floor, 1PL/2PL
underestimate the difficulty of genuinely hard items a candidate got lucky
on (looks like partial knowledge rather than a coin flip).

c is FIXED (not a free parameter per item) -- the existing Bayesian 3PL
(fit_bayesian_irt(), PyMC/NUTS) also fixes c=0.25 for the same reason: with
only ~2-11 responses/item, a free per-item c isn't identifiable. Fixing c
means this is still a plain MLE problem (unlike the Bayesian variant), so it
stays fast even at 10K items -- no NUTS needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ThreePLFitResult:
    item_difficulty: dict[str, float]
    item_discrimination: dict[str, float]
    person_ability: dict[str, float]
    converged: bool
    loss: float
    c: float


def fit_3pl(triples: list[tuple[str, str, bool]], c: float = 0.25,
            l2_b: float = 0.1, l2_theta: float = 0.1,
            a_prior_mean: float = 1.0, l2_a: float = 0.5,
            maxiter: int = 3000, maxfun: int = 100000) -> ThreePLFitResult:
    person_ids = sorted(set(t[0] for t in triples))
    item_ids = sorted(set(t[1] for t in triples))
    person_idx = {p: i for i, p in enumerate(person_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}
    n_persons, n_items = len(person_ids), len(item_ids)

    obs_person = np.array([person_idx[t[0]] for t in triples])
    obs_item = np.array([item_idx[t[1]] for t in triples])
    obs_correct = np.array([1.0 if t[2] else 0.0 for t in triples])

    eps = 1e-9

    def unpack(params):
        theta = params[:n_persons]
        b = params[n_persons:n_persons + n_items]
        log_a = params[n_persons + n_items:]
        return theta, b, log_a

    def neg_log_likelihood_and_grad(params):
        theta, b, log_a = unpack(params)
        a = np.exp(log_a)

        a_obs = a[obs_item]
        z = a_obs * (theta[obs_person] - b[obs_item])
        p = 1.0 / (1.0 + np.exp(-z))
        P = c + (1 - c) * p
        P = np.clip(P, eps, 1 - eps)

        ll = np.sum(obs_correct * np.log(P) + (1 - obs_correct) * np.log(1 - P))
        prior = -l2_theta * np.sum(theta ** 2) - l2_b * np.sum(b ** 2) \
            - l2_a * np.sum((log_a - np.log(a_prior_mean)) ** 2)
        nll = -(ll + prior)

        dP_dz = (1 - c) * p * (1 - p)
        dll_dz = dP_dz * (obs_correct / P - (1 - obs_correct) / (1 - P))

        grad_theta = np.zeros(n_persons)
        grad_b = np.zeros(n_items)
        grad_log_a = np.zeros(n_items)

        np.add.at(grad_theta, obs_person, dll_dz * a_obs)
        np.add.at(grad_b, obs_item, -dll_dz * a_obs)
        np.add.at(grad_log_a, obs_item, dll_dz * z)  # dz/d(log_a) = z

        grad_theta += -2 * l2_theta * theta
        grad_b += -2 * l2_b * b
        grad_log_a += -2 * l2_a * (log_a - np.log(a_prior_mean))

        grad = -np.concatenate([grad_theta, grad_b, grad_log_a])
        return nll, grad

    x0 = np.concatenate([
        np.zeros(n_persons),
        np.zeros(n_items),
        np.full(n_items, np.log(a_prior_mean)),
    ])

    result = minimize(
        neg_log_likelihood_and_grad, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxfun},
    )

    theta, b, log_a = unpack(result.x)
    a = np.exp(log_a)

    return ThreePLFitResult(
        item_difficulty={item_ids[i]: float(b[i]) for i in range(n_items)},
        item_discrimination={item_ids[i]: float(a[i]) for i in range(n_items)},
        person_ability={person_ids[i]: float(theta[i]) for i in range(n_persons)},
        converged=result.success,
        loss=float(result.fun),
        c=c,
    )
