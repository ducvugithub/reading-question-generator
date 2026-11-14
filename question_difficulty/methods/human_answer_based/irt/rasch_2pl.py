"""
2PL item response theory: adds a per-item discrimination parameter `a` on top
of the 1PL/Rasch model (rasch_model.py), P(correct) = sigmoid(a_item *
(theta_person - b_item)). Not used for the sparse human-response data (see
rasch_model.py's docstring -- ~2-3 responses/item there, not enough to
identify discrimination) but the race_pool cascade data is dense: all 11
candidates answer every sampled item, which is enough responses/item to
estimate how much each item's correctness actually depends on candidate
ability (discrimination) rather than everyone guessing it right/wrong the
same way regardless of ability.

Motivation: manual inspection of diff-control-irt-10k found main-idea/
"message of the story" questions labeled EASY even though they read as
harder than paired detail questions labeled HARD -- hypothesis is these get
answered correctly via topic-word overlap (a shortcut even weak models can
exploit), which would show up as LOW discrimination (correctness doesn't
track ability), distinct from genuinely easy items that strong AND weak
models both get right because the item really is unambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class TwoPLFitResult:
    item_difficulty: dict[str, float]     # item_id -> b (higher = harder)
    item_discrimination: dict[str, float] # item_id -> a (higher = better separates ability)
    person_ability: dict[str, float]      # person_id -> theta
    converged: bool
    loss: float


def fit_2pl(triples: list[tuple[str, str, bool]], l2_b: float = 0.1, l2_theta: float = 0.1,
            a_prior_mean: float = 1.0, l2_a: float = 0.5,
            maxiter: int = 3000, maxfun: int = 100000) -> TwoPLFitResult:
    """Fits a 2PL model to (person_id, item_id, is_correct) triples.

    a is parameterized as exp(log_a) and initialized near a_prior_mean so it
    stays positive (negative discrimination -- weaker candidates do BETTER --
    is not a sensible solution space here) and the L2 penalty on log_a keeps
    it from blowing up on items with near-perfect or near-zero accuracy
    across all 11 candidates (which would otherwise drive a -> infinity).
    """
    person_ids = sorted(set(t[0] for t in triples))
    item_ids = sorted(set(t[1] for t in triples))
    person_idx = {p: i for i, p in enumerate(person_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}
    n_persons, n_items = len(person_ids), len(item_ids)

    obs_person = np.array([person_idx[t[0]] for t in triples])
    obs_item = np.array([item_idx[t[1]] for t in triples])
    obs_correct = np.array([1.0 if t[2] else 0.0 for t in triples])

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
        log_p_correct = -np.logaddexp(0, -z)
        log_p_wrong = -np.logaddexp(0, z)
        ll = np.sum(obs_correct * log_p_correct + (1 - obs_correct) * log_p_wrong)

        prior = -l2_theta * np.sum(theta ** 2) - l2_b * np.sum(b ** 2) \
            - l2_a * np.sum((log_a - np.log(a_prior_mean)) ** 2)
        nll = -(ll + prior)

        p = 1.0 / (1.0 + np.exp(-z))
        d_z = obs_correct - p  # d(ll)/d(z)

        grad_theta = np.zeros(n_persons)
        grad_b = np.zeros(n_items)
        grad_log_a = np.zeros(n_items)

        np.add.at(grad_theta, obs_person, d_z * a_obs)
        np.add.at(grad_b, obs_item, -d_z * a_obs)
        # d(z)/d(log_a) = a * (theta - b) = z (since z = a*(theta-b))
        np.add.at(grad_log_a, obs_item, d_z * z)

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

    return TwoPLFitResult(
        item_difficulty={item_ids[i]: float(b[i]) for i in range(n_items)},
        item_discrimination={item_ids[i]: float(a[i]) for i in range(n_items)},
        person_ability={person_ids[i]: float(theta[i]) for i in range(n_persons)},
        converged=result.success,
        loss=float(result.fun),
    )
