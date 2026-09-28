"""
1PL (Rasch) item response theory: jointly fits one difficulty parameter
per item and one ability parameter per person from binary (correct/
incorrect) response data, via P(correct) = sigmoid(theta_person - b_item).

Only 1PL is implemented -- 2PL (adds a discrimination parameter per item)
and 3PL (also adds a guessing floor) both need more responses per item to
be reliably identifiable than this dataset has (~2-3 responses/item; see
question_difficulty/notebooks/human_difficulty_onestopqa.ipynb). A
Bayesian variant (e.g. NUTS via PyMC/Stan, giving a full posterior over
difficulty rather than a single point estimate) is a natural next step
and would live alongside this module, not replace it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class RaschFitResult:
    item_difficulty: dict[str, float]   # item_id -> b (higher = harder)
    person_ability: dict[str, float]    # person_id -> theta (higher = more able)
    converged: bool
    loss: float


def fit_rasch(triples: list[tuple[str, str, bool]], l2: float = 0.1,
              maxiter: int = 2000, maxfun: int = 50000) -> RaschFitResult:
    """Fits a 1PL/Rasch model to (person_id, item_id, is_correct) triples.

    l2: prior strength (L2 penalty) on both theta and b. Rasch models have
    a scale/location indeterminacy -- adding a constant to every theta and
    every b leaves P(correct) unchanged -- this prior resolves it and also
    keeps the fit well-posed given how sparse the data is (each person
    typically answers only a handful of items).

    Uses an analytic gradient (not finite-difference): with hundreds of
    person + item parameters, finite-difference gradients need one extra
    function evaluation per parameter per step and can blow through the
    optimizer's evaluation budget before converging.
    """
    person_ids = sorted(set(t[0] for t in triples))
    item_ids = sorted(set(t[1] for t in triples))
    person_idx = {p: i for i, p in enumerate(person_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}
    n_persons, n_items = len(person_ids), len(item_ids)

    obs_person = np.array([person_idx[t[0]] for t in triples])
    obs_item = np.array([item_idx[t[1]] for t in triples])
    obs_correct = np.array([1.0 if t[2] else 0.0 for t in triples])

    def neg_log_likelihood_and_grad(params):
        theta = params[:n_persons]
        b = params[n_persons:]
        z = theta[obs_person] - b[obs_item]
        log_p_correct = -np.logaddexp(0, -z)
        log_p_wrong = -np.logaddexp(0, z)
        ll = np.sum(obs_correct * log_p_correct + (1 - obs_correct) * log_p_wrong)
        reg = l2 * (np.sum(theta**2) + np.sum(b**2))
        nll = -ll + reg

        sig_z = 1.0 / (1.0 + np.exp(-z))
        d_nll_dz = sig_z - obs_correct
        grad_theta = np.zeros(n_persons)
        grad_b = np.zeros(n_items)
        np.add.at(grad_theta, obs_person, d_nll_dz)
        np.add.at(grad_b, obs_item, -d_nll_dz)
        grad_theta += 2 * l2 * theta
        grad_b += 2 * l2 * b
        return nll, np.concatenate([grad_theta, grad_b])

    x0 = np.zeros(n_persons + n_items)
    result = minimize(neg_log_likelihood_and_grad, x0, jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "maxfun": maxfun})

    theta_hat = result.x[:n_persons]
    b_hat = result.x[n_persons:]
    return RaschFitResult(
        item_difficulty={item_ids[i]: b_hat[i] for i in range(n_items)},
        person_ability={person_ids[i]: theta_hat[i] for i in range(n_persons)},
        converged=result.success,
        loss=result.fun,
    )
