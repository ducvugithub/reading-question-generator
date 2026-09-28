"""
Bayesian 3PL IRT via PyMC/NUTS -- gives a full POSTERIOR (mean, std, and a
credible interval) over each item's difficulty/discrimination and each
person's ability, unlike rasch_model.py's 1PL MLE point estimate.

  P(correct) = c + (1-c) * sigmoid(alpha * (theta - beta))

  theta (ability)        ~ Normal(0, 1)
  alpha (discrimination) ~ LogNormal(0, 0.5)
  beta  (difficulty)     ~ ZeroSumNormal(sigma=1)  -- resolves the theta/beta
      location indeterminacy (adding a constant to every theta and every
      beta leaves P unchanged) without needing rasch_model.py's L2-penalty
      trick.
  c (guessing floor)     a FIXED constant, NOT estimated -- must not be set
      to the empirical correct-rate (self-referential, collapses theta /
      breaks identifiability). Default 0.25, matching this dataset's
      4-option multiple-choice chance floor.

Adapted from a proven pattern in a sibling project
(models/knowledge_structure_models/irt-pipeline/src/irt_estimation.py),
which aggregates multiple attempts per (student, topic) into Binomial
correct/total counts. Our data has exactly one binary response per
(person, item) pair, so this uses a Bernoulli likelihood instead.

Requires a PyMC-compatible environment -- on Apple Silicon, `pip install
pymc` commonly fails building numba/llvmlite from source (needs a specific
matching LLVM toolchain). The irt-pipeline sibling project already has a
working environment at ~/envs/irt_arm_env; point your kernel/interpreter
there rather than re-fighting that build in this repo's plain .venv.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BayesianIRTResult:
    item_difficulty: dict[str, float]                     # posterior mean beta (higher = harder)
    item_difficulty_std: dict[str, float]
    item_difficulty_hdi: dict[str, tuple[float, float]]    # 95% highest-density interval
    item_discrimination: dict[str, float]                  # posterior mean alpha
    item_discrimination_std: dict[str, float]
    person_ability: dict[str, float]                       # posterior mean theta
    person_ability_std: dict[str, float]
    n_divergences: int
    max_rhat: float
    min_ess: float
    trace: object  # arviz.InferenceData -- typed as object so this module imports without arviz


def fit_bayesian_irt(triples: list[tuple[str, str, bool]], guessing: float = 0.25,
                      draws: int = 2000, tune: int = 2000, target_accept: float = 0.95,
                      chains: int = 2, cores: int = 2, random_seed: int = 42) -> BayesianIRTResult:
    """Fits a Bayesian 3PL IRT model to (person_id, item_id, is_correct)
    triples via PyMC/NUTS. Prints convergence diagnostics (divergences,
    R-hat, ESS) before returning -- check these before trusting the
    result, same as the sibling project's IRTEstimator.print_diagnostics."""
    import arviz as az
    import numpy as np
    import pymc as pm

    person_ids = sorted(set(t[0] for t in triples))
    item_ids = sorted(set(t[1] for t in triples))
    person_idx_map = {p: i for i, p in enumerate(person_ids)}
    item_idx_map = {it: i for i, it in enumerate(item_ids)}
    n_persons, n_items = len(person_ids), len(item_ids)

    person_idx = np.array([person_idx_map[t[0]] for t in triples])
    item_idx = np.array([item_idx_map[t[1]] for t in triples])
    correct = np.array([1 if t[2] else 0 for t in triples])

    with pm.Model():
        theta = pm.Normal("theta", mu=0, sigma=1, shape=n_persons)
        alpha = pm.LogNormal("alpha", mu=0, sigma=0.5, shape=n_items)
        beta = pm.ZeroSumNormal("beta", sigma=1, shape=n_items)

        ability = theta[person_idx]
        discrimination = alpha[item_idx]
        difficulty = beta[item_idx]

        logit_p = discrimination * (ability - difficulty)
        p_correct = guessing + (1 - guessing) * pm.math.sigmoid(logit_p)

        pm.Bernoulli("obs", p=p_correct, observed=correct)

        trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept,
                          chains=chains, cores=cores, random_seed=random_seed,
                          return_inferencedata=True)

    theta_post = trace.posterior["theta"]
    alpha_post = trace.posterior["alpha"]
    beta_post = trace.posterior["beta"]

    theta_mean = theta_post.mean(dim=["chain", "draw"]).values
    theta_std = theta_post.std(dim=["chain", "draw"]).values
    alpha_mean = alpha_post.mean(dim=["chain", "draw"]).values
    alpha_std = alpha_post.std(dim=["chain", "draw"]).values
    beta_mean = beta_post.mean(dim=["chain", "draw"]).values
    beta_std = beta_post.std(dim=["chain", "draw"]).values
    beta_hdi = az.hdi(trace, var_names=["beta"], hdi_prob=0.95)["beta"].values

    summ = az.summary(trace, var_names=["alpha", "beta"])
    n_div = int(trace.sample_stats.diverging.sum()) if "diverging" in trace.sample_stats else 0
    max_rhat = float(summ["r_hat"].max())
    min_ess = float(summ["ess_bulk"].min())
    print(f"divergences={n_div}  max_r_hat={max_rhat:.4f} (want <=~1.01)  "
          f"min_ess_bulk={min_ess:.0f} (want >=a few hundred)")

    return BayesianIRTResult(
        item_difficulty={item_ids[i]: float(beta_mean[i]) for i in range(n_items)},
        item_difficulty_std={item_ids[i]: float(beta_std[i]) for i in range(n_items)},
        item_difficulty_hdi={item_ids[i]: (float(beta_hdi[i, 0]), float(beta_hdi[i, 1])) for i in range(n_items)},
        item_discrimination={item_ids[i]: float(alpha_mean[i]) for i in range(n_items)},
        item_discrimination_std={item_ids[i]: float(alpha_std[i]) for i in range(n_items)},
        person_ability={person_ids[i]: float(theta_mean[i]) for i in range(n_persons)},
        person_ability_std={person_ids[i]: float(theta_std[i]) for i in range(n_persons)},
        n_divergences=n_div,
        max_rhat=max_rhat,
        min_ess=min_ess,
        trace=trace,
    )
