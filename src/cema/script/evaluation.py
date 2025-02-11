""" Evaluate the robustness of the explanation generation with increasing sample sizes
and distribution smoothing. Also plot explanation reults."""
import logging
from typing import Union, Dict, Any

import numpy as np
from tqdm import trange, tqdm
from cema import xavi, oxavi


logger = logging.getLogger(__name__)
SAMPLE_LIMITS = (5, 101)
DISTRIBUTION_ALPHAS = np.arange(0.0, 5.1, 0.1)


def sampling_robustness(
        n_resamples: int,
        agent: Union[xavi.XAVIAgent, oxavi.OXAVIAgent],
        query: xavi.Query) -> Dict[int, Any]:
    """ Evaluate the robustness of the explanation
    generation with increasing sample sizes."""
    sampling_size_results = {}
    for sample_size in (pbar := trange(*SAMPLE_LIMITS)):
        results = []
        logger.info("Generating explanations for sample size %d", sample_size)
        for i in range(n_resamples):
            pbar.set_description_str(f"Sample size {sample_size} iteration {i}")
            try:
                agent._cf_n_samples = sample_size
                causes = agent.explain_actions(query)
                results.append(causes)
            except ValueError:
                logger.warning("Failed iteration %d to generate "
                               "explanation for sample size %d", i, sample_size)
                results.append(None)
        sampling_size_results[sample_size] = results
    return sampling_size_results


def distribution_robustness(
        n_resamples: int,
        agent: Union[xavi.XAVIAgent, oxavi.OXAVIAgent],
        query: xavi.Query) -> Dict[int, Any]:
    """ Evaluate the robustness of the explanation generation
    with increasing distribution smoothing."""
    distribution_results = {}
    agent._cf_n_samples = 50
    for alpha in (pbar := tqdm(DISTRIBUTION_ALPHAS)):
        results = []
        logger.info("Generating explanations for distribution smoothing alpha %d", alpha)
        for i in range(n_resamples):
            pbar.set_description_str(f"Distribution alpha {alpha} iteration {i}")
            try:
                agent._alpha = alpha
                if isinstance(agent, oxavi.OXAVIAgent):
                    agent._alpha_occlusion = alpha
                causes = agent.explain_actions(query)
                results.append(causes)
            except ValueError:
                logger.warning("Failed iteration %d to generate "
                               "explanation for alpha %.1f", i, alpha)
                results.append(None)
        distribution_results[alpha] = results
    return distribution_results
