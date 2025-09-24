""" Evaluate the robustness of the explanation generation with increasing sample sizes
and distribution smoothing. Also plot explanation reults."""

import sys
sys.path.append("./src/cema")  # Add path to CEMA to fix unpickling errors
import os
import argparse
import json
import pickle
import logging
from typing import Tuple, Union, Dict, Any

import numpy as np
from tqdm import trange, tqdm
import matplotlib
from tqdm.contrib.logging import logging_redirect_tqdm
from cema import setup_cema_logging
from .plotting import plot_sampling_results, plot_distribution_results, plot_explanation
from cema import xavi
from cema import oxavi

logger = logging.getLogger(__name__)
SAMPLE_LIMITS = (5, 101)
DISTRIBUTION_ALPHAS = np.arange(0.0, 5.1, 0.1)


def load_scenario(scenario_id: int, query_idx: int) -> Tuple[Union[xavi.XAVIAgent, oxavi.OXAVIAgent], xavi.Query]:
    """ Load the scenario and query for the given scenario ID and query index."""
    scenario_path = os.path.join("output", f"scenario_{scenario_id}")
    query_path = os.path.join("scenarios", "queries", f"query_scenario{scenario_id}.json")
    queries = json.load(open(query_path, encoding="utf-8"))
    query = xavi.Query(**queries[query_idx])
    agent_path = os.path.join(scenario_path, f"agent_t{query.t_query}_m{query.type}.pkl")
    xavi_agent = pickle.load(open(agent_path, "rb"))
    sd_path = os.path.join(scenario_path, f"sd_t{query.t_query}_m{query.type}.pkl")
    if os.path.exists(sd_path):
        xavi_agent._cf_sampling_distribution = pickle.load(open(sd_path, "rb"))
    return xavi_agent, query


def sampling_robustness(n_resamples: int, agent: Union[xavi.XAVIAgent, oxavi.OXAVIAgent], query: xavi.Query) -> Dict[int, Any]:
    """ Evaluate the robustness of the explanation generation with increasing sample sizes."""
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
                logger.warning("Failed iteration %d to generate explanation for sample size %d", i, sample_size)
                results.append(None)
        sampling_size_results[sample_size] = results
    return sampling_size_results


def distribution_robustness(n_resamples: int, agent: Union[xavi.XAVIAgent, oxavi.OXAVIAgent], query: xavi.Query) -> Dict[int, Any]:
    """ Evaluate the robustness of the explanation generation with increasing distribution smoothing."""
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
                logger.warning("Failed iteration %d to generate explanation for alpha %.1f", i, alpha)
                results.append(None)
        distribution_results[alpha] = results
    return distribution_results


def main(args) -> int:
    """ Evaluate the robustness of the explanation generation with increasing sample sizes
    and distribution smoothing. Also plot explanation reults."""

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Setup output directories
    output_path = os.path.join("output", f"scenario_{args.sid}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger_path = os.path.join(output_path, "logs")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)
    plot_path = os.path.join(output_path, "plots")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)


    # Setup logging
    setup_cema_logging(log_dir=logger_path, log_name="evaluation")
    logging.getLogger("xavi.explainer").setLevel(logging.WARNING)
    logging.getLogger("oxavi.oexplainer").setLevel(logging.WARNING)
    logger.info(args)


    # Load scenario and query
    logger.info("Loading scenario and query . . .")
    agent, query = load_scenario(args.sid, args.qid)
    query_str = f"t{query.t_query}_m{query.type}"

    plot_path_query = os.path.join(plot_path, query_str)
    if not os.path.exists(plot_path_query):
        os.makedirs(plot_path_query, exist_ok=True)


    # Plot causal attributions
    logger.info("Generating plots of explanation for query . . .")
    causes = pickle.load(open(os.path.join(output_path, f"q_{query_str}.pkl"), "rb"))
    if query.type == xavi.QueryType.WHAT_IF:
        cf_action_group = causes[0]
        logger.info(cf_action_group)
        final_causes = causes[1]
        efficient_causes = causes[2]
    elif query.type == xavi.QueryType.WHAT:
        sys.exit("WHAT queries not supported in evaluation.py")
    else:
        final_causes = causes[0]
        efficient_causes = causes[1]
    plot_explanation(final_causes, efficient_causes[0:2], query_str, plot_path_query, scenario_id=args.sid)


    if args.norobust:
        return 0


    # Run explanation generation with increasing uniformity
    logger.info("Running alpha smoothing robustness evaluation . . .")
    distribution_path = os.path.join(output_path, f"distribution_{query_str}.pkl")
    if not os.path.exists(distribution_path):
        with logging_redirect_tqdm():
            distribution_results = distribution_robustness(10, agent, query)
        pickle.dump(distribution_results, open(distribution_path, "wb"))
    else:
        distribution_results = pickle.load(open(distribution_path, "rb"))
    plot_distribution_results(distribution_results, plot_path_query, query_str)


    # Run explanation generation with increasing sample sizes
    logger.info("Running sample size robustness evaluation . . .")
    sampling_path = os.path.join(output_path, f"sampling_{query_str}.pkl")
    if not os.path.exists(sampling_path):
        with logging_redirect_tqdm():
            sampling_results = sampling_robustness(10, agent, query)
        pickle.dump(sampling_results, open(sampling_path, "wb"))
    else:
        sampling_results = pickle.load(open(sampling_path, "rb"))
    plot_sampling_results(sampling_results, plot_path_query, query_str)


    return 0


def cli():
    """ Command line interface for evaluation.py """
    parser = argparse.ArgumentParser(description="Process scenario parameters.")
    parser.add_argument('--sid', type=int, default=1, help='Scenario ID')
    parser.add_argument('--qid', type=int, default=0, help='Index of query to run')
    parser.add_argument('--norobust', action='store_true', default=False, help='Do not run robustness evaluation.')
    arguments = parser.parse_args()

    sys.exit(main(arguments))


if __name__ == "__main__":
    cli()
