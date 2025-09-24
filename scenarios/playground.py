import logging
import sys
import argparse
import tqdm
import pandas as pd
from evaluation import load_scenario
from plotting import plot_sampling_results, plot_distribution_results, plot_explanation

logger = logging.Logger(__name__)



def main(args):
    agent, query = load_scenario(args.sid, args.qid)
    query_str = f"n30_t{query.t_query}_m{query.type}"

    # sampling_size_results = {}
    # for sample_size in (pbar := tqdm.tqdm([100, 250, 500, 750, 1000])):
    #     results = []
    #     logger.info("Generating explanations for sample size %d", sample_size)
    #     agent._cf_n_samples = sample_size
    #     causes = agent.explain_actions(query)
    #     results.append(causes)
    #     sampling_size_results[sample_size] = results
    # plot_sampling_results(sampling_size_results, "output/test", query_str)

    causes = agent.explain_actions(query)
    dataset_backup = agent.cf_datasets.copy()
    agent.cf_datasets["tau"] = {k: v for k, v in dataset_backup["tau"].items() if v.occluded_factor.no_occlusions}
    agent.cf_datasets["t_action"] = {k: v for k, v in dataset_backup["t_action"].items() if v.occluded_factor.no_occlusions}
    causes_no_occlusions = agent.explain_actions(query)
    agent.cf_datasets["tau"] = {k: v for k, v in dataset_backup["tau"].items() if not v.occluded_factor.no_occlusions}
    agent.cf_datasets["t_action"] = {k: v for k, v in dataset_backup["t_action"].items() if not v.occluded_factor.no_occlusions}
    causes_occlusions = agent.explain_actions(query)
    plot_explanation(causes_no_occlusions[0], causes_no_occlusions[1][0:2], "noc_" + query_str, "output/test")
    plot_explanation(causes_occlusions[0], causes_occlusions[1][0:2], "oc_" + query_str, "output/test")

    # teleological_causes = causes[0]
    # occluded_past_present = []
    # for item in agent.cf_datasets["tau"].values():
    #     for reward in item.rewards:
    #         occluded_past_present.append(not item.occluded_factor.no_occlusions)
    # occluded_past_present = pd.Series(occluded_past_present)
    # teleological_past_occluded = teleological_causes[0][1][occluded_past_present]
    # teleological_past_notoccluded = teleological_causes[0][1][~occluded_past_present]
    # mechanistic_causes = causes[1]
    # occluded_past_present = mechanistic_causes[2][0]["2_occluded"].astype(bool)

    # distribution_results = {}
    # agent._cf_n_samples = 100
    # for alpha in (pbar := tqdm.tqdm([0.01, 0.1, 0.5, 1.0])):
    #     results = []
    #     pbar.set_description(f"Generating explanations for distribution smoothing alpha {alpha:.3f}")
    #     agent._alpha = alpha
    #     for i in range(10):
    #         causes = agent.explain_actions(query)
    #         results.append(causes)
    #     distribution_results[alpha] = results
    # plot_distribution_results(distribution_results, "output/test", query_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process scenario parameters.")
    parser.add_argument('--sid', type=int, default=1, help='Scenario ID')
    parser.add_argument('--qid', type=int, default=0, help='Index of query to run')
    arguments = parser.parse_args()

    sys.exit(main(arguments))
