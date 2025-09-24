import json
import os
import pickle
import argparse
import logging
from util import setup_xavi_logging

import xavi
import oxavi
import gofi

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process scenario parameters.")
    parser.add_argument('--scenario', type=int, default=9, help='Scenario number')
    parser.add_argument('--query_idx', type=int, default=0, help='Query index')
    parser.add_argument('--n', type=int, default=30, help='Number of iterations')
    parser.add_argument('--load_existing', action="store_true",
                        default=False, help='Load existing data')
    parser.add_argument('--allow_hide_occluded', action="store_true",
                        default=False, help='Allow hiding occluded objects')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    scenario = args.scenario
    query_idx = args.query_idx
    n = args.n
    load_existing = args.load_existing
    allow_hide_occluded = args.allow_hide_occluded

    output_path = os.path.join("output", f"scenario_{scenario}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger_path = os.path.join(output_path, "logs")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)
    setup_xavi_logging(log_dir=logger_path, log_name=f"test_s{scenario}")

    logger.info(args)

    scenario_map = gofi.OMap.parse_from_opendrive(f"scenarios/maps/scenario{max(1, scenario)}.xodr")
    config = json.load(open(f"scenarios/configs/scenario{scenario}.json", "r", encoding="utf-8"))
    queries = json.load(open(f"scenarios/queries/query_scenario{scenario}.json", "r", encoding="utf-8"))
    query = xavi.Query(**queries[query_idx])

    oxavi.OFollowLaneCL.IGNORE_VEHICLE_IN_FRONT_CHANCE = config["scenario"].get("ignore_vehicle_in_front_chance", 0.0)

    if not load_existing:
        xavi_agent = pickle.load(open(f"output/scenario_{scenario}/agent_t{query.t_query}_m{query.type}.pkl", "rb"))
        sd_path = os.path.join(output_path, f"sd_t{query.t_query}_m{query.type}.pkl")

        for mcts in xavi_agent.cf_mcts.values():
            mcts.n = n
            mcts._allow_hide_occluded = allow_hide_occluded
        # xavi_agent._cf_n_samples = 100
        xavi_agent._n_trajectories = 1
        if os.path.exists(sd_path):
            xavi_agent._cf_sampling_distribution = pickle.load(open(sd_path, "rb"))
        causes = xavi_agent.explain_actions(query)
        if not os.path.exists(sd_path):
            pickle.dump(xavi_agent.cf_sampling_distributions, open(sd_path, "wb"))

        causes_path = os.path.join(output_path, f"q_t{query.t_query}_m{query.type}.pkl")
        pickle.dump(causes, open(causes_path, "wb"))
    else:
        causes = pickle.load(
            open(f"output/scenario_{scenario}/q_t{query.t_query}_m{query.type}.pkl", "rb"))

    if query.type == xavi.QueryType.WHAT_IF:
        cf_action_group = causes[0]
        logger.info(cf_action_group)
        final_causes = causes[1]
        efficient_causes = causes[2]
    else:
        final_causes = causes[0]
        efficient_causes = causes[1]

    # xavi.plot_explanation(final_causes, efficient_causes[0:2], query, uniform_teleological=False)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path, f"q_n{n}_t{query.t_query}_m{query.type}.png"))
    # plt.show()
