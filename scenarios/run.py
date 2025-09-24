import logging
import os.path
import pickle
import sys
from typing import List
import random

import igp2 as ip
from igp2.core.config import Configuration
import gofi
import numpy as np
import matplotlib.pyplot as plt

from util import generate_random_frame, setup_xavi_logging, parse_args, \
    load_config, parse_query, to_ma_list

import xavi
import oxavi

logger = logging.Logger(__name__)


def main():
    args = parse_args()

    if not os.path.exists("output"):
        os.mkdir("output")
    output_path = os.path.join("output", f"scenario_{args.scenario}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    setup_xavi_logging(log_dir=os.path.join(output_path, "logs"), log_name="run")

    logger.debug(args)

    config = load_config(args)
    queries = parse_query(args)


    logger.info(args)

    scenario_map = gofi.OMap.parse_from_opendrive(config["scenario"]["map_path"])

    # Get run parameters
    seed = args.seed if args.seed else config["scenario"]["seed"] if "seed" in config["scenario"] else 21

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip_config = Configuration()
    ip_config.set_properties(**config["scenario"])
    oxavi.OFollowLaneCL.IGNORE_VEHICLE_IN_FRONT_CHANCE = config["scenario"].get("ignore_vehicle_in_front_chance", 0.0)

    frame = generate_random_frame(scenario_map, config)

    fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20
    try:
        simulation = gofi.OSimulation(scenario_map, fps)

        for agent_config in config["agents"]:
            agent, rolename = create_agent(agent_config, scenario_map, frame, fps, args)
            simulation.add_agent(agent, rolename=rolename)

        if args.plot:
            xavi.plot_simulation(simulation, debug=False)
            plt.show()
        result = run_simple_simulation(simulation, args, queries, config, output_path)
    except Exception as e:
        logger.exception(msg=str(e), exc_info=e)
        result = False
    finally:
        del simulation
    return result


def create_agent(agent_config, scenario_map, frame, fps, args):
    base_agent = {"agent_id": agent_config["id"], "initial_state": frame[agent_config["id"]],
                  "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])), "fps": fps}

    mcts_agent = {"scenario_map": scenario_map,
                  "cost_factors": agent_config.get("cost_factors", None),
                  "view_radius": agent_config.get("view_radius", None),
                  "kinematic": not args.carla,
                  "velocity_smoother": agent_config.get("velocity_smoother", None),
                  "goal_recognition": agent_config.get("goal_recognition", None),
                  "stop_goals": agent_config.get("stop_goals", False)}

    agent_type = agent_config["type"]

    if agent_type == "OXAVIAgent":
        mcts_agent["occluded_factors_prior"] = agent_config.get("occluded_factors_prior", 0.1)
        agent = oxavi.OXAVIAgent(**base_agent, **mcts_agent, **agent_config["explainer"], **agent_config["mcts"])
        rolename = "ego"
    elif agent_type == "XAVIAgent":
        agent = xavi.XAVIAgent(**base_agent, **mcts_agent, **agent_config["explainer"], **agent_config["mcts"])
        rolename = "ego"
    elif agent_type in "TrafficAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        rolename = agent_config.get("rolename", "car")
        agent = ip.TrafficAgent(**base_agent)
    elif agent_type == "OccludedAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        agent = gofi.OccludedAgent(occlusions=agent_config["occlusions"], **base_agent)
        rolename = agent_config.get("rolename", "occluded")
    else:
        raise ValueError(f"Unsupported agent type {agent_config['type']}")
    return agent, rolename


def run_simple_simulation(simulation, args, queries, config, output_path) -> bool:
    for t in range(config["scenario"]["max_steps"]):
        simulation.step()
        if t % 20 == 0 and args.plot:
            xavi.plot_simulation(simulation, debug=False)
            plt.show()
        if not args.sim_only:
            explain(queries, simulation.agents[0], t, output_path, args)
    return True


def explain(queries: List[xavi.Query], xavi_agent: xavi.XAVIAgent, t: int, output_path: str, args):
    for query in queries:
        if t > 0 and t == query.t_query:
            if args.save_agent:
                file_name = f"agent_t{t}_m{query.type}.pkl"
                file_path = os.path.join(output_path, file_name)
                pickle.dump(xavi_agent, open(file_path, "wb"))

            causes = xavi_agent.explain_actions(query)

            if args.save_causes:
                file_path = os.path.join(output_path, f"q_t{t}_m{query.type}.pkl")
                pickle.dump(causes, open(file_path, "wb"))
                file_path = os.path.join(output_path, f"sd_t{t}_m{query.type}.pkl")
                pickle.dump(xavi_agent.cf_sampling_distributions, open(file_path, "wb"))


if __name__ == '__main__':
    sys.exit(main())
