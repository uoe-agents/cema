""" Utilties to load configurations, saves, and generate random frames. """
import os
import sys
import json
import logging
import pickle
from typing import Dict, Any, List, Tuple, Union

from shapely import Polygon
import matplotlib.pyplot as plt
import numpy as np

import igp2 as ip
import gofi
from cema import xavi, oxavi


logger = logging.getLogger(__name__)


def load_config(config_path: str = None, scenario: int = None) -> Dict[str, Any]:
    """ Load the scenario configuration from a file given a scenario ID or file path.

    Args:
        config_path: The path to the configuration file.
        scenario: The scenario ID.
    """
    if config_path is not None:
        path = config_path
    elif scenario is not None:
        path = os.path.join("scenarios", "configs", f"scenario{scenario}.json")
    else:
        raise ValueError("No scenario was specified!")
    return json.load(open(path, "r", encoding="utf-8"))


def load_scenario(scenario_id: int,
                  query_idx: int) -> Tuple[Union[xavi.XAVIAgent, oxavi.OXAVIAgent], xavi.Query]:
    """ Load the scenario and query for the given scenario ID and query index."""
    scenario_path = os.path.join("output", f"scenario_{scenario_id}")
    query_path = os.path.join("scenarios", "queries", f"query_scenario{scenario_id}.json")
    queries = json.load(open(query_path, encoding="utf-8"))
    query = xavi.Query(**queries[query_idx])
    agent_path = os.path.join(scenario_path, f"agent_t{query.t_query}_m{query.type}.pkl")
    sys.path.append("./src/cema")  # Add path to CEMA to fix unpickling errors
    xavi_agent = pickle.load(open(agent_path, "rb"))
    sd_path = os.path.join(scenario_path, f"sd_t{query.t_query}_m{query.type}.pkl")
    if os.path.exists(sd_path):
        xavi_agent._cf_sampling_distribution = pickle.load(open(sd_path, "rb"))
    return xavi_agent, query


def parse_query(query_path: str = None, scenario: int = None) -> List[xavi.Query]:
    """ Returns a list of parsed queries.

    Args:
        query_path: The path to the query file.
        scenario: The scenario number.
    """
    if query_path is not None:
        path = query_path
    elif scenario is not None:
        path = os.path.join("scenarios", "queries", f"query_scenario{scenario}.json")
    else:
        raise ValueError("No query was specified!")
    try:
        queries = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(queries, dict):
            queries = queries[f"s{scenario}"]
        return [xavi.Query(**query_dict) for query_dict in queries]
    except FileNotFoundError as e:
        logger.exception(str(e), exc_info=e)
        return []


def generate_random_frame(layout: ip.Map, config) -> Dict[int, ip.AgentState]:
    """ Generate a new frame with randomised spawns and velocities for each vehicle.

    Args:
        layout: The current road layout
        config: Dictionary of properties describing agent spawns.

    Returns:
        A new randomly generated frame
    """
    ret = {}
    for agent in config["agents"]:
        spawn_box = ip.Box(**agent["spawn"]["box"])
        spawn_vel = agent["spawn"]["velocity"]

        poly = Polygon(spawn_box.boundary)
        best_lane = None
        max_overlap = 0.0
        for road in layout.roads.values():
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if not lane.boundary.intersects(poly):
                        continue
                    overlap = lane.boundary.intersection(poly)
                    if overlap.area > max_overlap:
                        best_lane = lane
                        max_overlap = overlap.area

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * np.random.random() + start_d

        spawn_position = best_lane.point_at(position_d)
        spawn_heading = best_lane.get_heading_at(position_d)

        vel = (spawn_vel[1] - spawn_vel[0]) * np.random.random() + spawn_vel[0]
        vel = min(vel, ip.Maneuver.MAX_SPEED)
        spawn_velocity = vel * np.array([np.cos(spawn_heading), np.sin(spawn_heading)])

        agent_metadata = ip.AgentMetadata(**agent["metadata"]) if "metadata" in agent \
            else ip.AgentMetadata(**ip.AgentMetadata.CAR_DEFAULT)

        ret[agent["id"]] = ip.AgentState(
            time=0,
            position=spawn_position,
            velocity=spawn_velocity,
            acceleration=np.array([0.0, 0.0]),
            heading=spawn_heading,
            metadata=agent_metadata)
    return ret


def to_ma_list(ma_confs: List[Dict[str, Any]], agent_id: int,
               start_frame: Dict[int, ip.AgentState], scenario_map: ip.Map) \
        -> List[ip.MacroAction]:
    """ Convert a list of macro action configurations to a list of macro actions.

    Args:
        ma_confs: List of macro action configurations.
        agent_id: The ID of the agent.
        start_frame: The initial frame.
        scenario_map: The map of the scenario.
    """
    mas = []
    for config in ma_confs:
        config["open_loop"] = False
        frame = start_frame if not mas else mas[-1].final_frame
        if "target_sequence" in config:
            config["target_sequence"] = [
                scenario_map.get_lane(rid, lid) for rid, lid in config["target_sequence"]]
        ma = ip.MacroActionFactory.create(
            ip.MacroActionConfig(config), agent_id, frame, scenario_map)
        mas.append(ma)
    return mas


def create_agent(agent_config, scenario_map, frame, fps, carla):
    """ Create an agent based on the configuration. """
    base_agent = {"agent_id": agent_config["id"], "initial_state": frame[agent_config["id"]],
                  "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])), "fps": fps}

    mcts_agent = {"scenario_map": scenario_map,
                  "cost_factors": agent_config.get("cost_factors", None),
                  "view_radius": agent_config.get("view_radius", None),
                  "kinematic": not carla,
                  "velocity_smoother": agent_config.get("velocity_smoother", None),
                  "goal_recognition": agent_config.get("goal_recognition", None),
                  "stop_goals": agent_config.get("stop_goals", False)}

    agent_type = agent_config["type"]

    if agent_type == "OXAVIAgent":
        mcts_agent["occluded_factors_prior"] = agent_config.get("occluded_factors_prior", 0.1)
        agent = oxavi.OXAVIAgent(**base_agent, **mcts_agent,
                                 **agent_config["explainer"], **agent_config["mcts"])
        rolename = "ego"
    elif agent_type == "XAVIAgent":
        agent = xavi.XAVIAgent(**base_agent, **mcts_agent,
                               **agent_config["explainer"], **agent_config["mcts"])
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


def explain(
        queries: List[xavi.Query],
        xavi_agent: xavi.XAVIAgent,
        t: int,
        output_path: str,
        save_causes: bool,
        save_agent: bool
    ) -> None:
    """ Explain the actions of the agent at the given time step.

    Args:
        queries: The list of queries.
        xavi_agent: The XAVI agent.
        t: The time step.
        output_path: The path to save the output to.
        save_causes: Whether to save the generated causes.
        save_agent: Whether to save agent data.
    """
    for query in queries:
        if t > 0 and t == query.t_query:
            if save_agent:
                file_name = f"agent_t{t}_m{query.type}.pkl"
                file_path = os.path.join(output_path, file_name)
                pickle.dump(xavi_agent, open(file_path, "wb"))

            causes = xavi_agent.explain_actions(query)

            if save_causes:
                file_path = os.path.join(output_path, f"q_t{t}_m{query.type}.pkl")
                pickle.dump(causes, open(file_path, "wb"))
                file_path = os.path.join(output_path, f"sd_t{t}_m{query.type}.pkl")
                pickle.dump(xavi_agent.cf_sampling_distributions, open(file_path, "wb"))


def run_simple_simulation(
        simulation: gofi.OSimulation,
        plot: bool,
        sim_only: bool,
        queries: List[xavi.Query],
        config: Dict[str, Any],
        output_path: str,
        save_causes: bool,
        save_agent: bool
    ) -> bool:
    """ Run a simple simulation with the given configuration. """
    for t in range(config["scenario"]["max_steps"]):
        simulation.step()
        if t % 20 == 0 and plot:
            xavi.plot_simulation(simulation, debug=False)
            plt.show()
        if not sim_only:
            explain(queries, simulation.agents[0], t, output_path, save_causes, save_agent)
    return True
