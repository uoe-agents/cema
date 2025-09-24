from dataclasses import dataclass
from typing import List, Tuple, Dict
from itertools import product
import logging

import numpy as np
import igp2 as ip
import xavi
import gofi

logger = logging.getLogger(__name__)


class OFollowLaneCL(ip.FollowLaneCL):
    """ Extend the original FollowLaneCL class to include occluded factors. """
    IGNORE_VEHICLE_IN_FRONT_CHANCE = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_vehicle_in_front = (self.agent_id == 0 and
                                        np.random.random() >= 1 - self.IGNORE_VEHICLE_IN_FRONT_CHANCE)
        if self.ignore_vehicle_in_front:
            logger.debug(f"    Ego ignores vehicle in front.")

    def _get_acceleration(self, target_velocity: float, frame: Dict[int, ip.AgentState]):
        state = frame[self.agent_id]
        acceleration = target_velocity - state.speed
        vehicle_in_front, dist, _ = self.get_vehicle_in_front(self.agent_id, frame, self.lane_sequence)
        if vehicle_in_front is not None:
            if not self.ignore_vehicle_in_front:
                in_front_speed = frame[vehicle_in_front].speed
                gap = dist - state.metadata.length
                acc_acceleration = self._acc.get_acceleration(self.MAX_SPEED, state.speed, in_front_speed, gap)
                acceleration = min(acceleration, acc_acceleration)
        return acceleration


@dataclass
class OItem(xavi.Item):
    """ Extend the original item class to include occluded factors. """
    occluded_factor: gofi.OccludedFactor


def get_occluded_trajectory(
        agent: ip.Agent,
        start_observation: ip.Observation,
        end_state: ip.AgentState) -> Tuple[ip.StateTrajectory, List[ip.MacroAction]]:
    """ Find the trajectory of the occluded agent by using A* to generate an open-loop trajectory.

    Args:
        agent: The occluded agent.
        start_observation: The observation at the start of the occlusion.
        end_state: The state at the end of the occlusion.
    """
    start_state = start_observation.frame[agent.agent_id]

    goal = ip.PointGoal(end_state.position, threshold=0.1)
    if start_state.speed < ip.Stop.STOP_VELOCITY and goal.reached(start_state.position):
        stop_duration = (end_state.time - start_state.time) / agent.fps
        config = ip.MacroActionConfig({'type': 'Stop', "stop_duration": stop_duration})
        ma = ip.MacroActionFactory.create(
            config, agent.agent_id, start_observation.frame, start_observation.scenario_map)
        plan = [ma]
        trajectory = ma.get_trajectory()
    else:
        logging.disable(logging.CRITICAL)
        trajectory, plan = ip.AStar().search(
            agent.agent_id, start_observation.frame, goal, start_observation.scenario_map, debug=False)
        logging.disable(logging.NOTSET)
        trajectory, plan = trajectory[0], plan[0]
    trajectory = ip.StateTrajectory.from_velocity_trajectory(trajectory, fps=agent.fps)
    return trajectory, plan


def fill_missing_actions(
        trajectory: ip.StateTrajectory,
        plan: List[ip.MacroAction] = None,
        agent: ip.Agent = None,
        start_observation: ip.Observation = None
):
    """ Infer maneuver and macro action data for each point in the trajectory and modify trajectory in-place.

    Args:
        trajectory: The StateTrajectory with missing macro and maneuver information.
        plan: The macro action plan that generated the StateTrajectory.
        agent: The agent for which to fill the missing actions.
        start_observation: The observation at the start of the trajectory for trajectory agents
    """
    assert plan is not None or agent is not None, "Either plan or agent must be provided."

    if agent is not None and plan is None:
        _, plan = get_occluded_trajectory(agent, start_observation, agent.trajectory_cl.states[-1])

    ma_man_list = []
    points = None
    for ma in plan:
        for man in ma.maneuvers:
            ma_man_list.extend([(repr(ma), repr(man))] * len(man.trajectory))
            points = man.trajectory.path if points is None else \
                np.append(points, man.trajectory.path, axis=0)
    for state in trajectory:
        nearest_idx = np.argmin(np.linalg.norm(points - state.position, axis=1))
        state.macro_action, state.maneuver = ma_man_list[nearest_idx]


def overwrite_predictions(source: gofi.OGoalsProbabilities, target: gofi.OGoalsProbabilities):
    """ Replace the goals probabilities of the target with the goals probabilities of the source. """
    for factor in target.occluded_factors:
        for goal in target.goals:
            key = (goal, factor)
            for key_ in filter(lambda x: str(x) == str(key), source.goals_and_occluded_factors):
                target.goals_probabilities[key] = source.goals_probabilities[key_]


def get_deterministic_trajectories(goal_probabilities: Dict[int, gofi.OGoalsProbabilities]) -> List[Dict[int, gofi.OGoalsProbabilities]]:
    """ Retrieve all combinations of deterministic trajectories from the goal probabilities. """
    ret = []
    agent_keys = {
        aid: [key for key in product(gp.goals, gp.occluded_factors) if gp.optimum_trajectory[key] is not None] 
        for aid, gp in goal_probabilities.items()
        }
    agent_trajectories = {
        aid: {key: gp.all_trajectories[key] for key in agent_keys[aid]} 
        for aid, gp in goal_probabilities.items()
        }
    
    for combination in xavi.util.product_dict(agent_keys):
        new_gps = {aid: [] for aid in goal_probabilities}
        for aid, key in combination.items():
            for i, trajectory in enumerate(agent_trajectories[aid][key]):
                new_gp = gofi.OGoalsProbabilities([key[0]], [key[1]])
                new_gp.occluded_factors_probabilities[key[1]] = 1.0
                new_gp.merged_occluded_factors_probabilities[key[1]] = 1.0
                new_gp.goals_probabilities[key] = 1.0
                new_gp.all_trajectories[key] = [trajectory]
                new_gp.trajectories_probabilities[key] = [1.0]
                new_gp.all_plans[key] = [goal_probabilities[aid].all_plans[key][i]]
                new_gps[aid].append(new_gp)
        ret.extend(xavi.util.product_dict(new_gps))
    return ret