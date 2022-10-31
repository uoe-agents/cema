import numpy as np
import logging

import igp2 as ip
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Features:
    """ Class to store a (counter)factual rollout generated by XAVIAgent. """
    states: Dict[int, ip.StateTrajectory]
    feature_exist: Dict[int, List[bool]]
    reward: Dict[str, float]


class XAVIAgent(ip.MCTSAgent):
    """ Generate new rollouts and save results after MCTS in the observation tau time before. """

    def __init__(self, tau: int = None,
                 cf_n_simulations: int = 15,
                 cf_max_depth: int = 5,
                 **kwargs):
        """ Create a new XAVIAgent.

        Args:
            tau: The interval to roll back for counterfactual generation. By default set to FPS.
            cf_n_simulations: Number of MCTS simulations to run for counterfactual generation.
            cf_d_max: Maximum MCTS search depth for counterfactual simulations.

        Keyword Args: See arguments of parent-class MCTSAgent.
        """

        super(XAVIAgent, self).__init__(**kwargs)

        self.__previous_goal_probabilities = None
        self.__previous_mcts = ip.MCTS(scenario_map=kwargs["scenario_map"],
                                       n_simulations=kwargs.get("cf_n_simulations", cf_n_simulations),
                                       max_depth=kwargs.get("cf_max_depth", cf_max_depth),
                                       store_results="all")

        self.__scenario_map = kwargs["scenario_map"]
        self.__tau = tau if tau is not None else kwargs["fps"]
        assert self.__tau >= 0, f"Tau cannot be negative. "

        self.__previous_observations = {}
        self.__dataset = {}

    def explain_actions(self):
        """ Explain the behaviour of the ego considering the last tau time-steps and the future predicted actions. """
        pass

    def generate_counterfactuals(self):
        """ Get observations from tau time steps before, and call MCTS from that joint state."""
        logger.info("Generating counterfactual rollouts.")
        previous_frame = {}
        for agent_id, observation in self.observations.items():
            frame = observation[1]
            len_states = len(observation[0].states)
            if len_states > self.__tau:
                self.__previous_observations[agent_id] = (observation[0].slice(0, len_states - self.__tau), frame)
                previous_frame[agent_id] = observation[0].states[len_states - self.__tau - 1]

        if previous_frame:
            previous_observation = ip.Observation(previous_frame, self.__scenario_map)
            previous_goals = self.get_goals(previous_observation)
            self.__generate_rollouts(previous_observation, previous_goals)
            self.__dataset = self.get_dataset(previous_observation)

    def __generate_rollouts(self, previous_observation: ip.Observation, previous_goals: List[ip.Goal]):
        """ Runs MCTS to generate a new sequence of macro actions to execute using previous observations.

        Args:
            previous_observation: Observation of the env tau time steps back.
            previous_goals: Possible goals from the state tau time steps back.
        """
        frame = previous_observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        self.__previous_goal_probabilities = {aid: ip.GoalsProbabilities(previous_goals)
                                              for aid in frame.keys() if aid != self.agent_id}
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            # Generate all possible trajectories for non-egos from tau time steps back
            gps = self.__previous_goal_probabilities[agent_id]
            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=gps,
                observed_trajectory=self.__previous_observations[agent_id][0],
                agent_id=agent_id,
                frame_ini=self.__previous_observations[agent_id][1],
                frame=frame,
                visible_region=visible_region)

            # Set the probabilities equal for each goal and trajectory
            #  to make sure we can sample all counterfactual scenarios
            n_reachable = sum(map(lambda x: len(x) > 0, gps.trajectories_probabilities.values()))
            for goal, traj_prob in gps.trajectories_probabilities.items():
                traj_len = len(traj_prob)
                if traj_len > 0:
                    gps.goals_probabilities[goal] = 1 / n_reachable
                    gps.trajectories_probabilities[goal] = [1 / traj_len for _ in range(traj_len)]

        # Run MCTS search for counterfactual simulations while storing run results
        self.__previous_mcts.search(
            agent_id=self.agent_id,
            goal=self.goal,
            frame=frame,
            meta=agents_metadata,
            predictions=self.__previous_goal_probabilities)

    def get_dataset(self, previous_observation: ip.Observation) -> Dict[int, Features]:
        """ Return dataset recording states, boolean feature, and reward """
        dataset = {}
        mcts_results = self.__previous_mcts.results

        for m, rollout in enumerate(mcts_results):
            trajectories = {}
            r = []
            last_node = rollout.tree[rollout.trace[:-1]]

            # save trajectories of each agent
            for agent_id, agent in last_node.run_results[-1].agents.items():
                agent.trajectory_cl.calculate_path_and_velocity()
                trajectories[agent_id] = agent.trajectory_cl

            # save reward for each component
            for last_action, reward_value, in last_node.reward_results.items():
                if last_action == rollout.trace[-1]:
                    r = reward_value[-1].reward_components

            data_set_m = Features(trajectories, self.get_outcome_y(trajectories), r)
            dataset[m] = data_set_m

        logger.info('Counterfactual dataset generation done.')
        return dataset

    @staticmethod
    def get_outcome_y(states: Dict[int, ip.AgentState]) -> Dict[int, List[bool]]:
        """ Return boolean value for each predefined feature """
        """
        Args:
            states: the state of all agents.
        """
        features = {
            'accelerating': False,
            'decelerating': False,
            'maintaining': False,
            'relative slower': False,
            'relative faster': False,
            'same speed': False,
            'ever stop': False,
            'mu': False,
            'omega': False,
            'safety': False
        }
        y = {}
        for agent_id, state in states.items():
            if agent_id == 0:
                acc = np.mean(state.acceleration)
                if acc > 0:
                    features['accelerating'] = True
                elif acc == 0:
                    features['maintaining'] = True
                else:
                    features['decelerating'] = True
            y[agent_id] = list(features.values())
        return y

    @property
    def tau(self) -> int:
        """ Rollback time steps for counterfactual generation. """
        return self.__tau
