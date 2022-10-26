import numpy as np

import igp2 as ip
from typing import Dict, List
from dataclasses import dataclass
from igp2.agents.mcts_agent import MCTSAgent


@dataclass
class feature_set:
    states: List[ip.AgentState]
    feature_exist: List[bool]
    reward: float


class XAVIAgent(ip.MCTSAgent):
    """ generate new rollouts and save results after MCTS in the observation tau time before """

    def __init__(self,
                 agent_id: int,
                 initial_state: ip.AgentState,
                 t_update: float,
                 scenario_map: ip.Map,
                 goal: ip.Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 cost_factors: Dict[str, float] = None,
                 reward_factors: Dict[str, float] = None,
                 n_simulations: int = 5,
                 max_depth: int = 3,
                 store_results: str = 'final',
                 kinematic: bool = False):

        super(XAVIAgent, self).__init__(agent_id, initial_state, t_update, scenario_map, goal, view_radius,
                                        fps, cost_factors, reward_factors, n_simulations, max_depth, store_results,
                                        kinematic)

        self.__macro_actions = None
        self._mcts = ip.MCTS(scenario_map, n_simulations=n_simulations, max_depth=max_depth,
                             store_results=store_results)

        self._scenario_map = scenario_map
        self._tau = fps
        self._previous_observations = {}
        self._dataset = {}

    def rollout_generation(self):
        previous_state = {}
        for agent_id, observation in self.observations.items():
            frame = observation[1]
            len_states = len(observation[0].states)
            if len_states > self._tau:
                self._previous_observations[agent_id] = (observation[0].slice(0, len_states - self._tau), frame)
                previous_state[agent_id] = observation[0].states[len_states - self._tau - 1]

        if previous_state:
            previous_observation = ip.Observation(previous_state, self._scenario_map)
            self.get_goals(previous_observation)
            self.update_previous_plan(previous_observation)
            self._dataset = self.get_dataset(previous_observation)

    def update_previous_plan(self, previous_observation: ip.Observation):
        """ Runs MCTS to generate a new sequence of macro actions to execute using previous observations."""
        frame = previous_observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        self._goal_probabilities = {aid: ip.GoalsProbabilities(self._goals)
                                    for aid in frame.keys() if aid != self.agent_id}
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            self._goal_recognition.update_goals_probabilities(self._goal_probabilities[agent_id],
                                                              self._previous_observations[agent_id][0],
                                                              agent_id,
                                                              self._previous_observations[agent_id][1],
                                                              frame,
                                                              visible_region=visible_region)
        self.__macro_actions = self._mcts.search(self.agent_id, self.goal, frame,
                                                 agents_metadata, self._goal_probabilities)

    @staticmethod
    def get_outcome_y(states: List[ip.AgentState]) -> List[bool]:
        """ Return boolean value for each predefined feature """
        """
        Args:
            states: the state of all  vehicles.
        """
        # only define the first 7 features, should be added more
        y = {
            'accelerating': False,
            'decelerating': False,
            'maintaining': False,
            'relative slower': False,
            'relative faster': False,
            'same speed': False,
            'ever stop': False
        }
        acc = []
        for state in states:
            acc.append(state.acceleration)
        if np.average(acc) > 0:
            y['accelerating'] = True
        elif np.average(acc) == 0:
            y['maintaining'] = True
        else:
            y['decelerating'] = True

        return list(y.values())

    def get_dataset(self, previous_observation: ip.Observation) -> Dict[int, feature_set]:
        """ Return dataset recording states, boolean feature, and reward """
        dataset = {}
        mcts_results = self._mcts.results
        if isinstance(mcts_results, ip.MCTSResult):
            mcts_results = ip.AllMCTSResult()
            mcts_results.add_data(self._mcts.results)

        # save trajectories of non-ego agents
        for m, rollout in enumerate(mcts_results):
            states = {}
            for aid, pred in rollout.tree.predictions.items():
                states[aid] = {}
                for goal, trajectories in pred.all_trajectories.items():
                    states[aid][goal] = trajectories

            # save trajectories of the ego agent
            states[self.agent_id] = {}
            for macro_action in self.__macro_actions:
                current_macro = self.update_macro_action(macro_action.macro_action_type,
                                                         macro_action.ma_args,
                                                         previous_observation)
                marco = macro_action.__repr__()

                states[self.agent_id][marco] = current_macro.current_maneuver.trajectory


            print('ok')


            data_set_m = feature_set(states, self.get_outcome_y(states), r)
            dataset[m] = data_set_m

        return dataset
