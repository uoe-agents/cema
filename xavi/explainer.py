import numpy as np
import pandas as pd
import logging

import igp2 as ip
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from xavi.features import Features
from xavi.util import fill_missing_actions, truncate_observations, \
    to_state_trajectory, find_join_index, Observations
from xavi.matching import ActionMatching, ActionGroup, ActionSegment
from xavi.query import Query, QueryType

logger = logging.getLogger(__name__)


@dataclass
class Item:
    """ Class to store a (counter)factual trajectories, the actions of the ego,
    the rewards, and the query observation generated by MCTS. """
    trajectories: Dict[int, ip.StateTrajectory]
    action_present: bool
    reward: Dict[str, float]


class XAVIAgent(ip.MCTSAgent):
    """ Generate new rollouts and save results after MCTS in the observation tau time before. """

    def __init__(self,
                 cf_n_trajectories: int = 3,
                 cf_n_simulations: int = 15,
                 cf_max_depth: int = 5,
                 **kwargs):
        """ Create a new XAVIAgent.

        Args:
            tau: The interval to roll back for counterfactual generation. By default set to FPS.
            cf_n_trajectories: Number of maximum trajectories to generate with A*.
            cf_n_simulations: Number of MCTS simulations to run for counterfactual generation.
            cf_d_max: Maximum MCTS search depth for counterfactual simulations.

        Keyword Args: See arguments of parent-class MCTSAgent.
        """

        super(XAVIAgent, self).__init__(**kwargs)

        self.__n_trajectories = cf_n_trajectories
        self.__scenario_map = kwargs["scenario_map"]

        self.__cf_n_simulations = kwargs.get("cf_n_simulations", cf_n_simulations)
        self.__cf_max_depth = kwargs.get("cf_max_depth", cf_max_depth)
        self.__cf_goal_probabilities_dict = {"tau": None, "t_action": None}
        self.__cf_observations_dict = {"tau": None, "t_action": None}
        self.__cf_dataset_dict = {"tau": None, "t_action": None}
        self.__cf_mcts_dict = {
            "tau": ip.MCTS(scenario_map=self.__scenario_map,
                           n_simulations=self.__cf_n_simulations,
                           max_depth=self.__cf_max_depth,
                           reward=self.mcts.reward,
                           store_results="all"),
            "t_action": ip.MCTS(scenario_map=self.__scenario_map,
                                n_simulations=self.__cf_n_simulations,
                                max_depth=self.__cf_max_depth,
                                reward=self.mcts.reward,
                                store_results="all"),
        }

        self.__features = Features()
        self.__matching = ActionMatching()
        self.__previous_queries = []
        self.__user_query = None
        self.__current_t = None
        self.__observations_segments = None
        self.__total_trajectories = None
        self.__mcts_resampled = None

    def explain_actions(self, user_query: Query) -> Any:  # TODO (high): Replace return value once we know what it is.
        """ Explain the behaviour of the ego considering the last tau time-steps and the future predicted actions.

        Args:
            user_query: The parsed query of the user.
        """
        self.__user_query = user_query
        self.__current_t = self.observations[self.agent_id][0].states[-1].time
        if self.__observations_segments is None or user_query.t_query != self.__current_t:
            self.__observations_segments = {}
            for aid, obs in self.observations.items():
                self.__observations_segments[aid] = self.__matching.action_segmentation(obs[0])
        if self.__total_trajectories is None or user_query.t_query != self.__current_t:
            self.__total_trajectories = self.__get_total_trajectories()

        # Determine timing information of the query.
        self.query.get_tau(self.total_observations)
        logger.info(f"Running explanation for {self.query}.")

        if self.query.type == QueryType.WHAT:
            return self.__explain_what()
        elif self.query.type == QueryType.WHY:
            # Generate new or update existing dataset.
            self.__get_counterfactuals(["tau", "t_action"])

            assert self.__cf_dataset_dict["t_action"] is not None, f"Missing counterfactual dataset."

            # If t_action < t_current_action: Past time; Need tau and t_action
            # If t_action in t_current_action: Current time; Need tau and t_action
            # If t_action > t_current_action: Future time; Only t_current_action

            final_causes = self.__final_causes()
            past_causes, future_causes = self.__efficient_causes()

        elif self.query.type == QueryType.WHAT_IF:
            # TODO: Determine most likely action under counterfactual condition.
            # generate a new dataset, output the most likely action
            mcts_results_label = []
            self.__generate_counterfactuals_from_time("t_action")
            for key, tra in self.cf_datasets["t_action"].items():
                if self.query.negative and not tra.action_present:
                    mcts_results_label.append(self.__mcts_resampled.results.mcts_results[key])
                elif not self.query.negative and tra.action_present:
                    mcts_results_label.append(self.__mcts_resampled.results.mcts_results[key])
            # find the maximum q value and the corresponding action
            q_max = float('-inf')
            tra_optimal = None
            for m, rollout in enumerate(mcts_results_label):
                last_node = rollout.tree[rollout.trace[:-1]]
                if last_node.run_results[-1].q_values.max() > q_max:
                    q_max = last_node.run_results[-1].q_values.max()
                    agent = last_node.run_results[-1].agents[self.agent_id]
                    tra_optimal = agent.trajectory_cl
                    action = last_node.actions_names[-1]
                    reward_counter = last_node.reward_results[action]
            segments = self.__matching.action_segmentation(tra_optimal)
            grouped_segments = ActionGroup.group_by_maneuver(segments)
            maneuver = [seg for seg in grouped_segments if seg.start <= self.query.t_action <= seg.end][0]

            # determine final cause, compare initial optimal maneuver and counter optimal maneuver
            map_predictions = {aid: p.map_prediction() for aid, p in self.cf_goals_probabilities["t_action"].items()}
            optimal_rollouts = self.__mcts_resampled.results.optimal_rollouts
            matching_rollout = None
            for rollout in optimal_rollouts:
                for aid, prediction in map_predictions.items():
                    if rollout.samples[aid] != prediction: break
                else:
                    matching_rollout = rollout
                    break
            last_node = matching_rollout.tree[matching_rollout.trace[:-1]]
            action = last_node.actions_names[-1]
            reward_init = last_node.reward_results[action]

            # TODO: compare reward counter and reward init

            return maneuver

        # TODO: Convert to NL explanations through language templates.

        self.__previous_queries.append(self.__user_query)

    def __get_total_trajectories(self) -> [Observations, List]:
        """ Return the optimal predicted trajectories for all agents. This would be the optimal MCTS plan for
        the ego and the MAP predictions for non-ego agents.

         Returns:
             Optimal predicted trajectories and their initial state as Observations.
             The reward
         """
        # Use observations until current time
        ret = {}
        map_predictions = {aid: p.map_prediction() for aid, p in self.goal_probabilities.items()}

        for agent_id in self.observations:
            trajectory = ip.StateTrajectory(self.fps)
            trajectory.extend(self.observations[agent_id][0], reload_path=False)

            # Find simulated trajectory that matches best with observations and predictions
            if agent_id == self.agent_id:
                optimal_rollouts = self.mcts.results.optimal_rollouts
                matching_rollout = None
                for rollout in optimal_rollouts:
                    for aid, prediction in map_predictions.items():
                        if rollout.samples[aid] != prediction: break
                    else:
                        matching_rollout = rollout
                        break
                last_node = matching_rollout.tree[matching_rollout.trace[:-1]]
                agent = last_node.run_results[-1].agents[agent_id]
                sim_trajectory = agent.trajectory_cl
            else:
                goal, sim_trajectory = map_predictions[agent_id]
                plan = self.goal_probabilities[agent_id].trajectory_to_plan(goal, sim_trajectory)
                sim_trajectory = to_state_trajectory(sim_trajectory, plan, self.fps)

            # Truncate trajectory to time step nearest to the final observation of the agent
            join_index = find_join_index(self.__scenario_map, trajectory, sim_trajectory)
            sim_trajectory = sim_trajectory.slice(int(join_index), None)

            trajectory.extend(sim_trajectory, reload_path=True, reset_times=True)
            ret[agent_id] = (trajectory, sim_trajectory.states[0])
        return ret

    def __explain_what(self) -> ActionGroup:
        """ Generate an explanation to a what query. Involves looking up the trajectory segment at T and
        returning a feature set of it. We assume for the future that non-egos follow their MAP-prediction for
        goal and trajectory.

        Returns:
            An action group of the executed action at the user given time point.
        """
        if self.query.agent_id is None:
            logger.warning(f"No Agent ID given for what-query. Falling back to ego ID.")
            self.query.agent_id = self.agent_id

        trajectory = self.total_observations[self.query.agent_id][0]
        start_t = self.query.t_action
        if start_t >= len(trajectory):
            logger.warning(f"Total trajectory for Agent {self.query.agent_id} is not "
                           f"long enough for query! Falling back to final timestep.")
            start_t = len(trajectory) - 1
        segments = self.__matching.action_segmentation(trajectory)
        grouped_segments = ActionGroup.group_by_maneuver(segments)
        return [seg for seg in grouped_segments if seg.start <= start_t <= seg.end][0]

    def __final_causes(self) -> pd.DataFrame:
        """ Generate final causes for the queried action.

        Returns:
            Dataframe of reward components with the absolute and relative changes for each component.
        """
        query_present = {}
        query_not_present = {}
        for mid, item in self.cf_datasets["t_action"].items():
            # TODO: Filter for trajectories that match the observations between tau and t_action_start.
            if item.action_present:
                query_present[mid] = item
            else:
                query_not_present[mid] = item

        diffs = {}
        for component in self._reward.COMPONENTS:
            factor = self._reward.factors.get(component, 1.0)
            r_qp = [factor * item.reward[component] for item in query_present.values()
                    if item.reward[component] is not None]
            r_qp = np.sum(r_qp) / len(r_qp) if r_qp else 0.0
            r_qnp = [factor * item.reward[component] for item in query_not_present.values()
                     if item.reward[component] is not None]
            r_qnp = np.sum(r_qnp) / len(r_qnp) if r_qnp else 0.0
            diff = r_qp - r_qnp
            rel_diff = diff / np.abs(r_qnp)
            diffs[component] = (diff if not np.isnan(diff) else 0.0,
                                rel_diff if not np.isnan(rel_diff) else 0.0)
        df = pd.DataFrame.from_dict(diffs, orient="index", columns=["absolute", "relative"])
        return df.sort_values(ascending=False, by="absolute", key=abs)

    def __efficient_causes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Generate efficient causes for the queried action.

        Returns:
            Dataframes for past and future causes with causal effect size.
        """
        xs_past, ys_past = [], []
        xs_future, ys_future = [], []
        tau_dataset = self.cf_datasets["tau"]
        t_action_dataset = self.cf_datasets["t_action"]
        for m in range(self.__cf_n_simulations):
            trajectories_past, trajectories_future = {}, {}
            item_past = tau_dataset[m]
            for aid, traj in tau_dataset[m].trajectories.items():
                trajectories_past[aid] = traj.slice(self.query.tau, self.query.t_action)
            item_future = t_action_dataset[m]
            for aid, traj in t_action_dataset[m].trajectories.items():
                trajectories_future[aid] = traj.slice(self.query.t_action, None)
            xs_past.append(self.__features.to_features(self.agent_id, trajectories_past))
            xs_future.append(self.__features.to_features(self.agent_id, trajectories_future))
            ys_past.append(item_past.action_present)
            ys_future.append(item_future.action_present)
        X_past, y_past = self.__features.binarise(xs_past, ys_past)
        X_future, y_future = self.__features.binarise(xs_future, ys_future)
        model_past = LogisticRegression().fit(X_past, y_past)
        model_future = LogisticRegression().fit(X_future, y_future)

        coefs_past = pd.DataFrame(
            np.squeeze(model_past.coef_) * X_past.std(axis=0),
            columns=["Coefficient importance"],
            index=X_past.columns,
        )
        coefs_future = pd.DataFrame(
            np.squeeze(model_future.coef_) * X_future.std(axis=0),
            columns=["Coefficient importance"],
            index=X_future.columns,
        )
        return coefs_past, coefs_future

    # ---------Counterfactual rollout generation---------------

    def __get_counterfactuals(self, times: List[str]):
        """ Get observations from tau time steps before, and call MCTS from that joint state.

        Args:
            times: The time reference points at which timestep to run MCTS from. Either tau or t_action for now.
        """
        logger.info("Generating counterfactual rollouts.")

        for time_reference in times:
            self.__generate_counterfactuals_from_time(time_reference)

    def __generate_counterfactuals_from_time(self, time_reference: str):
        """ Generate a counterfactual dataset from the time reference point.

         Args:
             time_reference: Either tau or t_action.
         """
        t = getattr(self.query, time_reference)
        truncated_observations, previous_frame = truncate_observations(self.observations, t)
        self.__cf_observations_dict[time_reference] = truncated_observations

        logger.debug(f"Generating counterfactuals at {time_reference} ({t})")
        if previous_frame:
            observation = ip.Observation(previous_frame, self.__scenario_map)
            goals = self.get_goals(observation)
            goal_probabilities = {aid: ip.GoalsProbabilities(goals)
                                  for aid in previous_frame.keys() if aid != self.agent_id}
            mcts = self.__cf_mcts_dict[time_reference]

            self.__generate_rollouts(previous_frame,
                                     truncated_observations,
                                     goal_probabilities,
                                     mcts)
            self.__cf_goal_probabilities_dict[time_reference] = goal_probabilities
            self.__cf_dataset_dict[time_reference] = self.__get_dataset(
                mcts.results, goal_probabilities, truncated_observations)
            self.__mcts_resampled = mcts

    def __generate_rollouts(self,
                            frame: Dict[int, ip.AgentState],
                            observations: Observations,
                            goal_probabilities: Dict[int, ip.GoalsProbabilities],
                            mcts: ip.MCTS):
        """ Runs MCTS to generate a new sequence of macro actions to execute using previous observations.

        Args:
            frame: Observation of the env tau time steps back.
            observations: Dictionary of observation history.
            goal_probabilities: Dictionary of predictions for each non-ego agent.
        """
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        # Increase number of trajectories to generate
        n_trajectories = self._goal_recognition._n_trajectories
        self._goal_recognition._n_trajectories = self.__n_trajectories

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            # Generate all possible trajectories for non-egos from tau time steps back
            gps = goal_probabilities[agent_id]
            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=gps,
                observed_trajectory=observations[agent_id][0],
                agent_id=agent_id,
                frame_ini=observations[agent_id][1],
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

        # Reset the number of trajectories for goal generation
        self._goal_recognition._n_trajectories = n_trajectories

        # Run MCTS search for counterfactual simulations while storing run results
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        mcts.search(
            agent_id=self.agent_id,
            goal=self.goal,
            frame=frame,
            meta=agents_metadata,
            predictions=goal_probabilities)

    def __get_dataset(self,
                      mcts_results: ip.AllMCTSResult,
                      goal_probabilities: Dict[int, ip.GoalsProbabilities],
                      observations: Observations) \
            -> Dict[int, Item]:
        """ Return dataset recording states, boolean feature, and reward

         Args:
             mcts_results: MCTS results class to convert to a dataset.
             goal_probabilities: Predictions for non-ego vehicles.
             observations: The observations that preceded the planning step.
         """
        dataset = {}
        import matplotlib.pyplot as plt
        ip.plot_map(self.__scenario_map, markings=True)
        for m, rollout in enumerate(mcts_results):
            trajectories = {}
            r = []
            last_node = rollout.tree[rollout.trace[:-1]]
            trajectory_queried_agent = None

            # save trajectories of each agent
            for agent_id, agent in last_node.run_results[-1].agents.items():
                trajectory = ip.StateTrajectory(self.fps)
                observed_trajectory = observations[agent_id][0]
                trajectory.extend(observed_trajectory, reload_path=False)
                sim_trajectory = agent.trajectory_cl.slice(1, None)

                # Retrieve maneuvers and macro actions for non-ego vehicles
                if agent_id != self.agent_id:
                    plan = goal_probabilities[agent_id].trajectory_to_plan(*rollout.samples[agent_id])
                    fill_missing_actions(sim_trajectory, plan)

                if agent_id == self.query.agent_id:
                    trajectory_queried_agent = sim_trajectory

                trajectory.extend(sim_trajectory, reload_path=True)
                trajectories[agent_id] = trajectory

                if agent_id == 1:
                    plt.plot(*list(zip(*trajectory.path)))

            # save reward for each component
            for last_action, reward_value, in last_node.reward_results.items():
                if last_action == rollout.trace[-1]:
                    r = reward_value[-1].reward_components

            # Determine outcome
            y = self.__matching.action_matching(self.query.action, trajectory_queried_agent)

            data_set_m = Item(trajectories, y, r)
            dataset[m] = data_set_m

        plt.show()
        logger.debug('Dataset generation done.')
        return dataset

    # -------------Field access properties-------------------

    @property
    def cf_datasets(self) -> Dict[str, Optional[Dict[int, Item]]]:
        """ The most recently generated set of counterfactuals rolled back to tau. """
        return self.__cf_dataset_dict

    @property
    def cf_goals_probabilities(self) -> Dict[str, Optional[Dict[int, ip.GoalsProbabilities]]]:
        """ The goal and trajectory probabilities inferred from tau time steps ago. """
        return self.__cf_goal_probabilities_dict

    @property
    def cf_n_simulations(self) -> int:
        """ The number of rollouts to perform in counterfactual MCTS. """
        return self.__cf_n_simulations

    @property
    def total_observations(self) -> Observations:
        """ Returns the factual observations extended with the most optimal predicted trajectory for all agents. """
        return self.__total_trajectories

    @property
    def observation_segmentations(self) -> Dict[int, List[ActionSegment]]:
        """ Segmentations of the observed trajectories for each vehicle. """
        return self.__observations_segments

    @property
    def query(self) -> Query:
        """ The most recently asked user query. """
        return self.__user_query
