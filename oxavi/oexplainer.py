from typing import Tuple, Dict
import logging

import gofi
import xavi
import igp2 as ip
from oxavi.ofeatures import OFeatures
from oxavi.odistribution import ODistribution
from oxavi.util import fill_missing_actions, get_occluded_trajectory, \
    OItem, OFollowLaneCL, overwrite_predictions, get_deterministic_trajectories

logger = logging.getLogger(__name__)


class OXAVIAgent(gofi.GOFIAgent, xavi.XAVIAgent):
    """ Generate new rollouts and save results after MCTS in the observation tau time before. """

    def __init__(self,
                 occluded_factors_prior: float = 0.1,
                 cf_n_trajectories: int = 3,
                 cf_n_simulations: int = 15,
                 cf_max_depth: int = 5,
                 tau_limits: Tuple[float, float] = (1., 5.),
                 time_limits: Tuple[float, float] = (5., 5.),
                 alpha: float = 0.1,
                 alpha_occlusion: float = 0.1,
                 allow_hide_occluded: bool = True,
                 **kwargs):
        """
        Initialise a new OXAVIAgent. The arguments to this agent are the same as xavi.XAVIAgent except for the below.

        ArgS:
            alpha_occlusion: the smoothing weight for occluded factors.
            allow_hide_occluded: whether to allow hiding the occluded factor in simulation despite it being present.
        """
        super().__init__(
            occluded_factors_prior=occluded_factors_prior,
            cf_n_trajectories=cf_n_trajectories,
            cf_n_simulations=cf_n_simulations,
            cf_max_depth=cf_max_depth,
            tau_limits=tau_limits,
            time_limits=time_limits,
            alpha=alpha,
            **kwargs)

        self._alpha_occlusion = alpha_occlusion
        self._allow_hide_occluded = allow_hide_occluded

        mcts_params = {"scenario_map": self._scenario_map,
                       "n_simulations": self._cf_n_simulations,
                       "max_depth": self._cf_max_depth,
                       "reward": self.mcts.reward,
                       "store_results": "all",
                       "tree_type": gofi.OTree,
                       "rollout_type": gofi.ORollout,
                       "allow_hide_occluded": allow_hide_occluded,
                       "trajectory_agents": False}
        self._cf_mcts_dict = {
            "tau": gofi.OMCTS(**mcts_params),
            "t_action": gofi.OMCTS(**mcts_params),
        }

        self._features = OFeatures(self._scenario_map)
        self._matching = xavi.ActionMatching(scenario_map=self._scenario_map)
        self._language = xavi.Language()

    def __repr__(self):
        return f"OXAVIAgent(ID={self.agent_id})"

    def __str__(self):
        return repr(self)

    def update_plan(self, observation: ip.Observation):
        super(OXAVIAgent, self).update_plan(observation)

        # Retrieve maneuvers and macro actions for non-ego vehicles
        for rollout in self.mcts.results:
            last_node = rollout.leaf
            if len(last_node.key) == 2:
                continue
            for agent_id, agent in last_node.run_result.agents.items():
                agent.trajectory_cl.calculate_path_and_velocity()

                # Fill in missing macro action and maneuver information
                if isinstance(agent, ip.TrajectoryAgent):
                    if agent_id in self.goal_probabilities:
                        plan = self.goal_probabilities[agent_id].trajectory_to_plan(*rollout.samples[agent_id])
                        fill_missing_actions(agent.trajectory_cl, plan)
                    else:
                        start_frame = {aid: ag.trajectory_cl.states[0] for aid, ag in
                                       last_node.run_result.agents.items()}
                        start_observation = ip.Observation(start_frame, observation.scenario_map)
                        fill_missing_actions(agent.trajectory_cl, None, agent, start_observation)

        current_t = int(self.observations[self.agent_id][0].states[-1].time)
        self._mcts_results_buffer.append((current_t, self.mcts.results))

    def _get_goals_probabilities(self,
                                 observation: ip.Observation,
                                 previous_frame: Dict[int, ip.AgentState]) -> Dict[int, gofi.OGoalsProbabilities]:
        """ Create a new data structure to store goal probability and occluded factor computations.

        Args:
            observation: the observation of the environment for which to generate the data structure
        """
        goals = self.get_goals(observation)
        occluded_factors = self.get_occluded_factors(observation)
        gps = {}
        for aid in previous_frame.keys():
            if aid != self.agent_id:
                gps[aid] = gofi.OGoalsProbabilities(
                    goals, occluded_factors, occluded_factors_priors=self._occluded_factors_prior)
        return gps

    def _generate_rollouts(self,
                           frame: Dict[int, ip.AgentState],
                           observations: xavi.Observations,
                           goal_probabilities: Dict[int, gofi.OGoalsProbabilities],
                           mcts: gofi.OMCTS,
                           time_reference: str):
        """ Runs OMCTS to generate a new sequence of macro actions to execute using previous observations.

        Args:
            frame: Observation of the env tau time steps back.
            observations: Dictionary of observation history.
            goal_probabilities: Dictionary of predictions for each non-ego agent.
        """
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        # Increase number of trajectories to generate
        n_trajectories = self._goal_recognition._n_trajectories
        self._goal_recognition._n_trajectories = self._n_trajectories

        previous_agent_id = None
        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            # Perform belief merging by using previous agent posterior as next agent prior
            if previous_agent_id is not None:
                for factor, pz in goal_probabilities[previous_agent_id].occluded_factors_probabilities.items():
                    goal_probabilities[agent_id].occluded_factors_priors[factor] = pz

            # Generate all possible trajectories for non-egos from tau time steps back
            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=goal_probabilities[agent_id],
                observed_trajectory=observations[agent_id][0],
                agent_id=agent_id,
                frame_ini=observations[agent_id][1],
                frame=frame,
                visible_region=visible_region)

            # For past queries, use existing most recent goal probabilities
            latest_predictions = self.mcts_results_buffer[-1][1].predictions
            if self.query.tense == "past" and agent_id in latest_predictions:
                overwrite_predictions(
                    latest_predictions[agent_id],
                    goal_probabilities[agent_id])

            # Set the probabilities equal for each goal and trajectory
            #  to make sure we can sample all counterfactual scenarios
            goal_probabilities[agent_id].add_smoothing(
                alpha_goal=self._alpha,
                alpha_occlusion=self._alpha_occlusion,
                uniform_goals=False)
            previous_agent_id = agent_id

            logger.info("")
            logger.info("Goals probabilities for agent %s after (possible) overriding and smoothing.", agent_id)
            goal_probabilities[agent_id].log(logger)
            logger.info("")

        # Set merged beliefs to be the same for all agents, i.e., the beliefs of the last agent in the merging order
        pz = goal_probabilities[previous_agent_id].occluded_factors_probabilities
        for agent_id, probabilities in goal_probabilities.items():
            probabilities.set_merged_occluded_factors_probabilities(pz)

        # Reset the number of trajectories for goal generation
        self._goal_recognition._n_trajectories = n_trajectories

        # Run MCTS search for counterfactual simulations while storing run results

        ip.CLManeuverFactory.maneuver_types["follow-lane"] = OFollowLaneCL
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        all_deterministic_trajectories = get_deterministic_trajectories(goal_probabilities)
        distribution = ODistribution(goal_probabilities)

        ip.MacroActionFactory.macro_action_types["Exit"] = xavi.util.Exit_
        xavi.util.Exit_.ALWAYS_STOPS = self._always_check_stop
        for i, deterministic_trajectories in enumerate(all_deterministic_trajectories):
            logger.info(f"Running deterministic simulations {i + 1}/{len(all_deterministic_trajectories)}")

            mcts.search(
                agent_id=self.agent_id,
                goal=self.goal,
                frame=frame,
                meta=agents_metadata,
                predictions=deterministic_trajectories)
            
            # Record rollout data for sampling
            key_trajectories = {}
            for aid, gp in deterministic_trajectories.items():
                key = (gp.goals[0], gp.occluded_factors[0])
                key_trajectories[aid] = (key, gp.all_trajectories[key][0])
            probabilities, data, reward_data = xavi.util.get_visit_probabilities(mcts.results, p_optimal=self._p_optimal)
            distribution.add_distribution(key_trajectories, probabilities, data, reward_data)
        ip.MacroActionFactory.macro_action_types["Exit"] = ip.Exit

        self._cf_sampling_distribution[time_reference] = distribution
    ip.CLManeuverFactory.maneuver_types["follow-lane"] = ip.FollowLaneCL

    def _get_dataset(self,
                     sampling_distribution: ODistribution,
                     observations: xavi.Observations,
                     reference_t: int) \
            -> Dict[int, xavi.Item]:
        """ Return a dataset recording states, boolean feature, and reward taking occlusions into account

         Args:
             mcts_results: OMCTS results class to convert to a dataset.
             goal_probabilities: Predictions for non-ego vehicles with occluded factors.
             observations: The observations that preceded the planning step.
             reference_t: The time of the start of the counterfactual simulation.
         """
        raw_samples = sampling_distribution.sample_dataset(self._cf_n_samples)
        dataset = {}
        for m, (goal_trajectories, trace, last_node, rewards, occluded_factor) in enumerate(raw_samples):
            trajectories = {}
            trajectory_queried_agent = None

            # save trajectories of each agent
            agents = last_node.run_result.agents
            for agent_id, agent in agents.items():
                trajectory = ip.StateTrajectory(self.fps)
                if agent_id in observations:
                    observed_trajectory = observations[agent_id][0]
                else:
                    # For occluded factors, use A* to plan an open-loop
                    # trajectory from the initial state to the beginning of the simulation
                    initial_frame = {aid: ag.trajectory_cl.states[0] for aid, ag in agents.items()}
                    end_state = agent.trajectory_cl.states[1]
                    observed_trajectory, plan = get_occluded_trajectory(
                        agent, ip.Observation(initial_frame, self._scenario_map), end_state)
                    fill_missing_actions(observed_trajectory, plan)
                    observed_trajectory = observed_trajectory.slice(None, -1)
                trajectory.extend(observed_trajectory, reload_path=False)
                sim_trajectory = agent.trajectory_cl.slice(1, None)

                # Retrieve maneuvers and macro actions for non-ego vehicles
                if isinstance(agent, ip.TrajectoryAgent):
                    start_frame = {aid: ag.trajectory_cl.states[1] for aid, ag in agents.items()}
                    start_observation = ip.Observation(start_frame, self._scenario_map)
                    fill_missing_actions(sim_trajectory, None, agent, start_observation)
                elif isinstance(agent, ip.TrafficAgent):
                    plan = sampling_distribution.agent_distributions[agent_id].trajectory_to_plan(*goal_trajectories[agent_id])
                    fill_missing_actions(sim_trajectory, plan)

                if agent_id == self.query.agent_id:
                    trajectory_queried_agent = sim_trajectory

                trajectory.extend(sim_trajectory, reload_path=True)
                trajectories[agent_id] = trajectory

            # Slice the trajectory according to the tense in case of multiple actions in query exist in a trajectory
            sliced_trajectory = self.query.slice_segment_trajectory(
                trajectory_queried_agent, self._current_t, present_ref_t=reference_t)
            query_factual = self.query.factual if not self.query.all_factual and self.query.exclusive else None
            y = self._matching.action_matching(
                self.query.action, sliced_trajectory, query_factual)
            if self.query.negative:
                y = not y


            data_set_m = OItem(trajectories, y, rewards, trace, last_node, occluded_factor)
            dataset[m] = data_set_m

            # data_set_m = OItem(trajectories, y, r, rollout, rollout.occluded_factor)
            # dataset[m] = data_set_m

        logger.debug('Dataset generation done.')
        return dataset
