from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, List, Optional, Union

import logging
import igp2 as ip
import numpy as np

from .matching import ActionMatching, ActionSegment

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """ Supported query types. """
    WHY = "why"
    WHY_NOT = "whynot"
    WHAT_IF = "whatif"
    WHAT = "what"


@dataclass
class Query:
    """ Dataclass to store parsed query information.

    Args:
        type: The type of the query. Either why, whynot, whatif, or what.
        t_query: The time the user asked the query.
        action: The action the user is interested in.
        agent_id: The specific ID of the agent, the user queried.
        negative: If true then the query will be matched against all trajectories that are NOT equal to action.
        t_action: The start timestep of the action in question.
        tau: The number of timesteps to rollback from the present for counterfactual generation.
        tense: Past, future, or present. Indicates the time of the query.
        factual: The factual action of the agent. Useful for whynot and positive what-if queries.
        exclusive: Whether action and factual are exclusive of one another.
    """

    type: QueryType
    t_query: int = None
    action: str = None
    agent_id: int = None
    negative: bool = None
    t_action: int = None
    tau: int = None
    tense: str = None
    factual: str = None
    exclusive: str = True

    fps: int = 20
    # Minimum and maximum distance of tau in seconds from the current time
    tau_limits = np.array([1, 5])
    t_action_min: int = 1  # Minimum distance in seconds for the action start
    # Maximum lengths of the trajectories, both in past and future, in seconds.
    time_limits = np.array([5, 5])

    def __post_init__(self):
        self.__all_factual = self.type == QueryType.WHY_NOT or \
            self.type == QueryType.WHAT_IF and not self.negative
        self.__matching = ActionMatching()
        self.type = QueryType(self.type)
        if self.negative is None:
            self.negative = self.type == QueryType.WHY_NOT
        if self.action is not None:
            assert all([act in self.__matching.action_library
                        for act in self.__matching.action_library]), \
                f"Unknown action {self.action}."
            if self.factual is not None:
                assert self.factual != self.action, f"Factual {self.factual} cannot " \
                                                    f"be the same as the action {self.action}."
        assert self.tense in ["past", "present", "future", None], f"Unknown tense {self.tense}."

    def get_tau(self,
                current_t: int,
                scenario_map: ip.Map,
                observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]],
                rollouts_buffer: List[Tuple[int, ip.AllMCTSResult]]):
        """ Calculate tau and the start time step of the queried action.
        Storing results in fields tau, and t_action.

        Args:
            current_t: The current timestep of the simulation
            scenario_map: The current road layout
            observations: Trajectories observed (and possibly extended with future predictions) of the environment.
            rollouts_buffer: The actual MCTS rollouts of the agent.
        """
        if self.t_action is not None and self.tau is not None:
            return

        agent_id = self.agent_id
        trajectory = observations[agent_id][0]
        self.__matching.set_scenario_map(scenario_map)
        if self.tense == "future":
            action = self.action if self.factual is None else self.factual
            action_segmentations = self.__get_future_rollouts(action, rollouts_buffer, agent_id, current_t)
        else:
            action_segmentations = [self.slice_segment_trajectory(trajectory, current_t)]

        tau = len(trajectory) - 1
        if self.type in [QueryType.WHAT_IF, QueryType.WHY_NOT, QueryType.WHY]:
            t_actions_taus = []
            action = self.action if self.factual is None else self.factual
            for segmentation in action_segmentations:
                t_actions_taus.append(self.__get_t_tau(action, segmentation, True))
            t_action, tau = min(t_actions_taus, key=lambda x: x[0])
        elif self.type == QueryType.WHAT:
            t_action = self.t_action
        else:
            raise ValueError(f"Unknown query type {self.type}.")

        if tau is not None:
            assert tau > 0, f"Tau has to be positive."

        if self.tau is None:
            self.tau = tau  # If user gave fixed tau then we shouldn't override that.
        if self.t_action is None:
            self.t_action = t_action

        if self.factual is not None:
            self.__get_all_factual(rollouts_buffer, self.agent_id, current_t)


    def __get_t_tau(self,
                    action: Union[str, List[str]],
                    action_segmentations: List[ActionSegment],
                    rollback: bool) -> Tuple[int, int]:
        """ determine t_action for final causes, tau for efficient cause.
        Args:
            action_segmentations: the segmented action of the observed trajectory.
            rollback: if rollback is needed

        Returns:
            t_action: the timestep when the factual action starts
            tau: the timesteps to rollback
        """
        tau = None
        action_matched = False
        n_segments = len(action_segmentations)

        if self.tense == "future":
            for i, act in enumerate(action_segmentations):
                if action in act.actions or action == act.actions:
                    t_action = act.times[0]
                    segment_inx = i
                    break
            else:
                raise ValueError(f"Could not match action {action} to trajectory.")
        else:
            for i, act in enumerate(reversed(action_segmentations)):
                if action in act.actions or action == act.actions:
                    action_matched = True
                elif action_matched:
                    t_action = act.times[-1] + 1
                    segment_inx = i
                    break
            else:
                if action_matched:
                    t_action = action_segmentations[0].times[0]  # t at the beginning
                    segment_inx = n_segments - 1
                else:
                    raise ValueError(f"Could not match action {action} to trajectory.")

        if rollback and segment_inx >= 0:
            # In case one extra segment is too short, then lower limit is used.
            lower_limit, upper_limit = self.tau_limits * self.fps
            if self.tense == "future":
                previous_inx = max(0, segment_inx - 1)
            else:
                previous_inx = max(0, n_segments - segment_inx - 1)
            previous_segment = action_segmentations[previous_inx]
            tau = previous_segment.times[0]

            if t_action - tau < lower_limit:
                tau = int(t_action - lower_limit)
            elif t_action - tau > upper_limit:
                tau = int(t_action - upper_limit)

            tau = max(1, tau)

        t_action = max(1, t_action)
        return t_action, tau

    def __get_factual_exists(self, rollouts: ip.AllMCTSResult, agent_id: int, current_t: int) -> List[bool]:
        """ Determine if factual action exist in these rollouts """
        exists = []
        for rollout in rollouts.mcts_results:
            trajectory = rollout.leaf.run_result.agents[agent_id].trajectory_cl
            segmentation = self.slice_segment_trajectory(trajectory, current_t)
            exists.append(ActionMatching.action_exists(segmentation, self.factual, self.tense))
        return exists

    def __get_all_factual(self, rollouts_buffer: List[Tuple[int, ip.AllMCTSResult]], agent_id: int, current_t: int):
        """ Return whether the query factual appears in all rollouts or not. """
        past_limit = self.time_limits[0] * self.fps
        for start_t, rollouts in rollouts_buffer[::-1]:
            if start_t == current_t:
                continue  # Ignore rollout of current time step as no action would have been taken.
            if start_t < current_t - past_limit:
                break  # Limit the amount of time we look back into the past.
            all_factual = all(self.__get_factual_exists(rollouts, agent_id, current_t))
            self.__all_factual = self.__all_factual or all_factual


    def __get_future_rollouts(self,
                              action: List[str],
                              rollouts_buffer: List[Tuple[int, ip.AllMCTSResult]],
                              agent_id: int,
                              current_t: int) -> List[List[ActionSegment]]:
        """ determine the action segmentation of the rollout that matches the query for whynot and what-if questions.
        Args:
            rollouts_buffer: all previous rollouts from MCTS.
            agent_id: the agent id in query
            current_t: current time, unit:s

        Returns:
            action_segmentations: the action segmentation of a rollout matched with the query
        """
        segmentations = []
        past_limit = self.time_limits[0] * self.fps
        for start_t, rollouts in rollouts_buffer[::-1]:
            if start_t == current_t:
                continue  # Ignore rollout of current time step as no action would have been taken.
            if start_t < current_t - past_limit:
                break  # Limit the amount of time we look back into the past.

            if self.factual is not None and not any(self.__get_factual_exists(rollouts, agent_id, current_t)):
                continue  # Skip rollouts that do not contain the factual action.

            for rollout in rollouts.mcts_results:
                trajectory = rollout.leaf.run_result.agents[agent_id].trajectory_cl
                segmentation = self.slice_segment_trajectory(trajectory, current_t)
                action_exists = ActionMatching.action_exists(segmentation, action, self.tense)
                if action_exists:
                    segmentations.append(segmentation)

            if segmentations:
                return segmentations
        raise ValueError(f"The queried action {self.action} does not exist in future rollouts!")


    def slice_segment_trajectory(self,
                                 trajectory: ip.StateTrajectory,
                                 current_t: int,
                                 segment: bool = True,
                                 present_ref_t: int = None) -> Union[ip.StateTrajectory, List[ActionSegment]]:
        # Obtain relevant trajectory slice and segment it
        """ Obtain relevant trajectory slice and segment it.
        Args:
            trajectory: the agent trajectory.
            current_t: the current time
            segment: If true, then return a segmentation
            present_ref_t: The reference time for the time queries.

        Returns:
            action_segmentations: the segmented actions
        """
        past_limit, future_limit = self.time_limits * self.fps
        current_inx = int(current_t - trajectory[0].time + 1)
        start_inx = max(0, current_inx - past_limit)
        end_inx = min(len(trajectory), current_inx + future_limit)
        if self.tense == "past":
            trajectory = trajectory.slice(start_inx, current_inx)
        elif self.tense == "present":
            if present_ref_t is not None:
                t_action_inx = int(present_ref_t - trajectory[0].time + 1)
                trajectory = trajectory.slice(t_action_inx, end_inx)
            else:
                trajectory = trajectory.slice(start_inx, current_inx)
        elif self.tense == "future":
            if present_ref_t is not None:
                t_action_inx = int(present_ref_t - trajectory[0].time + 1)
                if t_action_inx >= end_inx: t_action_inx = 0
                trajectory = trajectory.slice(t_action_inx, end_inx)
            else:
                trajectory = trajectory.slice(current_inx, end_inx)
        elif self.tense is None:
            logger.warning(f"Query time was not given. Falling back to observed trajectory.")
            trajectory = trajectory.slice(start_inx, current_inx)

        if segment:
            return self.__matching.action_segmentation(trajectory)
        return trajectory

    @property
    def all_factual(self) -> bool:
        """ Return whether the query factual appears in all rollouts or not. """
        return self.__all_factual
