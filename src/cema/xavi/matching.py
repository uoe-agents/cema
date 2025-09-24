import logging
import numpy as np
import numpy.ma as ma
import igp2 as ip
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

from .util import fix_initial_state

logger = logging.getLogger(__name__)


@dataclass
class ActionSegment:
    times: List[int]
    actions: List[str]


@dataclass
class ActionGroup:
    maneuver: str
    start: int
    end: int
    segments: List[ActionSegment]

    def __repr__(self):
        return f"{self.maneuver}[{self.start}-{self.end}]({len(self.segments)} segments)"

    @staticmethod
    def group_by_maneuver(segmentation: List[ActionSegment]) -> List["ActionGroup"]:
        """ Group consecutive action segmentations by maneuvers. """
        ret, group = [], []
        prev_man = segmentation[0].actions[-1]
        for segment in segmentation:
            man = segment.actions[-1]  # TODO (low): Relies on final action type being the maneuver.
            if prev_man != man:
                ret.append(ActionGroup(prev_man, group[0].times[0], group[-1].times[-1], group))
                group = []
            group.append(segment)
            prev_man = man
        else:
            ret.append(ActionGroup(prev_man, group[0].times[0], group[-1].times[-1], group))
        return ret


class ActionMatching:
    """ Determines if the action asked by a user is present in a trajectory. """
    action_library = ['SlowDown',
                      'Accelerate',
                      'Stop',
                      'ChangeLaneLeft',
                      'ChangeLaneRight',
                      'TurnLeft',
                      'TurnRight',
                      'GoStraightJunction',
                      'GiveWay',
                      'GoStraight']

    def __init__(self, eps: float = 0.1, scenario_map: ip.Map = None):
        self.__trajectory = None
        self.__segmentation = None
        self.__eps = eps
        self.__scenario_map = scenario_map

    def action_segmentation(self, trajectory: ip.StateTrajectory) -> List[ActionSegment]:
        """ Segment the trajectory into different actions and sorted with time.

        Args:
            trajectory: Trajectory to segment.
        """
        if self.__segmentation is not None and self.__trajectory == trajectory:
            return self.__segmentation

        fix_initial_state(trajectory)
        self.__trajectory = trajectory

        action_sequences = []
        for inx in range(len(trajectory.times)):
            action = []
            state = trajectory.states[inx]
            if trajectory.acceleration[inx] < - self.__eps:
                action.append('SlowDown')
            elif trajectory.acceleration[inx] > self.__eps:
                action.append('Accelerate')

            if trajectory.velocity[inx] < trajectory.VELOCITY_STOP:
                action.append('Stop')

            if state.macro_action is not None:
                if 'ChangeLaneLeft' in state.macro_action:
                    action.append('ChangeLaneLeft')
                elif 'ChangeLaneRight' in state.macro_action:
                    action.append('ChangeLaneRight')

            if state.maneuver is not None:
                if 'Turn' in state.maneuver:
                    road_in_roundabout = None
                    if self.__scenario_map is not None:
                        road = self.__scenario_map.best_road_at(state.position, state.heading)
                        road_in_roundabout = self.__scenario_map.road_in_roundabout(road)
                    angular_vel = trajectory.angular_velocity[inx]
                    if road_in_roundabout:
                        action.append('GoStraightJunction')
                    elif angular_vel > self.__eps:
                        action.append('TurnLeft')
                    elif angular_vel < -self.__eps:
                        action.append('TurnRight')
                    else:
                        action.append('GoStraightJunction')
                elif 'GiveWay' in state.maneuver:
                    action.append('GiveWay')
                elif 'FollowLane' in state.maneuver:
                    action.append('GoStraight')

            action_sequences.append(action)

        # Fix other turning actions appearing due to variable angular velocity.
        ActionMatching.__fix_turning_actions(action_sequences, trajectory, self.__eps / 2)

        # aggregate same actions during a period
        action_segmentations = []
        actions, times = [], []
        start_time = int(trajectory.states[0].time)
        previous_actions = action_sequences[0]
        for inx, actions in enumerate(action_sequences, start_time):
            if previous_actions != actions:
                action_segmentations.append(ActionSegment(times, previous_actions))
                times = []
                previous_actions = actions
            times.append(inx)
        else:
            action_segmentations.append(ActionSegment(times, actions))

        self.__segmentation = action_segmentations
        return action_segmentations

    @staticmethod
    def __fix_turning_actions(action_sequence: List[List[str]], trajectory: ip.StateTrajectory, eps: float):
        idx = [a[-1] in ["TurnLeft", "TurnRight", "GoStraightJunction"] for a in action_sequence]
        avels = ma.array(trajectory.angular_velocity, mask=idx)
        for slice in ma.clump_masked(avels):
            mean_avel = trajectory.angular_velocity[slice].mean()
            turn_type = "TurnLeft" if mean_avel > eps else "TurnRight" if mean_avel < -eps else "GoStraightJunction"
            for actions in action_sequence[slice]:
                actions[-1] = turn_type

    @staticmethod
    def action_matching(action: str,
                        action_segmentations: List[ActionSegment],
                        factual: Union[str, Tuple[str, ...]] = None) -> bool:
        """ Match user queried action with trajectories from MCTS.

        Args:
            action: the user queried action.
            action_segmentations: the action segmentation of a trajectory
            factual: the most frequent action(s) for whynot and whatif positive query
        Returns:
            True if action was matched with trajectory
        """
        if isinstance(action, str) and action not in ActionMatching.action_library or \
                isinstance(action, list) and any([act not in ActionMatching.action_library for act in action]):
            raise Exception('User action does not exist in action library.')

        action_found = False
        for action_segmentation in action_segmentations:
            if factual is not None and \
                    (factual == action_segmentation.actions or
                     factual in action_segmentation.actions):
                return False  # For whynot and positive whatif questions, factual actions are not relevant
            if action in action_segmentation.actions or \
                    action == action_segmentation.actions:
                action_found = True
        return action_found

    @staticmethod
    def action_exists(action_segmentations: List[ActionSegment],
                      action: Union[str, Tuple[str, ...]],
                      tense: str = "past") -> bool:
        """ determine if an action exists in the action segmentation """
        if action is None:
            return False

        iterator = action_segmentations if tense == "future" else reversed(action_segmentations)
        for seg in iterator:
            if action in seg.actions or action == seg.actions:
                return True
        return False

    @staticmethod
    def find_counter_actions(action: str) -> List[str]:
        """ find the counter action for whynot or whatif positive question.
        Args:
            action: the user queried action in whynot question.

        Returns:
            possible counter actions
        """
        # define actions and counter actions
        action_set = [['GoStraight', 'ChangeLaneLeft', 'ChangeLaneRight'],
                      ['SlowDown', 'Accelerate'],
                      ['GoStraight', 'Stop'],
                      ['GoStraight', 'TurnLeft', 'TurnRight']]
        counter_actions = []
        for set_ in action_set:
            if action not in set_:
                continue
            else:
                for action_ in set_:
                    if action != action_:
                        counter_actions.append(action_)
        if not counter_actions:
            raise Exception('No counter action is found!')

        return counter_actions

    def set_scenario_map(self, new_map: ip.Map):
        """ Set the current scenario map for the matching"""
        self.__scenario_map = new_map

    @property
    def segmentation(self) -> List[ActionSegment]:
        """ The most recently calculated action segmentation. """
        return self.__segmentation

    @property
    def trajectory(self) -> ip.StateTrajectory:
        """ The most recently passed trajectory for segmentation. """
        return self.__trajectory
