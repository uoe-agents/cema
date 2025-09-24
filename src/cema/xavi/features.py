<<<<<<< HEAD:xavi/features.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional

import igp2 as ip
from xavi.util import find_common, Item
from xavi.query import Query

logger = logging.getLogger(__name__)


class Features:
    """ Convert joint trajectories into a set features. """

    def __init__(self, scenario_map: ip.Map):
        self.__features = None
        self.__scenario_map = scenario_map

    def to_features(self,
                    agent_id: int,
                    item: Item,
                    query: Query,
                    t_slice: Tuple[Optional[int], Optional[int]],
                    eps: float = 1e-1,
                    exclude_ids: List[int] = None) -> Dict[str, Any]:
        """ Convert a joint set of trajectories to binary features.

        Args:
            agent_id: ID of agent for which to convert joint trajectories to features.
            item: The data item to convert to features.
            query: The query of the user to use for the conversion.
            t_slice: The time slice to use for trajectories.
            eps: Threshold to check equality to zero
            exclude_ids: Optionally, the IDs of agents to exclude from the features.
        """
        trajectories = {}
        for aid, traj in item.trajectories.items():
            if exclude_ids and aid in exclude_ids:
                continue
            trajectories[aid] = traj.slice(*t_slice)

        features = {}
        agent_trajectory = trajectories[agent_id]

        for aid, trajectory in trajectories.items():
            if aid == agent_id:
                continue

            # Agent acceleration as the weighted sum of acceleration with temporal difference
            if len(trajectory) <= 1:
                acceleration = np.nan
            else:
                acceleration = np.dot(trajectory.timesteps, trajectory.acceleration)
            features.update({f"{aid}_decelerate": int(acceleration < -eps),
                             f"{aid}_maintain": int(-eps <= acceleration <= eps),
                             f"{aid}_accelerate": int(eps < acceleration)})

            # Get average relative velocity as compared to each agent
            if len(trajectory) == 0:
                rel_velocity = np.nan
            else:
                min_len = min(len(trajectory.velocity), len(agent_trajectory.velocity))
                d_velocity = trajectory.velocity[:min_len] - agent_trajectory.velocity[:min_len]
                rel_velocity = d_velocity.mean()
            features.update({f"{aid}_slower": int(rel_velocity < -eps),
                             f"{aid}_same_velocity": int(-eps <= rel_velocity <= eps),
                             f"{aid}_faster": int(eps < rel_velocity)})

            # Get stopping feature for each vehicle.
            if len(trajectory) == 0:
                stopped = False
            else:
                stopped = trajectory.velocity < trajectory.VELOCITY_STOP
            features[f"{aid}_stops"] = int(np.any(stopped))

            # Select the most frequent macro action
            mas, mans = [], []
            if len(trajectory) != 0:
                for state in trajectory:
                    mas.append(state.macro_action)
                    mans.append(state.maneuver)
                inr = self.__scenario_map.in_roundabout(
                    trajectory.path[0], trajectory.heading[0])
                most_common = find_common(mas, False, in_roundabout=inr) or None
                least_common = find_common(mas, True, in_roundabout=inr) or None
                # sequence = "=>".join(unique_sequence(mas))
                # features[f"{aid}_sequence"] = sequence
                features[f"{aid}_xmacro"] = most_common
                if most_common != least_common:
                    features[f"{aid}_nmacro"] = least_common
            # cat_features[f"{aid}_maneuver"] = most_common(mans)

        self.__features = features
        return features

    def binarise(self, data: List[Dict[str, Any]], labels: List[Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """ Binarise a data set of features and labels. """
        data = pd.DataFrame().from_records(data)
        macro_cols = data.filter(regex="macro|sequence|maneuver")
        one_hot = pd.get_dummies(macro_cols)
        new_data = pd.concat([data, one_hot], axis=1)
        new_data = new_data.drop(columns=macro_cols.columns)
        return new_data, np.array(labels, dtype=int)

    @property
    def features(self) -> Dict[str, int]:
        """ The features that were last calculated. """
        return self.__features
=======
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional

import igp2 as ip
from .util import find_common, Item
from .query import Query

logger = logging.getLogger(__name__)


class Features:
    """ Convert joint trajectories into a set features. """

    def __init__(self, scenario_map: ip.Map):
        self.__features = None
        self.__scenario_map = scenario_map

    def to_features(self,
                    agent_id: int,
                    item: Item,
                    query: Query,
                    t_slice: Tuple[Optional[int], Optional[int]],
                    eps: float = 1e-1,
                    exclude_ids: List[int] = None) -> Dict[str, Any]:
        """ Convert a joint set of trajectories to binary features.

        Args:
            agent_id: ID of agent for which to convert joint trajectories to features.
            item: The data item to convert to features.
            query: The query of the user to use for the conversion.
            t_slice: The time slice to use for trajectories.
            eps: Threshold to check equality to zero
            exclude_ids: Optionally, the IDs of agents to exclude from the features.
        """
        trajectories = {}
        for aid, traj in item.trajectories.items():
            if exclude_ids and aid in exclude_ids:
                continue
            trajectories[aid] = traj.slice(*t_slice)

        features = {}
        agent_trajectory = trajectories[agent_id]

        for aid, trajectory in trajectories.items():
            if aid == agent_id:
                continue

            # Agent acceleration as the weighted sum of acceleration with temporal difference
            if len(trajectory) <= 1:
                acceleration = np.nan
            else:
                acceleration = np.dot(trajectory.timesteps, trajectory.acceleration)
            features.update({f"{aid}_decelerate": int(acceleration < -eps),
                             f"{aid}_maintain": int(-eps <= acceleration <= eps),
                             f"{aid}_accelerate": int(eps < acceleration)})

            # Get average relative velocity as compared to each agent
            if len(trajectory) == 0:
                rel_velocity = np.nan
            else:
                min_len = min(len(trajectory.velocity), len(agent_trajectory.velocity))
                d_velocity = trajectory.velocity[:min_len] - agent_trajectory.velocity[:min_len]
                rel_velocity = d_velocity.mean()
            features.update({f"{aid}_slower": int(rel_velocity < -eps),
                             f"{aid}_same_velocity": int(-eps <= rel_velocity <= eps),
                             f"{aid}_faster": int(eps < rel_velocity)})

            # Get stopping feature for each vehicle.
            if len(trajectory) == 0:
                stopped = False
            else:
                stopped = trajectory.velocity < trajectory.VELOCITY_STOP
            features[f"{aid}_stops"] = int(np.any(stopped))

            # Select the most frequent macro action
            mas, mans = [], []
            if len(trajectory) != 0:
                for state in trajectory:
                    mas.append(state.macro_action)
                    mans.append(state.maneuver)
                inr = self.__scenario_map.in_roundabout(
                    trajectory.path[0], trajectory.heading[0])
                most_common = find_common(mas, False, in_roundabout=inr) or None
                least_common = find_common(mas, True, in_roundabout=inr) or None
                # sequence = "=>".join(unique_sequence(mas))
                # features[f"{aid}_sequence"] = sequence
                features[f"{aid}_xmacro"] = most_common
                if most_common != least_common:
                    features[f"{aid}_nmacro"] = least_common
            # cat_features[f"{aid}_maneuver"] = most_common(mans)

        self.__features = features
        return features

    def binarise(self, data: List[Dict[str, Any]], labels: List[Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """ Binarise a data set of features and labels. """
        data = pd.DataFrame().from_records(data)
        macro_cols = data.filter(regex="macro|sequence|maneuver")
        one_hot = pd.get_dummies(macro_cols)
        new_data = pd.concat([data, one_hot], axis=1)
        new_data = new_data.drop(columns=macro_cols.columns)
        return new_data, np.array(labels, dtype=int)

    @property
    def features(self) -> Dict[str, int]:
        """ The features that were last calculated. """
        return self.__features
>>>>>>> 90fede0243d6f17b426f2d2d9ba80de72684e61a:src/cema/xavi/features.py
