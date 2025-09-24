from typing import Dict, Any, List, Tuple, Optional

import xavi

from oxavi.util import OItem


class OFeatures(xavi.Features):
    def to_features(self,
                    agent_id: int,
                    item: OItem,
                    query: xavi.Query,
                    t_slice: Tuple[Optional[int], Optional[int]],
                    eps: float = 1e-1,
                    exclude_ids: List[int] = None) -> Dict[str, Any]:
        """ Convert a (counter)factual rollout item to binary features. By default, this will exclude the features
        for occluded factors and only indicate the presence of the occluded factors, however, this can be overridden
        by passing an empty list as excluded_ids.

        Args:
            agent_id: ID of agent for which to convert joint trajectories to features.
            item: The data item to convert to features.
            query: The query of the user to use for the conversion.
            t_slice: The time slice to use for trajectories.
            eps: Threshold to check equality to zero
            exclude_ids: Optionally, the IDs of agents to exclude from the features.
        """
        occluded_factor = item.occluded_factor

        # Get standard features from the parent class
        if exclude_ids is None:
            exclude_ids = [elem.agent_id for elem in occluded_factor.present_elements]
        features = super().to_features(
            agent_id, item, query, t_slice, eps=eps, exclude_ids=exclude_ids)

        # Get features for the presence of
        for agent, presence in zip(occluded_factor.elements, occluded_factor.presence):
            features[f"{agent.agent_id}_occluded"] = int(presence)

        return features
