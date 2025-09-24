from typing import Dict, Tuple
import gofi
import igp2 as ip

from cema import xavi


class ODistribution(xavi.Distribution):
    """ Distribution of agent goals and trajectories supporting occluded factors."""

    def sample_agents(self) -> Tuple[Dict[int, Tuple[ip.Goal, ip.VelocityTrajectory]], gofi.OccludedFactor]:
        """ Sample goals and trajectories for all non-ego agents using the given agent distributions. """
        ret = {}
        occluded_factor = None
        for aid, gp in self.agent_distributions.items():
            if occluded_factor is None:
                occluded_factor = gp.sample_occluded_factor()[0]
            goal = gp.sample_goals_given_factor(occluded_factor)[0]
            trajectory = gp.sample_trajectory_to_goal_with_factor(goal, occluded_factor)[0][0]
            ret[aid] = ((goal, occluded_factor), trajectory)
        return ret, occluded_factor

    def sample_dataset(self, k):
        """ Sample k datasets from the distribution. """
        ret = []
        for _ in range(k):
            goal_trajectories, occluded_factor = self.sample_agents()
            trace = self.sample_plan(goal_trajectories)[0]
            ix = self.agent_trajectories.index(goal_trajectories)
            data = self.plan_data[ix][trace]
            ret.append((goal_trajectories, trace, data, self.reward_data[ix][trace], occluded_factor))
        return ret