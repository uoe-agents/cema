from typing import Dict, Any, Tuple
import random
import igp2 as ip


class Distribution:
    def __init__(self, agent_distributions: Dict[int, ip.GoalsProbabilities]):
        self.agent_distributions = agent_distributions
        self.agent_trajectories = []
        self.plan_distribution = []
        self.plan_data = []
        self.reward_data = []


    def add_distribution(self, 
                         agent_trajectories: Dict[int, ip.VelocityTrajectory], 
                         plan_distribution: Dict[Any, float],
                         plan_data: Dict[Any, ip.Node],
                         reward_data: Dict[Any, ip.Reward]):
        """ Add a new plan distribution to the list of distributions given fixed agent trajectories. """
        self.agent_trajectories.append(agent_trajectories)
        self.plan_distribution.append(plan_distribution)
        self.plan_data.append(plan_data)
        self.reward_data.append(reward_data)

    def sample_agents(self) -> Dict[int, Tuple[ip.Goal, ip.VelocityTrajectory]]:
        """ Sample goals and trajectories for all non-ego agents using the given agent distributions. """
        ret = {}
        for aid, gp in self.agent_distributions.items():
            goal = gp.sample_goals()[0]
            trajectory = gp.sample_trajectories_to_goal(goal)[0][0]
            ret[aid] = (goal, trajectory)
        return ret

    def sample_plan(self, goal_trajectory: Dict[int, Tuple[ip.Goal, ip.VelocityTrajectory]], k: int = 1):
        """ Sample from the plan distribution given a fixed setting of non-ego agent goals and trajectories. """
        distribution_idx = self.agent_trajectories.index(goal_trajectory)
        distribution = self.plan_distribution[distribution_idx]
        traces = list(distribution.keys())
        weights = list(distribution.values())
        traces = random.choices(traces, weights=weights, k=k)
        return traces
    
    def sample_dataset(self, k: int):
        """ Sample k datasets from the distribution. """
        ret = []
        for _ in range(k):
            goal_trajectories = self.sample_agents()
            trace = self.sample_plan(goal_trajectories)[0]
            ix = self.agent_trajectories.index(goal_trajectories)
            data = self.plan_data[ix][trace]
            ret.append((goal_trajectories, trace, data, self.reward_data[ix][trace]))
        return ret
