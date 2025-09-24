<<<<<<< HEAD:xavi/plotting.py
from typing import Dict, List, Optional, Tuple

import igp2 as ip
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from xavi.query import Query


# -----------Simulation plotting functions---------------------


def plot_simulation(simulation: ip.simplesim.Simulation, axes: plt.Axes = None, debug: bool = False) \
        -> (plt.Figure, plt.Axes):
    """ Plot the current agents and the road layout for visualisation purposes.

    Args:
        simulation: The simulation to plot.
        axes: Axis to draw on
        debug: If True then plot diagnostic information.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig = plt.gcf()
    fig.suptitle(f"T={simulation.t}")

    color_map_ego = plt.cm.get_cmap('Reds')
    color_map_non_ego = plt.cm.get_cmap('Blues')
    color_ego = 'r'
    color_non_ego = 'b'
    color_bar_non_ego = None

    ax = axes[0]
    ip.plot_map(simulation.scenario_map, markings=True, hide_road_bounds_in_junction=True, ax=ax)
    for agent_id, agent in simulation.agents.items():
        if agent is None or not agent.alive:
            continue

        if isinstance(agent, ip.MCTSAgent):
            color = color_ego
            color_map = color_map_ego
        else:
            color = color_non_ego
            color_map = color_map_non_ego

        path = None
        velocity = None
        if isinstance(agent, ip.MacroAgent) and agent.current_macro is not None:
            path = agent.current_macro.current_maneuver.trajectory.path
            velocity = agent.current_macro.current_maneuver.trajectory.velocity
        elif isinstance(agent, ip.TrajectoryAgent) and agent.trajectory is not None:
            path = agent.trajectory.path
            velocity = agent.trajectory.velocity

        agent_plot = None
        if path is not None and velocity is not None:
            agent_plot = ax.scatter(path[:, 0], path[:, 1], c=velocity, cmap=color_map, vmin=-4, vmax=20, s=8)

        vehicle = agent.vehicle
        pol = plt.Polygon(vehicle.boundary, color=color)
        ax.add_patch(pol)
        ax.text(*agent.state.position, agent_id)
        if isinstance(agent, ip.MCTSAgent) and agent_plot is not None:
            plt.colorbar(agent_plot, ax=ax)
            plt.text(0, 0.1, 'Current Velocity: ' + str(agent.state.speed), horizontalalignment='left',
                     verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0, 0.05, 'Current Macro Action: ' + agent.current_macro.__repr__(), horizontalalignment='left',
                     verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0, 0, 'Current Maneuver: ' + agent.current_macro.current_maneuver.__repr__(),
                     horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

            # Plot goals
            for gid, goal in enumerate(agent.possible_goals):
                loc = goal.center
                ax.plot(*loc, "ro")
                ax.plot(*loc, "kx")
                ax.text(*loc, gid)

            # Plot goal probabilities
            plot_predictions(agent, simulation.agents, axes[1], debug)
        elif isinstance(agent, ip.TrajectoryAgent) and color_bar_non_ego is None:
            color_bar_non_ego = plt.colorbar(agent_plot, location="left")
        plt.text(*agent.state.position, agent_id)

    if debug:
        plot_diagnostics(simulation.agents, simulation.actions)

    return fig, axes


def plot_diagnostics(agents: Dict[int, ip.Agent], actions: Dict[int, List[ip.Action]]) -> (plt.Figure, plt.Axes):
    # attributes = ["velocity", "heading", "angular_velocity"]
    attributes = ["velocity", "acceleration", "jerk"]
    n_agents = len(agents)
    n_attributes = len(attributes)
    subplot_w = 5

    # Plot observations
    fig, axes = plt.subplots(n_agents, n_attributes,
                             figsize=(n_attributes * subplot_w, n_agents * subplot_w))
    if n_agents < 2:
        axes = axes[None, :]
    for i, (aid, agent) in enumerate(agents.items()):
        if agent is None:
            continue
        agent.trajectory_cl.calculate_path_and_velocity()
        ts = agent.trajectory_cl.times
        for j, attribute in enumerate(attributes):
            ax = axes[i, j]
            ys = getattr(agent.trajectory_cl, attribute)
            ys = np.round(ys, 4)
            ax.plot(ts, ys, label="Observed")
            ax.scatter(ts, ys, s=5)

            # Plot target velocities
            if attribute == "velocity":
                ys = [action.target_speed for action in actions[aid]]
                ys = [ys[0]] + ys
                ax.plot(ts, ys, c="red", label="Target")
                ax.scatter(ts, ys, s=5, c="red")
            axes[0, j].set_title(attribute)
            plot_maneuvers(agent, ax)
        axes[i, 0].set_ylabel(f"Agent {aid}")
        axes[i, 0].legend()
    fig.tight_layout()
    return fig, axes


def plot_maneuvers(agent: ip.Agent, ax: plt.Axes) -> plt.Axes:
    man_list = np.array([state.maneuver for state in agent.trajectory_cl.states])
    man_list[0] = man_list[1]
    ts = agent.trajectory_cl.times
    colors = ["red", "blue", "green"]
    t_start = 0
    i = 0
    t_max = len(man_list)
    for t_end, (a, b) in enumerate(zip(man_list[:-1], man_list[1:]), 1):
        if a != b:
            ax.axvspan(ts[t_start], ts[t_end], facecolor=colors[i % len(colors)], alpha=0.2)
            ax.annotate(a, xy=((t_start + 0.5 * (t_end - t_start)) / t_max, 0.0), rotation=-45,
                        xycoords='axes fraction', fontsize=10, xytext=(-20, 5), textcoords='offset points')
            t_start = t_end
            i += 1
    if ts[t_start] != ts[-1]:
        ax.axvspan(ts[t_start], ts[-1], facecolor=colors[i % len(colors)], alpha=0.2)
        ax.annotate(a, xy=((t_start + 0.5 * (t_max - t_start)) / t_max, 0.0), rotation=-45,
                    xycoords='axes fraction', fontsize=10, xytext=(-30, 5), textcoords='offset points')
    return ax


def plot_predictions(ego_agent: ip.MCTSAgent,
                     agents: Dict[int, ip.Agent],
                     ax: plt.Axes,
                     debug: bool = False) -> plt.Axes:
    x, y = 0., 1.
    dx, dy = 0.5, 0.05
    ax.text(x, y, "Goal Prediction Probabilities", fontsize="large")
    y -= 2 * dy
    for i, (aid, goals_probs) in enumerate(ego_agent.goal_probabilities.items()):
        if i > 0 and i % 2 == 0:
            x += dx
            y = 0.9
        ax.text(x, y, f"Agent {aid}:", fontsize="medium")
        y -= dy
        for gid, (goal, gp) in enumerate(goals_probs.goals_probabilities.items()):
            if np.isclose(gp, 0.0):
                continue
            ax.text(x, y,
                    rf"   $P(g^{aid}_{gid}|s^{aid}_{{1:{ego_agent.trajectory_cl.states[-1].time}}})={gp:.3f}$:")
            y -= dy
            for tid, tp in enumerate(goals_probs.trajectories_probabilities[goal]):
                ax.text(x, y, rf"       $P(\hat{{s}}^{{{aid}, {tid}}}_{{1:n}}|g^{aid}_{gid})={tp:.3f}$")
                y -= dy
        y -= dy
    ax.axis("off")

    # Plot prediction trajectories
    if debug:
        attribute = "velocity"
        n_agents = max(2, len(agents) - 1)  # To make sure indexing works later on, at least 2 agents
        n_goals = len(ego_agent.possible_goals)
        subplot_w = 5

        fig, axes = plt.subplots(n_agents, n_goals,
                                 figsize=(n_goals * subplot_w, n_agents * subplot_w,))
        i = 0
        for aid, agent in agents.items():
            if agent.agent_id == ego_agent.agent_id:
                continue
            axes[i, 0].set_ylabel(f"Agent {aid}")
            probs = ego_agent.goal_probabilities[aid]
            for gid, goal in enumerate(probs.goals_probabilities):
                axes[0, gid].set_title(f"{goal[0]}", fontsize=10)
                ax = axes[i, gid]
                opt_trajectory = probs.optimum_trajectory[goal]
                if probs.all_trajectories[goal]:
                    trajectory = probs.all_trajectories[goal][0]
                    ax.plot(opt_trajectory.times, getattr(opt_trajectory, attribute), "r", label="Optimal")
                    ax.plot(trajectory.times, getattr(trajectory, attribute), "b", label="Observed")
            axes[i, 0].legend()
            i += 1
        fig.suptitle(attribute)
        fig.tight_layout()
    return ax


# -----------Explanation plotting functions---------------------
def plot_explanation(
        d_rewards_tuple: Optional[Tuple[pd.DataFrame, pd.DataFrame]],
        coefs: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
        query: Query,
        save_path: str = None,
        uniform_teleological: bool = False):
    """ Plot causal attributions

    Args:
        d_rewards: Reward differences for final causes.
        coefs: Feature coefficient importances for efficient causes.
        save_path: Optional save path for the image.

    Returns:
    """
    reward_map = {
        # "dead": "Goal not\nreached",
        "coll": "Collision",
        "time": "Time Efficiency",
        "angular_velocity": "Angular velocity",
        "curvature": "Curvature",
        "jerk": "Jolt"
    }
    to_drop = ["term", "dead"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if d_rewards_tuple is not None:
        for inx, (d_causes, d_rewards) in enumerate(d_rewards_tuple):
            if d_causes is None:
                axs[inx, 0].text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
                continue
            ax = axs[inx, 0]
            d_rewards = d_rewards.drop(to_drop, axis=1)
            if not uniform_teleological:
                qp = d_rewards["query_present"]
                a = d_rewards[qp].mul(d_causes["p_r_qp"], axis=1)
                b = d_rewards[~qp].mul(d_causes["p_r_qnp"], axis=1)
                d_rewards = pd.concat([a, b], axis=0).sort_index()[d_rewards.columns]
                d_rewards["query_present"] = qp
            d_rewards = d_rewards.rename(reward_map, axis=1)
            d_rewards = d_rewards.melt(id_vars="query_present", var_name="Factor", value_name="Reward")
            d_rewards = d_rewards.dropna(subset="Reward", axis=0)
            d_causes = d_causes.drop(to_drop, axis=0)
            d_causes = d_causes.rename(reward_map, axis=0)
            sns.barplot(d_rewards, x="Reward", y="Factor", hue="query_present",
                        order=d_causes.index, ax=ax, palette=["red", "blue"])
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles=handles, labels=["Not present", "Present"], title="Query")
            ax.axvline(x=0, color=".5")
            ax.set_title(f"{'Past' if inx == 0 else 'Present-future'} teleological causes")

    # plot past and future efficient causes
    for inx, coef in enumerate(coefs):
        ax = axs[inx, 1]
        if coef is None:
            ax.text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
            continue
        inxs = (-coef.mean(0)).argsort()
        coef = coef.iloc[:, inxs]
        inxs = np.isclose(coef.mean(0), 0)
        coef_rest = coef.loc[:, inxs].sum(1)
        coef = coef.loc[:, ~inxs]
        coef = pd.concat([coef, coef_rest], axis=1)
        sns.stripplot(data=coef, orient="h", palette="dark:k", alpha=0.5, ax=ax)
        sns.violinplot(data=coef, orient="h", palette="coolwarm", saturation=0.5,
                       whis=10, width=.8, scale="count", ax=ax)
        ax.axvline(x=0, color=".5")
        ax.set_xlabel(f"({'b' if inx == 1 else 'c'}) Coefficient importance")
        if inx == 0:
            ax.set_title(f"Past causes ({query.tau})")
            ax.set_xlabel("")
        else:
            ax.set_title(f"Present-future causes ({query.t_action})")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
=======
from typing import Dict, List, Optional, Tuple

import igp2 as ip
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .query import Query


# -----------Simulation plotting functions---------------------


def plot_simulation(simulation: ip.simplesim.Simulation, axes: plt.Axes = None, debug: bool = False) \
        -> (plt.Figure, plt.Axes):
    """ Plot the current agents and the road layout for visualisation purposes.

    Args:
        simulation: The simulation to plot.
        axes: Axis to draw on
        debug: If True then plot diagnostic information.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig = plt.gcf()
    fig.suptitle(f"T={simulation.t}")

    color_map_ego = plt.cm.get_cmap('Reds')
    color_map_non_ego = plt.cm.get_cmap('Blues')
    color_ego = 'r'
    color_non_ego = 'b'
    color_bar_non_ego = None

    ax = axes[0]
    ip.plot_map(simulation.scenario_map, markings=True, hide_road_bounds_in_junction=True, ax=ax)
    for agent_id, agent in simulation.agents.items():
        if agent is None or not agent.alive:
            continue

        if isinstance(agent, ip.MCTSAgent):
            color = color_ego
            color_map = color_map_ego
        else:
            color = color_non_ego
            color_map = color_map_non_ego

        path = None
        velocity = None
        if isinstance(agent, ip.MacroAgent) and agent.current_macro is not None:
            path = agent.current_macro.current_maneuver.trajectory.path
            velocity = agent.current_macro.current_maneuver.trajectory.velocity
        elif isinstance(agent, ip.TrajectoryAgent) and agent.trajectory is not None:
            path = agent.trajectory.path
            velocity = agent.trajectory.velocity

        agent_plot = None
        if path is not None and velocity is not None:
            agent_plot = ax.scatter(path[:, 0], path[:, 1], c=velocity, cmap=color_map, vmin=-4, vmax=20, s=8)

        vehicle = agent.vehicle
        pol = plt.Polygon(vehicle.boundary, color=color)
        ax.add_patch(pol)
        ax.text(*agent.state.position, agent_id)
        if isinstance(agent, ip.MCTSAgent) and agent_plot is not None:
            plt.colorbar(agent_plot, ax=ax)
            plt.text(0, 0.1, 'Current Velocity: ' + str(agent.state.speed), horizontalalignment='left',
                     verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0, 0.05, 'Current Macro Action: ' + agent.current_macro.__repr__(), horizontalalignment='left',
                     verticalalignment='bottom', transform=ax.transAxes)
            plt.text(0, 0, 'Current Maneuver: ' + agent.current_macro.current_maneuver.__repr__(),
                     horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

            # Plot goals
            for gid, goal in enumerate(agent.possible_goals):
                loc = goal.center
                ax.plot(*loc, "ro")
                ax.plot(*loc, "kx")
                ax.text(*loc, gid)

            # Plot goal probabilities
            plot_predictions(agent, simulation.agents, axes[1], debug)
        elif isinstance(agent, ip.TrajectoryAgent) and color_bar_non_ego is None:
            color_bar_non_ego = plt.colorbar(agent_plot, location="left")
        plt.text(*agent.state.position, agent_id)

    if debug:
        plot_diagnostics(simulation.agents, simulation.actions)

    return fig, axes


def plot_diagnostics(agents: Dict[int, ip.Agent], actions: Dict[int, List[ip.Action]]) -> (plt.Figure, plt.Axes):
    # attributes = ["velocity", "heading", "angular_velocity"]
    attributes = ["velocity", "acceleration", "jerk"]
    n_agents = len(agents)
    n_attributes = len(attributes)
    subplot_w = 5

    # Plot observations
    fig, axes = plt.subplots(n_agents, n_attributes,
                             figsize=(n_attributes * subplot_w, n_agents * subplot_w))
    if n_agents < 2:
        axes = axes[None, :]
    for i, (aid, agent) in enumerate(agents.items()):
        if agent is None:
            continue
        agent.trajectory_cl.calculate_path_and_velocity()
        ts = agent.trajectory_cl.times
        for j, attribute in enumerate(attributes):
            ax = axes[i, j]
            ys = getattr(agent.trajectory_cl, attribute)
            ys = np.round(ys, 4)
            ax.plot(ts, ys, label="Observed")
            ax.scatter(ts, ys, s=5)

            # Plot target velocities
            if attribute == "velocity":
                ys = [action.target_speed for action in actions[aid]]
                ys = [ys[0]] + ys
                ax.plot(ts, ys, c="red", label="Target")
                ax.scatter(ts, ys, s=5, c="red")
            axes[0, j].set_title(attribute)
            plot_maneuvers(agent, ax)
        axes[i, 0].set_ylabel(f"Agent {aid}")
        axes[i, 0].legend()
    fig.tight_layout()
    return fig, axes


def plot_maneuvers(agent: ip.Agent, ax: plt.Axes) -> plt.Axes:
    man_list = np.array([state.maneuver for state in agent.trajectory_cl.states])
    man_list[0] = man_list[1]
    ts = agent.trajectory_cl.times
    colors = ["red", "blue", "green"]
    t_start = 0
    i = 0
    t_max = len(man_list)
    for t_end, (a, b) in enumerate(zip(man_list[:-1], man_list[1:]), 1):
        if a != b:
            ax.axvspan(ts[t_start], ts[t_end], facecolor=colors[i % len(colors)], alpha=0.2)
            ax.annotate(a, xy=((t_start + 0.5 * (t_end - t_start)) / t_max, 0.0), rotation=-45,
                        xycoords='axes fraction', fontsize=10, xytext=(-20, 5), textcoords='offset points')
            t_start = t_end
            i += 1
    if ts[t_start] != ts[-1]:
        ax.axvspan(ts[t_start], ts[-1], facecolor=colors[i % len(colors)], alpha=0.2)
        ax.annotate(a, xy=((t_start + 0.5 * (t_max - t_start)) / t_max, 0.0), rotation=-45,
                    xycoords='axes fraction', fontsize=10, xytext=(-30, 5), textcoords='offset points')
    return ax


def plot_predictions(ego_agent: ip.MCTSAgent,
                     agents: Dict[int, ip.Agent],
                     ax: plt.Axes,
                     debug: bool = False) -> plt.Axes:
    x, y = 0., 1.
    dx, dy = 0.5, 0.05
    ax.text(x, y, "Goal Prediction Probabilities", fontsize="large")
    y -= 2 * dy
    for i, (aid, goals_probs) in enumerate(ego_agent.goal_probabilities.items()):
        if i > 0 and i % 2 == 0:
            x += dx
            y = 0.9
        ax.text(x, y, f"Agent {aid}:", fontsize="medium")
        y -= dy
        for gid, (goal, gp) in enumerate(goals_probs.goals_probabilities.items()):
            if np.isclose(gp, 0.0):
                continue
            ax.text(x, y,
                    rf"   $P(g^{aid}_{gid}|s^{aid}_{{1:{ego_agent.trajectory_cl.states[-1].time}}})={gp:.3f}$:")
            y -= dy
            for tid, tp in enumerate(goals_probs.trajectories_probabilities[goal]):
                ax.text(x, y, rf"       $P(\hat{{s}}^{{{aid}, {tid}}}_{{1:n}}|g^{aid}_{gid})={tp:.3f}$")
                y -= dy
        y -= dy
    ax.axis("off")

    # Plot prediction trajectories
    if debug:
        attribute = "velocity"
        n_agents = max(2, len(agents) - 1)  # To make sure indexing works later on, at least 2 agents
        n_goals = len(ego_agent.possible_goals)
        subplot_w = 5

        fig, axes = plt.subplots(n_agents, n_goals,
                                 figsize=(n_goals * subplot_w, n_agents * subplot_w,))
        i = 0
        for aid, agent in agents.items():
            if agent.agent_id == ego_agent.agent_id:
                continue
            axes[i, 0].set_ylabel(f"Agent {aid}")
            probs = ego_agent.goal_probabilities[aid]
            for gid, goal in enumerate(probs.goals_probabilities):
                axes[0, gid].set_title(f"{goal[0]}", fontsize=10)
                ax = axes[i, gid]
                opt_trajectory = probs.optimum_trajectory[goal]
                if probs.all_trajectories[goal]:
                    trajectory = probs.all_trajectories[goal][0]
                    ax.plot(opt_trajectory.times, getattr(opt_trajectory, attribute), "r", label="Optimal")
                    ax.plot(trajectory.times, getattr(trajectory, attribute), "b", label="Observed")
            axes[i, 0].legend()
            i += 1
        fig.suptitle(attribute)
        fig.tight_layout()
    return ax


# -----------Explanation plotting functions---------------------
def plot_explanation(
        d_rewards_tuple: Optional[Tuple[pd.DataFrame, pd.DataFrame]],
        coefs: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
        query: Query,
        save_path: str = None,
        uniform_teleological: bool = False):
    """ Plot causal attributions

    Args:
        d_rewards: Reward differences for final causes.
        coefs: Feature coefficient importances for efficient causes.
        save_path: Optional save path for the image.

    Returns:
    """
    reward_map = {
        # "dead": "Goal not\nreached",
        "coll": "Collision",
        "time": "Time Efficiency",
        "angular_velocity": "Angular velocity",
        "curvature": "Curvature",
        "jerk": "Jolt"
    }
    to_drop = ["term", "dead"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if d_rewards_tuple is not None:
        for inx, (d_causes, d_rewards) in enumerate(d_rewards_tuple):
            if d_causes is None:
                axs[inx, 0].text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
                continue
            ax = axs[inx, 0]
            d_rewards = d_rewards.drop(to_drop, axis=1)
            if not uniform_teleological:
                qp = d_rewards["query_present"]
                a = d_rewards[qp].mul(d_causes["p_r_qp"], axis=1)
                b = d_rewards[~qp].mul(d_causes["p_r_qnp"], axis=1)
                d_rewards = pd.concat([a, b], axis=0).sort_index()[d_rewards.columns]
                d_rewards["query_present"] = qp
            d_rewards = d_rewards.rename(reward_map, axis=1)
            d_rewards = d_rewards.melt(id_vars="query_present", var_name="Factor", value_name="Reward")
            d_rewards = d_rewards.dropna(subset="Reward", axis=0)
            d_causes = d_causes.drop(to_drop, axis=0)
            d_causes = d_causes.rename(reward_map, axis=0)
            sns.barplot(d_rewards, x="Reward", y="Factor", hue="query_present",
                        order=d_causes.index, ax=ax, palette=["red", "blue"])
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles=handles, labels=["Not present", "Present"], title="Query")
            ax.axvline(x=0, color=".5")
            ax.set_title(f"{'Past' if inx == 0 else 'Present-future'} teleological causes")

    # plot past and future efficient causes
    for inx, coef in enumerate(coefs):
        ax = axs[inx, 1]
        if coef is None:
            ax.text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
            continue
        inxs = (-coef.mean(0)).argsort()
        coef = coef.iloc[:, inxs]
        inxs = np.isclose(coef.mean(0), 0)
        coef_rest = coef.loc[:, inxs].sum(1)
        coef = coef.loc[:, ~inxs]
        coef = pd.concat([coef, coef_rest], axis=1)
        sns.stripplot(data=coef, orient="h", palette="dark:k", alpha=0.5, ax=ax)
        sns.violinplot(data=coef, orient="h", palette="coolwarm", saturation=0.5,
                       whis=10, width=.8, scale="count", ax=ax)
        ax.axvline(x=0, color=".5")
        ax.set_xlabel(f"({'b' if inx == 1 else 'c'}) Coefficient importance")
        if inx == 0:
            ax.set_title(f"Past causes ({query.tau})")
            ax.set_xlabel("")
        else:
            ax.set_title(f"Present-future causes ({query.t_action})")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
>>>>>>> 90fede0243d6f17b426f2d2d9ba80de72684e61a:src/cema/xavi/plotting.py
