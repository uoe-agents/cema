<<<<<<< HEAD:scenarios/plotting.py
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

logger = logging.getLogger(__name__)

reward_map = {
    "coll": "Collision",
    "time": "Time Efficiency",
    "angular_velocity": "Angular velocity",
    "curvature": "Curvature",
    "jerk": "Jolt"
}

sid_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5, 10: 6, 11: 7
}


def plot_explanation(
        d_rewards_tuple: Optional[Tuple[pd.DataFrame, pd.DataFrame]],
        coefs: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
        query_string: str,
        save_path: str,
        max_n_causes: int = 14,
        scenario_id: int =  0,
        figsize: Tuple[float, float] = (5., 3.)) -> None:
    """ Plot causal attributions and save each on individual files.

    Args:
        d_rewards: Reward differences for final causes.
        coefs: Feature coefficient importances for efficient causes.
        query: The query object.
        query_string: The query string to identify the current query.
        save_path: Optional save path for the image.
    """
    sns.set_style("whitegrid")
    scenario_id = sid_map[scenario_id]
    to_drop = ["term", "dead"]
    if d_rewards_tuple is not None:
        for time, (d_causes, d_rewards) in zip(["past", "future"], d_rewards_tuple):
            fig, ax = plt.subplots(figsize=(4, 2.5))
            if d_causes is None:
                ax.text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
                continue
            d_rewards = d_rewards.drop(to_drop, axis=1)
            qp = d_rewards["query_present"]
            a = d_rewards[qp].mul(d_causes["p_r_qp"], axis=1)
            b = d_rewards[~qp].mul(d_causes["p_r_qnp"], axis=1)
            d_rewards = pd.concat([a, b], axis=0).sort_index()[d_rewards.columns]
            d_rewards["query_present"] = qp
            d_rewards = d_rewards.rename(reward_map, axis=1)
            d_rewards = d_rewards.melt(id_vars="query_present", var_name="Factor", value_name="Reward")
            d_rewards = d_rewards.rename({"query_present": "Query Present"}, axis=1)
            d_rewards = d_rewards.dropna(subset="Reward", axis=0)
            d_causes = d_causes.drop(to_drop, axis=0)
            d_causes = d_causes.rename(reward_map, axis=0)
            sns.barplot(d_rewards, x="Reward", y="Factor", hue="Query Present",
                        order=d_causes.index, ax=ax, palette="tab10")
            ax.axvline(x=0, color=".5")
            time_str = "past" if time == "past" else "present"
            # ax.set_title(f"{time_str} teleological causes")
            ax.set_ylabel("")
            ax.set_xlabel(rf"Expected Reward (S{scenario_id}, $\tau={time_str}$)")
            ax.legend(loc="lower right", title="Query present?")
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, f"q_{time}_teleological_{query_string}.pdf"))

    # plot past and future mechanistic causes
    for time, coef in zip(["past", "future"], coefs):
        fig, ax = plt.subplots(figsize=figsize)
        if coef is None:
            ax.text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
            continue
        inxs = (-coef.mean(0)).argsort()
        if coef.shape[1] > max_n_causes:
            lim = max_n_causes // 2
            inxs = np.concatenate([inxs[:lim], inxs[-lim:]])
        coef = coef.iloc[:, inxs]
        inxs = np.isclose(coef.mean(0), 0)
        coef_rest = coef.loc[:, inxs].sum(1)
        coef = coef.loc[:, ~inxs]
        coef = pd.concat([coef, coef_rest], axis=1)
        coef.rename({coef.columns[-1]: "Other features"}, axis=1, inplace=True)
        coef.rename(lambda x: x.replace("xmacro_", "x").replace("nmacro_", "n")[:], axis=1, inplace=True)
        sns.stripplot(data=coef, orient="h", palette="dark:k", alpha=0.5, ax=ax, size=3)
        sns.violinplot(data=coef, orient="h", palette="coolwarm", saturation=0.5,
                       width=.8, density_norm="count", ax=ax)
        ax.axvline(x=0, color=".5")
        time_str = "past" if time == "past" else "present"
        ax.set_xlabel(rf"Coefficient importance ($\tau={time_str}$)")
        # n_unique = coef.shape[1] - 1
        # ax.set_title(f"Largest {n_unique} {time_str} mechanistic causes")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"q_{time}_mechanistic_{query_string}.pdf"), bbox_inches='tight')


def process_evaluation_dictionary(
        evaluation_results: Dict[Union[int, float], List[Any]],
        max_n_causes: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def renamer(x: str) -> str:
        return x.replace("xmacro_", "x").replace("nmacro_", "n")[:20]

    teleological_df = []
    mechanistic_df = []
    logger.info("Processing evaluation results for plotting . . .")
    for value, samples in tqdm(evaluation_results.items()):
        for causes in samples:
            if causes is None:
                continue
            if len(causes) == 3:
                causes = causes[1:]
            for time, (d_causes, d_rewards) in zip(["past", "future"], causes[0]):
                if d_causes is None:
                    continue
                d_rewards = d_rewards.drop(["term", "dead"], axis=1)
                qp = d_rewards["query_present"]
                a = d_rewards[qp].mul(d_causes["p_r_qp"], axis=1)
                b = d_rewards[~qp].mul(d_causes["p_r_qnp"], axis=1)
                d_rewards = pd.concat([a, b], axis=0).sort_index()[d_rewards.columns]
                d_rewards["query_present"] = qp
                d_rewards = d_rewards.groupby("query_present").mean().diff().iloc[1].dropna().rename(reward_map)
                teleological_causes = {"value": value, "time": time}
                teleological_causes.update(d_rewards.to_dict())
                teleological_df.append(teleological_causes)

            for time, coefs in zip(["past", "future"], causes[1][0:2]):
                if coefs is None:
                    continue
                inxs = (-coefs.mean(0)).argsort()
                coefs = coefs.iloc[:, inxs]
                inxs = np.isclose(coefs.mean(0), 0)
                coefs = coefs.loc[:, ~inxs]
                mechanistic_causes = {"value": value, "time": time}
                mechanistic_causes.update(coefs.mean(0).rename(renamer).to_dict())
                mechanistic_df.append(mechanistic_causes)

    teleological_df = pd.DataFrame(teleological_df).melt(id_vars=["value", "time"],
                                                         var_name="Factor", value_name="Reward")
    teleological_df = teleological_df.dropna(axis=0, subset="Factor")

    mechanistic_df = pd.DataFrame(mechanistic_df)
    n_causes = mechanistic_df.shape[1] - 2
    mechanistic_df = mechanistic_df.melt(id_vars=["value", "time"], var_name="Feature", value_name="Causal Effect")
    # Limit the number of causes visible on the plot if necessary
    if n_causes > max_n_causes:
        sorted_features = (mechanistic_df.groupby(["time", "Feature"])
                           .mean("Causal Effect")
                           .sort_values(by="Causal Effect", ascending=False, key=abs))
        if "past" in sorted_features.index:
            top_n_past = sorted_features.xs("past", drop_level=False).index[:max_n_causes]
        else:
            top_n_past = []
        top_n_future = sorted_features.xs("future", drop_level=False).index[:max_n_causes]
        indexer = mechanistic_df.set_index(["time", "Feature"]).index
        mechanistic_df = mechanistic_df[indexer.isin(top_n_past) | indexer.isin(top_n_future)]
    mechanistic_df = mechanistic_df.dropna(axis=0, subset="Causal Effect")

    return teleological_df, mechanistic_df


def plot_sampling_results(
        sampling_results: Dict[int, List[Any]],
        output_path: str,
        query_str: str,
        max_n_causes: int = 14) -> None:
    """ Plots the sampling size robustness results.

    Args:
        sampling_results: A dictionary of lists of sampling results.
        output_path: The path to save the plots.
        query_str: The query string to identify the current query.
        max_n_causes: The maximum number of causes to display on the plot.
    """
    sns.set_style("whitegrid")

    teleological_df, mechanistic_df = process_evaluation_dictionary(sampling_results, max_n_causes=max_n_causes)
    teleological_df.rename({"value": "Sample Size"}, axis=1, inplace=True)
    mechanistic_df.rename({"value": "Sample Size"}, axis=1, inplace=True)

    if "scenario_1/" in output_path:
        teleological_df["Reward"][teleological_df["time"] == "future"] *= -1.0
        mechanistic_df["Causal Effect"][mechanistic_df["time"] == "future"] *= -1.0

    logger.info("Plotting sampling robustness results . . .")
    for time in ["past", "future"]:
        df = teleological_df[teleological_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Factor").mean("Reward").sort_values("Reward", ascending=False).index
        g = sns.relplot(data=df, x="Sample Size", y="Reward", hue="Factor", style="Factor",
                        kind="line", markers=True, hue_order=hue_order)
        time_str = r"$\tau=past$" if time == "past" else r"$\tau=present$"
        # g.figure.suptitle(f"Teleological causes ({time_str})")
        g.set_ylabels(r"$\Delta$Reward")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"sampling_{time}_teleological_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()

    for time in ["past", "future"]:
        df = mechanistic_df[mechanistic_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Feature").mean("Causal Effect").sort_values("Causal Effect", ascending=False).index
        g = sns.relplot(data=df, x="Sample Size", y="Causal Effect", hue="Feature",
                        style="Feature", kind="line", markers=True, hue_order=hue_order)
        time_str = r"$\tau=past$" if time == "past" else r"$\tau=present$"
        # g.figure.suptitle(f"Mechanistic causes ({time_str})")
        n_features = df["Feature"].nunique()
        g.set_xlabels(r"Sample Size $N_D$")
        g.legend.set_title(f"Largest {n_features} Features")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"sampling_{time}_mechanistic_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()



def plot_distribution_results(
        distribution_results: Dict[float, List[Any]],
        output_path: str,
        query_str: str,
        max_n_causes: int = 14) -> None:
    """ Plots the distribution smoothing robustness results.

    Args:
        distribution_results: A dictionary of lists of distribution results.
        output_path: The path to save the plots.
        query_str: The query string to identify the current query.
        max_n_causes: The maximum number of causes to display on the plot.
    """
    sns.set_style("whitegrid")

    teleological_df, mechanistic_df = process_evaluation_dictionary(distribution_results, max_n_causes=max_n_causes)
    teleological_df.rename({"value": "Smoothing Alpha"}, axis=1, inplace=True)
    mechanistic_df.rename({"value": "Smoothing Alpha"}, axis=1, inplace=True)

    if "scenario_1/" in output_path:
        teleological_df["Reward"][teleological_df["time"] == "future"] *= -1.0
        mechanistic_df["Causal Effect"][mechanistic_df["time"] == "future"] *= -1.0

    logger.info("Plotting distribution robustness results . . .")
    for time in ["past", "future"]:
        df = teleological_df[teleological_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Factor").mean("Reward").sort_values("Reward", ascending=False).index
        g = sns.relplot(data=df, x="Smoothing Alpha", y="Reward", hue="Factor", style="Factor",
                        kind="line", markers=True, hue_order=hue_order)
        time_str = r"$\tau=past$" if time == "past" else r"$\tau=present$"
        # g.figure.suptitle(f"Teleological causes ({time_str})")
        g.set_ylabels(r"$\Delta$Reward")
        g.set_xlabels(r"Smoothing $\alpha$")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"distribution_{time}_teleological_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()

    for time in ["past", "future"]:
        df = mechanistic_df[mechanistic_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Feature").mean("Causal Effect").sort_values("Causal Effect", ascending=False).index
        g = sns.relplot(data=df, x="Smoothing Alpha", y="Causal Effect", hue="Feature",
                        style="Feature", kind="line", markers=True, hue_order=hue_order)
        time_str = r"$\tau=past$" if time == "past" else r"$\tau=present$"
        # g.figure.suptitle(f"Mechanistic causes ({time_str})")
        g.set_xlabels(r"Smoothing $\alpha$")
        n_features = df["Feature"].nunique()
        g.legend.set_title(f"Largest {n_features} Features")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"distribution_{time}_mechanistic_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()
=======
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

logger = logging.getLogger(__name__)

reward_map = {
    "coll": "Collision",
    "time": "Time Efficiency",
    "angular_velocity": "Angular velocity",
    "curvature": "Curvature",
    "jerk": "Jolt"
}


def plot_explanation(
        d_rewards_tuple: Optional[Tuple[pd.DataFrame, pd.DataFrame]],
        coefs: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
        query_string: str,
        save_path: str,
        max_n_causes: int = 14,
        drop_dead: bool = True) -> None:
    """ Plot causal attributions and save each on individual files.

    Args:
        d_rewards: Reward differences for final causes.
        coefs: Feature coefficient importances for efficient causes.
        query_string: The query string for saving plots to file.
        save_path: Optional save path for the image.
        max_n_causes: The maximum number of causes to display on the plot.
        drop_dead: Whether to drop goal not reached state from the rewards.
    """
    sns.set_style("whitegrid")
    to_drop = ["term"]
    if drop_dead:
        to_drop.append("dead")
    if d_rewards_tuple is not None:
        for time, (d_causes, d_rewards) in zip(["past", "future"], d_rewards_tuple):
            fig, ax = plt.subplots()
            if d_causes is None:
                ax.text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
                continue
            d_rewards = d_rewards.drop(to_drop, axis=1)
            qp = d_rewards["query_present"]
            a = d_rewards[qp].mul(d_causes["p_r_qp"], axis=1)
            b = d_rewards[~qp].mul(d_causes["p_r_qnp"], axis=1)
            d_rewards = pd.concat([a, b], axis=0).sort_index()[d_rewards.columns]
            d_rewards["query_present"] = qp
            d_rewards = d_rewards.rename(reward_map, axis=1)
            d_rewards = d_rewards.melt(id_vars="query_present", var_name="Factor", value_name="Reward")
            d_rewards = d_rewards.rename({"query_present": "Query Present"}, axis=1)
            d_rewards = d_rewards.dropna(subset="Reward", axis=0)
            d_causes = d_causes.drop(to_drop, axis=0)
            d_causes = d_causes.rename(reward_map, axis=0)
            sns.barplot(d_rewards, x="Reward", y="Factor", hue="Query Present",
                        order=d_causes.index, ax=ax, palette="tab10")
            ax.axvline(x=0, color=".5")
            time_str = "Past" if time == "past" else "Present-future"
            ax.set_title(f"{time_str} teleological causes")
            ax.set_ylabel("")
            ax.set_xlabel("Expected Reward")
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, f"q_{time}_teleological_{query_string}.pdf"))

    # plot past and future mechanistic causes
    for time, coef in zip(["past", "future"], coefs):
        fig, ax = plt.subplots()
        if coef is None:
            ax.text(0.2, 0.45, "No past causes because \n action starts from $t=1$.", fontsize=14)
            continue
        inxs = (-coef.mean(0)).argsort()
        if coef.shape[1] > max_n_causes:
            lim = max_n_causes // 2
            inxs = np.concatenate([inxs[:lim], inxs[-lim:]])
        coef = coef.iloc[:, inxs]
        inxs = np.isclose(coef.mean(0), 0)
        coef_rest = coef.loc[:, inxs].sum(1)
        coef = coef.loc[:, ~inxs]
        coef = pd.concat([coef, coef_rest], axis=1)
        sns.stripplot(data=coef, orient="h", palette="dark:k", alpha=0.5, ax=ax)
        sns.violinplot(data=coef, orient="h", palette="coolwarm", saturation=0.5,
                       whis=10, width=.8, scale="count", ax=ax)
        ax.axvline(x=0, color=".5")
        ax.set_xlabel("Coefficient importance")
        time_str = "past" if time == "past" else "present-future"
        n_unique = coef.shape[1] - 1
        ax.set_title(f"Largest {n_unique} {time_str} mechanistic causes")
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"q_{time}_mechanistic_{query_string}.pdf"), bbox_inches='tight')


def process_evaluation_dictionary(
        evaluation_results: Dict[Union[int, float], List[Any]],
        max_n_causes: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def renamer(x: str) -> str:
        return x.replace("xmacro", "xm").replace("nmacro", "nm")[:20]

    teleological_df = []
    mechanistic_df = []
    logger.info("Processing evaluation results for plotting . . .")
    for value, samples in tqdm(evaluation_results.items()):
        for causes in samples:
            if causes is None:
                continue
            if len(causes) == 3:
                causes = causes[1:]
            for time, (d_causes, d_rewards) in zip(["past", "future"], causes[0]):
                if d_causes is None:
                    continue
                d_rewards = d_rewards.drop(["term", "dead"], axis=1)
                qp = d_rewards["query_present"]
                a = d_rewards[qp].mul(d_causes["p_r_qp"], axis=1)
                b = d_rewards[~qp].mul(d_causes["p_r_qnp"], axis=1)
                d_rewards = pd.concat([a, b], axis=0).sort_index()[d_rewards.columns]
                d_rewards["query_present"] = qp
                d_rewards = d_rewards.groupby("query_present").mean().diff().iloc[1].dropna().rename(reward_map)
                teleological_causes = {"value": value, "time": time}
                teleological_causes.update(d_rewards.to_dict())
                teleological_df.append(teleological_causes)

            for time, coefs in zip(["past", "future"], causes[1][0:2]):
                if coefs is None:
                    continue
                inxs = (-coefs.mean(0)).argsort()
                coefs = coefs.iloc[:, inxs]
                inxs = np.isclose(coefs.mean(0), 0)
                coefs = coefs.loc[:, ~inxs]
                mechanistic_causes = {"value": value, "time": time}
                mechanistic_causes.update(coefs.mean(0).rename(renamer).to_dict())
                mechanistic_df.append(mechanistic_causes)

    teleological_df = pd.DataFrame(teleological_df).melt(id_vars=["value", "time"],
                                                         var_name="Factor", value_name="Reward")
    teleological_df = teleological_df.dropna(axis=0, subset="Factor")

    mechanistic_df = pd.DataFrame(mechanistic_df)
    n_causes = mechanistic_df.shape[1] - 2
    mechanistic_df = mechanistic_df.melt(id_vars=["value", "time"], var_name="Feature", value_name="Causal Effect")
    # Limit the number of causes visible on the plot if necessary
    if n_causes > max_n_causes:
        sorted_features = (mechanistic_df.groupby(["time", "Feature"])
                           .mean("Causal Effect")
                           .sort_values(by="Causal Effect", ascending=False, key=abs))
        top_n_past = sorted_features.xs("past", drop_level=False).index[:max_n_causes]
        top_n_future = sorted_features.xs("future", drop_level=False).index[:max_n_causes]
        indexer = mechanistic_df.set_index(["time", "Feature"]).index
        mechanistic_df = mechanistic_df[indexer.isin(top_n_past) | indexer.isin(top_n_future)]
    mechanistic_df = mechanistic_df.dropna(axis=0, subset="Causal Effect")

    return teleological_df, mechanistic_df


def plot_sampling_results(
        sampling_results: Dict[int, List[Any]],
        output_path: str,
        query_str: str,
        max_n_causes: int = 14) -> None:
    """ Plots the sampling size robustness results.

    Args:
        sampling_results: A dictionary of lists of sampling results.
        output_path: The path to save the plots.
        query_str: The query string to identify the current query.
        max_n_causes: The maximum number of causes to display on the plot.
    """
    sns.set_style("whitegrid")

    teleological_df, mechanistic_df = process_evaluation_dictionary(sampling_results, max_n_causes=max_n_causes)
    teleological_df.rename({"value": "Sample Size"}, axis=1, inplace=True)
    mechanistic_df.rename({"value": "Sample Size"}, axis=1, inplace=True)

    logger.info("Plotting sampling robustness results . . .")
    for time in ["past", "future"]:
        df = teleological_df[teleological_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Factor").mean("Reward").sort_values("Reward", ascending=False).index
        g = sns.relplot(data=df, x="Sample Size", y="Reward", hue="Factor", style="Factor",
                        kind="line", markers=True, hue_order=hue_order)
        time_str = "Past" if time == "past" else "Present-future"
        g.figure.suptitle(f"{time_str} teleological causes")
        g.set_ylabels(r"$\Delta$Reward")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"sampling_{time}_teleological_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()

    for time in ["past", "future"]:
        df = mechanistic_df[mechanistic_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Feature").mean("Causal Effect").sort_values("Causal Effect", ascending=False).index
        g = sns.relplot(data=df, x="Sample Size", y="Causal Effect", hue="Feature",
                        style="Feature", kind="line", markers=True, hue_order=hue_order)
        time_str = "Past" if time == "past" else "Present-future"
        g.figure.suptitle(f"{time_str} mechanistic causes")
        n_features = df["Feature"].nunique()
        g.legend.set_title(f"Largest {n_features} Features")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"sampling_{time}_mechanistic_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()



def plot_distribution_results(
        distribution_results: Dict[float, List[Any]],
        output_path: str,
        query_str: str,
        max_n_causes: int = 14) -> None:
    """ Plots the distribution smoothing robustness results.

    Args:
        distribution_results: A dictionary of lists of distribution results.
        output_path: The path to save the plots.
        query_str: The query string to identify the current query.
        max_n_causes: The maximum number of causes to display on the plot.
    """
    sns.set_style("whitegrid")

    teleological_df, mechanistic_df = process_evaluation_dictionary(distribution_results, max_n_causes=max_n_causes)
    teleological_df.rename({"value": "Smoothing Alpha"}, axis=1, inplace=True)
    mechanistic_df.rename({"value": "Smoothing Alpha"}, axis=1, inplace=True)

    logger.info("Plotting distribution robustness results . . .")
    for time in ["past", "future"]:
        df = teleological_df[teleological_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Factor").mean("Reward").sort_values("Reward", ascending=False).index
        g = sns.relplot(data=df, x="Smoothing Alpha", y="Reward", hue="Factor", style="Factor",
                        kind="line", markers=True, hue_order=hue_order)
        time_str = "Past" if time == "past" else "Present-future"
        g.figure.suptitle(f"{time_str} teleological causes")
        g.set_ylabels(r"$\Delta$Reward")
        g.set_xlabels(r"Smoothing $\alpha$")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"distribution_{time}_teleological_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()

    for time in ["past", "future"]:
        df = mechanistic_df[mechanistic_df["time"] == time]
        if df.empty:
            continue
        hue_order = df.groupby("Feature").mean("Causal Effect").sort_values("Causal Effect", ascending=False).index
        g = sns.relplot(data=df, x="Smoothing Alpha", y="Causal Effect", hue="Feature",
                        style="Feature", kind="line", markers=True, hue_order=hue_order)
        time_str = "Past" if time == "past" else "Present-future"
        g.figure.suptitle(f"{time_str} mechanistic causes")
        g.set_xlabels(r"Smoothing $\alpha$")
        n_features = df["Feature"].nunique()
        g.legend.set_title(f"Largest {n_features} Features")
        g.ax.axhline(y=0, color=".5")
        g.tight_layout()
        plt.savefig(os.path.join(output_path, f"distribution_{time}_mechanistic_{query_str}.pdf"), bbox_inches='tight')
    # plt.show()
>>>>>>> 90fede0243d6f17b426f2d2d9ba80de72684e61a:src/cema/script/plotting.py
