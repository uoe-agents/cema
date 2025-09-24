# Causal Explanations for Sequential Decision-Making in Multi-Agent Systems
This is the code repository for our [paper](https://arxiv.org/abs/2302.10809) "Causal Social Explanations for Stochastic Sequential Multi-Agent Decision-Making" by Gyevnar et al [1] published at AAMAS 2024 in Auckland, New Zealand.
The following document describes the installation process and usage of the code to reproduce our results.

This software relies on the [IGP2](https://github.com/uoe-agents/IGP2) package to simulate road interactions in scenarios with an autonomous vehicle.

## Please Cite
If you use our code in your work then please cite our [paper](https://arxiv.org/abs/2302.10809):
```text
@inproceedings{gyevnar2024causal,
  author = {Gyevnar, Balint and Wang, Cheng and Lucas, Christopher G. and Cohen, Shay B. and Albrecht, Stefano V.},
  title = {Causal Explanations for Sequential Decision-Making in Multi-Agent Systems},
  year = {2024},
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
  address = {Richland, SC},
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
  pages = {771-779},
  numpages = {9},
  location = {Auckland, New Zealand},
  series = {AAMAS '24},
}
```

## Getting Started

### Recommended Installation

The recommended way to install CEMA is using `uv`, which is a super-fast Python package manager which supports pip. You can download `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/). Once downloaded, you can run the following commands to download and use CEMA:
```commandline
uv pip install git+https://github.com/uoe-agents/cema.git
cd cema
uv run cema
```

If you wish to use the LLM-based functionallity of CEMA, you should run the following command:
```commandline
uv pip install "cema[llm] @ git+https://github.com/uoe-agents/cema.git"
cd cema
uv run cema
```

### Alternative Installation

Alternatively, you can follow the standard pip installation method:
```commandline
pip install git+https://github.com/uoe-agents/cema.git
cd cema
python -m cema
```

## Reproducing results

To reproduce results from the paper you can use the already existing scripts.
Simply run the script ```python run.py``` to generate explanations by specifying various commandline arguments which you can get help with using the ```-h``` commandline argument.

For example, to obtain explanations of the autonomous vehicle behaviour in scenario 1, you can run:
```commandline
python run.py 1
```

If you wish to plot the simulation, you can do so by specifying the ```--plot``` commandline argument (currently we only use matplotlib so plotting is rather slow).
Plots of the explanations are generated automatically after executing each query. 

To reproduce all our results you can just run the ```bash eval_all.sh``` script to generate all results and outputs. 

To generate results for a particular scenario and query, first run the scenario and the corresponding queries (currently defined in ```final_queries.json```) and save the results using the following command:
```commandline
python run.py $scenario_id$ --query_path scenarios/queries/final_queries.json --save_causes 
```

After saving the results, you can plot the causal attributions and generate the natural language explanations for a given scenario and query by running the following command:
```commandline
python eval.py $scenario_id$ $query_index$
```

## Configuration

### Map file
The road layout in each scenario is defined using the ASAM OpenDrive standard v1.6.
You can add your own maps for testing in the folder ```scenarios/maps```.

### Configuration file
You can add new scenarios for testing by creating a new configuration file in the ```scenarios/configs``` folder. 
The file should be a well-formatted JSON file, that can be read using Python's built-in tools.
If the new scenario has ID N, then the filename should be ```scenarioN.json```.
For an example usage you can refer to ```scenarios/configs/scenario1.json```.
The configuration file has the following structure, fields, and types. 

Note:
1. Fields with an exclamation mark (!) must always be included.
2. Angles should be in radians in the range [-pi,pi].

```text
scenarios/configs/scenarioN.json
|____scenario: dict
|    |____!map_path: str    The path to the OpenDrive *.xodr file for the road layout
|    |____max_speed: float  The maximum speed allowed in the scenario
|    |____fps: int          Execution frequency of the simulation
|    |____seed: int         Random seed
|    |____max_steps: int    If given, stop executing simulation after this many steps
|____agents: list
|    |____agent*: dict     List of agents to spawn
|    |    |____!id: int        The numerical identifier of the agent
|    |    |____!type: str      The type of the agent. Currently, only MCTSAgent and TrafficAgent is supported
|    |    |____!spawn: dict
|    |    |   |____!box: dict                The area to spawn the vehicle in
|    |    |   |   |____!center: list[float]      The center of the spawn area
|    |    |   |   |____!length: float            The length of the spawn area
|    |    |   |   |____!width: float             The width of the spawn area
|    |    |   |   |____!heading: float           The heading of the spawn area
|    |    |   |____velocity: list[float]     The spawn velocity range for random sampling
|    |    |____!goal: dict
|    |    |   |____!box: dict                The area for the vehicle to reach
|    |    |   |   |____!center: list[float]      The center of the goal area
|    |    |   |   |____!length: float            The length of the goal area
|    |    |   |   |____!width: float             The width of the goal area
|    |    |   |   |____!heading: float           The heading of the goal area

------The following is for MCTSAgents (and subclasses) only------
|    |    |____cost_factors: dict           The cost weighing factors for IGP2 goal recognition
|    |    |    |____time: float                  Time to goal
|    |    |    |____velocity: float              Average velocity
|    |    |    |____acceleration: float          Average acceleration
|    |    |    |____jerk: float                  Average jerk
|    |    |    |____heading: float               Average heading
|    |    |    |____angular_velocity: float      Average angular velocity
|    |    |    |____angular_acceleration: float  Average angular acceleration
|    |    |    |____curvature: float             Average curvature
|    |    |    |____safety: float                Safety of trajectory
|    |    |____mcts: dict                  MCTS parameters
|    |    |    |____t_update: float                Runtime period of MCTS in seconds
|    |    |    |____n_simulations: int             Number of rollouts in MCTS
|    |    |    |____store_results: str             Either 'all' or 'final'. If absent, results are not stored
|    |    |    |____reward_factors: dict  MCTS reward factors
|    |    |    |    |____time: float                    Time to goal
|    |    |    |    |____jerk: float                    Average jerk
|    |    |    |    |____angular_velocity: float        Average angular velocity
|    |    |    |    |____curvature: float               Average curvature
|    |    |    |    |____safety: float                  Safety term. Currently, not used.
|    |    |____view_radius: float         Radius of circle centered on the vehicle, in which it can observe the environment
|    |    |____kinematic: bool            Whether to use a bicycle-model vehicle for simulations or a trajectory state-following vehicle;
                                               the latter must be used for CARLA.
------The following is for XAVIAgents only------                                               
|    |    |____explainer: dict
|    |    |    |____tau_limits: list[int]      Lower and upper bounds on the distance of tau from t_action
|    |    |    |____cf_n_trajectories: int     Number of trajectories to generate for non-ego agents with A*
|    |    |    |____cf_n_simulations: int      Number of simulations for counterfactual MCTS simulation
|    |    |    |____cf_max_depth: int          Maximum search depth for counterfactual MCTS simulation   
|    |    |    |____tau_limits: list[float]    A list of two floats that specify tau_min and tau_max
|    |    |    |____time_limits: list[float]   Limits on how far back to the past and into the future we look for explanations
|    |    |    |____alpha: float               Additive smoothing weight parameter                               
```

### Query file
For a given scenario you can define a set of queries that you wish to be explained during the simulation of the scenario.

A list of queries is given as a well-formed JSON file.
If your scenario has ID N, then you can place your query file into the folder ```scenarios/queries``` with the filename ```query_scenarioN.json``` and our script will automatically use the queries in that file.
Alternatively, you can specify the query path using the ```--query_path``` commandline argument. 

A query file should have the following format:
```text
scenarios/queries/query_scenarioN.json
|____query_list: list       A list of queries
|    |____query: dict       A query defined as dictionary
|    |    |____type: str                The query type: either why, whynot, whatif, or what
|    |    |____agent_id: int            The ID of the vehicle to explain. Usually 0 for the ego.
|    |    |____action: list[str]|str    The list of actions to explain. Or a single action to explain.
|    |    |____tense: str               The grammatical tense of the user question. Either, past, present, or future.
|    |    |____factual: list[str]|str   The factual list of actions used for non-negative whatif and whynot queries.
|    |    |____t_query: int             The timestep to execute the query on.
|    |    |____t_action: int            The start timestep of the actions in question. Only needed for what type query.
|    |    |____negative: bool           Whether the sentence is negated or not.
```

## References
[1] B. Gyevnar, C. Wang, C.G. Lucas, S.B. Cohen, S.V. Albrecht; Causal Explanations for Sequential Decision-Making in Multi-Agent System, _Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems_ (AAMAS '24), Auckland, New Zealand, 2024 
