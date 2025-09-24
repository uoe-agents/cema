""" Utility functions for the language model verbalization. """
from typing import List

import numpy as np
import igp2 as ip


############  Utility functions  ############

def subsample_trajectory(
        trajectory: ip.StateTrajectory,
        f_subsample: int) -> ip.StateTrajectory:
    """ Subsample a trajectory by a factor of f_subsample.

        Args:
            trajectory: The trajectory to subsample.
            f_subsample: The factor to subsample by.

        Returns:
            The subsampled trajectory of type ip.StateTrajectory.
    """
    ts = trajectory.times
    num_frames = np.ceil(len(ts) / f_subsample).astype(int)
    points = np.linspace(ts[0], ts[-1], num_frames)

    xs_r = np.interp(points, ts, trajectory.path[:, 0])
    ys_r = np.interp(points, ts, trajectory.path[:, 1])
    v_r = np.interp(points, ts, trajectory.velocity)
    a_r = np.interp(points, ts, trajectory.acceleration)
    h_r = np.interp(points, ts, trajectory.heading)
    path = np.c_[xs_r, ys_r]

    states = []
    for i in range(num_frames):
        states.append(ip.AgentState(time=i * f_subsample,
                                position=path[i],
                                velocity=v_r[i],
                                acceleration=a_r[i],
                                heading=h_r[i]))

    return ip.StateTrajectory(
        trajectory.fps // f_subsample,
        states=states,
        path=path,
        velocity=v_r)


def ndarray2str(array: np.ndarray, precision: int = 2) -> str:
    """ Format a numpy array to a string.

        Args:
            array: The array to format.
            precision: The number of decimal places to use.

        Returns:
            The formatted string.
    """
    ret = np.array2string(array, separator=", ", precision=precision, suppress_small=True)
    ret = ret.replace("\n", "")
    ret = " ".join(ret.split())
    return ret


############  Grammatical transformations  ############

def to_gerund(words: List[str]) -> str:
    """ Convert a list of words where the first word is a verb to gerund form."""
    ends_with_e = words[0][-1] == "e"
    words[0] = (words[0][:1] if ends_with_e else words[0]) + "ing"
    return " ".join(words)


def to_past(words: List[str], participle: bool = False):
    """ Convert a list of words where the first word is a verb to past tense."""
    if words[0] == "go":
        words[0] = "gone" if participle else "went"
    elif words[0][-1] == "e":
        words[0] += "d"
    else:
        words[0] += "ed"
    return " ".join(words)


def to_3rd_person(words: List[str]):
    """ Convert a list of words where the first word is a verb to 3rd person present tense."""
    if words[0] == "be":
        words[0] = "is"
    elif words[0] == "have":
        words[0] = "has"
    elif words[0][-1] == "s" or words[0][-1] == "x":
        words[0] += "es"
    else:
        words[0] += "s"
    return " ".join(words)