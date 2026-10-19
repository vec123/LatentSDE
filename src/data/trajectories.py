"""
Loading trajectories from CSV files with columns trajectory_id, time_step and one column per state.

load_trajectories: port of the old JAX src/loader/loader.py. load_observations: the data part of the old JAX
client/assembly.py (build_data), without the observation matrix.
"""
from typing import Optional, Sequence

import pandas as pd
import torch
from torch import Tensor


def load_trajectories(filepath: str, state_columns: Sequence[str] = ("x1", "x2"), time_column: Optional[str] = None):
    """
    (n_trajectories, n_steps, len(state_columns)) states, sorted by trajectory and time step. With time_column,
    (t, states), with t (n_steps,) the times of the first trajectory: all trajectories must share them.
    """
    df = pd.read_csv(filepath).sort_values(by=["trajectory_id", "time_step"])
    n_trajectories = df["trajectory_id"].nunique()
    n_steps = df["time_step"].nunique()
    dtype = torch.get_default_dtype()
    values = df[list(state_columns)].to_numpy().reshape(n_trajectories, n_steps, len(state_columns))
    states = torch.as_tensor(values, dtype=dtype)
    if time_column is None:
        return states
    return torch.as_tensor(df[time_column].to_numpy()[:n_steps], dtype=dtype), states


def load_observations(filepath: str, trajectory: int = 0, state_columns: Sequence[str] = ("x1", "x2"),
                      time_column: Optional[str] = None, dt: Optional[float] = None,
                      truth_columns: Optional[Sequence[str]] = None, obs_std_columns: Optional[Sequence[str]] = None,
                      every: int = 1) -> dict:
    """
    One trajectory of the CSV, subsampled every `every` steps.

    Times come from time_column, or are time_step * dt. obs_std_columns name the per-state noise standard
    deviations stored with the data (first row of the trajectory).

    Returns:
        dict: t (n,), y (n, d) observations, t_true (N,) and x_true (N, d) on the full grid (truth_columns, or
        the observations), obs_std (d,) or None
    """
    df = pd.read_csv(filepath)
    df = df.loc[df["trajectory_id"] == trajectory].sort_values("time_step")
    dtype = torch.get_default_dtype()
    if time_column is not None:
        t_true = torch.as_tensor(df[time_column].to_numpy(), dtype=dtype)
    elif dt is not None:
        t_true = torch.as_tensor(df["time_step"].to_numpy(), dtype=dtype) * dt
    else:
        raise ValueError("load_observations needs time_column or dt")
    y_full = torch.as_tensor(df[list(state_columns)].to_numpy(), dtype=dtype)
    x_true = torch.as_tensor(df[list(truth_columns)].to_numpy(), dtype=dtype) if truth_columns else y_full
    obs_std = torch.as_tensor(df[list(obs_std_columns)].iloc[0].to_numpy(), dtype=dtype) if obs_std_columns else None
    idx = torch.arange(0, len(t_true), every)
    return {"t": t_true[idx], "y": y_full[idx], "t_true": t_true, "x_true": x_true, "obs_std": obs_std}
