import numpy as np
import pandas as pd
from qutip import (
    basis, sigmax, sigmay, sigmaz,
    mesolve, fidelity
)
from typing import Literal, Optional


def compute_acceptance(
    tlist: np.ndarray,
    fidelities: np.ndarray,
    threshold: float,
    accept_mode: Literal["final", "window_max", "window_mean"] = "final",
    accept_window: float = 1.0,
) -> tuple[bool, float, float, float]:
    """
    Compute acceptance based on fidelity trajectory.
    
    Args:
        tlist: Time array
        fidelities: Fidelity values at each time
        threshold: Acceptance threshold
        accept_mode: 
            - "final": accept if fidelity[-1] >= threshold
            - "window_max": accept if max(fidelity in [t_final - window, t_final]) >= threshold
            - "window_mean": accept if mean(fidelity in [t_final - window, t_final]) >= threshold
        accept_window: Time window size Δ for windowed modes
    
    Returns:
        (accepted, window_metric, window_start, window_end)
    """
    t_final = tlist[-1]
    
    if accept_mode == "final":
        # Classic acceptance: final fidelity only
        window_start = t_final
        window_end = t_final
        window_metric = float(fidelities[-1])
        accepted = window_metric >= threshold
    else:
        # Windowed acceptance
        window_start = max(0.0, t_final - accept_window)
        window_end = t_final
        
        # Find indices in window [window_start, window_end]
        window_mask = (tlist >= window_start) & (tlist <= window_end)
        window_fidelities = fidelities[window_mask]
        
        if len(window_fidelities) == 0:
            # Edge case: no points in window
            window_metric = float(fidelities[-1])
        elif accept_mode == "window_max":
            window_metric = float(np.max(window_fidelities))
        elif accept_mode == "window_mean":
            window_metric = float(np.mean(window_fidelities))
        else:
            raise ValueError(f"Unknown accept_mode: {accept_mode}")
        
        accepted = window_metric >= threshold
    
    return accepted, window_metric, window_start, window_end


def run_simulation(
    drive_amp: float = 1.0,
    drive_freq: float = 1.0,
    gamma_phi: float = 0.02,
    threshold: float = 0.95,
    t_max: float = 10.0,
    n_steps: int = 400,
    accept_mode: Literal["final", "window_max", "window_mean"] = "final",
    accept_window: float = 1.0,
    seed: Optional[int] = None,
):
    """
    Single driven-qubit simulation with dephasing and
    post-selection based on fidelity threshold.
    
    Args:
        drive_amp: Drive amplitude
        drive_freq: Drive frequency (currently unused in Hamiltonian, reserved)
        gamma_phi: Dephasing rate
        threshold: Acceptance threshold
        t_max: Maximum simulation time
        n_steps: Number of time steps
        accept_mode: Acceptance mode ("final", "window_max", "window_mean")
        accept_window: Time window for windowed acceptance modes
        seed: Random seed (for future stochastic extensions)
    
    Returns:
        (df, summary): DataFrame with time series, dict with summary statistics
    """
    # Set seed if provided (for reproducibility in future stochastic extensions)
    if seed is not None:
        np.random.seed(seed)

    # Time grid
    tlist = np.linspace(0, t_max, n_steps)

    # Operators
    sx, sy, sz = sigmax(), sigmay(), sigmaz()

    # Hamiltonian: driven qubit
    H0 = 0.5 * sz
    Hdrive = drive_amp * sx
    H = H0 + Hdrive

    # Noise: pure dephasing
    c_ops = [np.sqrt(gamma_phi) * sz]

    # Initial and target states
    psi0 = basis(2, 0)
    target = basis(2, 1)

    # Evolve
    result = mesolve(H, psi0, tlist, c_ops, [])

    # Compute fidelity trajectory
    fidelities = np.array([
        fidelity(state, target) for state in result.states
    ])

    # Compute acceptance using specified mode
    accepted, window_metric, window_start, window_end = compute_acceptance(
        tlist, fidelities, threshold, accept_mode, accept_window
    )

    df = pd.DataFrame({
        "t": tlist,
        "fidelity": fidelities,
        "accepted": accepted
    })

    summary = {
        "accepted": bool(accepted),
        "fidelity_final": float(fidelities[-1]),
        "fidelity_max": float(fidelities.max()),
        "window_metric": window_metric,
        "threshold": threshold,
        "gamma_phi": gamma_phi,
        "drive_amp": drive_amp,
        "drive_freq": drive_freq,
        "t_final": float(t_max),
        "accept_mode": accept_mode,
        "accept_window": accept_window,
        "window_start": window_start,
        "window_end": window_end,
    }

    return df, summary
