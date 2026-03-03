import pandas as pd
from src.sim import run_simulation
from src.plot import plot_fidelity
from src.utils import make_run_dir, save_summary

def main():
    run_dir, run_id = make_run_dir()

    # --- PARAMETERS (easy to vary later) ---
    params = dict(
        drive_amp=1.0,
        drive_freq=1.0,
        gamma_phi=0.03,
        threshold=0.90,
        t_max=12.0,
        n_steps=500
    )

    df, summary = run_simulation(**params)

    # Save numeric results
    df.to_parquet(run_dir / "results.parquet")

    # Save summary
    summary["run_id"] = run_id
    save_summary(run_dir, summary)

    # Save plots
    plot_fidelity(df, run_dir)

    print(f"Run completed: {run_dir}")
    print(summary)

if __name__ == "__main__":
    main()
