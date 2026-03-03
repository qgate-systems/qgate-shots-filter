import matplotlib.pyplot as plt

def plot_fidelity(df, run_dir):
    plt.figure(figsize=(6,4))
    plt.plot(df["t"], df["fidelity"], label="Fidelity")
    if df["accepted"].iloc[0]:
        plt.axhline(
            y=df["fidelity"].iloc[-1],
            linestyle="--",
            alpha=0.3
        )
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.title("Fidelity vs Time (Conditioned Run)")
    plt.grid(True)

    svg = run_dir / "figures" / "fidelity_vs_time.svg"
    png = run_dir / "figures" / "fidelity_vs_time.png"
    plt.savefig(svg)
    plt.savefig(png, dpi=300)
    plt.close()
