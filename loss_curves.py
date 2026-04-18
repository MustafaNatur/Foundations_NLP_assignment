"""Parse training log files and save training/validation loss plots."""

import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_FILES = ["baseline.log", "dropout0.log"]

def parse_log_file(log_path):
    """Return steps, train losses, and validation losses for one run."""
    steps = []
    train_losses = []
    val_losses = []

    pattern = re.compile(r"step\s+(\d+):\s+train loss\s+(\d+\.\d+),\s+val loss\s+(\d+\.\d+)")

    with open(log_path, "r") as fileref:
        for line in fileref:
            match = pattern.search(line)
            if match:
                step = match.group(1)
                train_loss = match.group(2)
                val_loss = match.group(3)
                steps.append(int(step))
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))

    return steps, train_losses, val_losses


def plot_training_loss(all_runs):
    """Create and save the training-loss figure for all runs."""
    plt.figure(figsize=(10, 6))
    for experiment_name, experiment_value in all_runs.items():
        plt.plot(experiment_value["steps"], experiment_value["train"], label=experiment_name)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Step")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.close()


def plot_validation_loss(all_runs):
    """Create and save the validation-loss figure for all runs."""
    plt.figure(figsize=(10,6))
    for experiment_name, experiment_value in all_runs.items():
        plt.plot(experiment_value["steps"], experiment_value["val"], label=experiment_name)

    plt.xlabel("Step")
    plt.ylabel("Val Loss")
    plt.title("Validation Loss vs. Step")
    plt.legend()
    plt.savefig("validation_loss.png")
    plt.close()


def main():
    """Parse the configured logs and generate two loss-curve plots."""
    all_runs = {}
    
    for log_name in LOG_FILES:
        log_path = Path(log_name)

        if not log_path.exists():
            print(f"Skipping missing file: {log_path}")
            continue

        steps, train_losses, val_losses = parse_log_file(log_path)

        run_name = log_path.stem
        all_runs[run_name] = {
            "steps": steps,
            "train": train_losses,
            "val": val_losses,
        }

    if not all_runs:
        print("No valid log files were found.")
        return

    plot_training_loss(all_runs)
    plot_validation_loss(all_runs)


if __name__ == "__main__":
    main()
