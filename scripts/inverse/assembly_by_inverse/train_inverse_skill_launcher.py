from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
app = app_launcher.app

from train_inverse_skill import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--reward_weight", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
    app.close()