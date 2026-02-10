"""
Quick training script for MAPPO on CybORG

Usage:
    python quick_train_mappo.py --mode ctde --steps 100000
"""

import argparse
from mappo_training import train_mappo


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO on CybORG")
    parser.add_argument("--mode", type=str, default="ctde", choices=["ippo", "ctde"],
                        help="Training mode: ipso (Independent PPO) or ctde (MAPPO)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--red", type=str, default="bline", choices=["bline", "meander"],
                        help="Red agent policy")
    parser.add_argument("--save-dir", type=str, default="./mappo_models",
                        help="Directory to save models")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, or cuda")

    args = parser.parse_args()

    print("="*60)
    print(f"Training MAPPO - Mode: {args.mode}, Steps: {args.steps}")
    print("="*60)

    train_mappo(
        mode=args.mode,
        total_timesteps=args.steps,
        red_policy=args.red,
        save_dir=args.save_dir,
        device=args.device,
    )

    print("\nTraining complete!")
    print(f"Models saved to {args.save_dir}")


if __name__ == "__main__":
    main()
