"""
run_ablation_experiments.py

Unified ablation launcher for:
- train_multi_agent_mappo.py
- baseline/train_hierarchical_mappo.py

Default behavior prints commands (plan mode).
Use --run to execute sequentially.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_seeds(text: str) -> List[int]:
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def build_multi_commands(args, seeds: Iterable[int], group: str) -> List[Tuple[str, List[str]]]:
    """Build command list for multi-agent MAPPO ablations."""
    variants = [
        ("baseline", []),
        ("no_transformer", ["--no_transformer"]),
        ("no_action_mask", ["--no_action_mask"]),
        ("no_stability", ["--no_stability_tricks"]),
    ]

    cmds: List[Tuple[str, List[str]]] = []
    for seed in seeds:
        for variant, variant_flags in variants:
            run_tag = f"{group}_multi_{variant}_seed{seed}"
            cmd = [
                sys.executable,
                "mini_CAGE/train_multi_agent_mappo.py",
                "--total_timesteps", str(args.multi_timesteps),
                "--red_policy", args.red_policy,
                "--n_envs", str(args.multi_n_envs),
                "--n_steps", str(args.multi_n_steps),
                "--batch_size", str(args.multi_batch_size),
                "--n_epochs", str(args.multi_n_epochs),
                "--learning_rate", str(args.multi_lr),
                "--entropy_coef", str(args.multi_entropy_coef),
                "--min_entropy_coef", str(args.multi_min_entropy_coef),
                "--target_kl", str(args.multi_target_kl),
                "--non_executed_weight", str(args.multi_non_executed_weight),
                "--device", args.device,
                "--seed", str(seed),
                "--save_dir", f"multi_agent_mappo_models/{run_tag}",
                "--tensorboard_log", f"multi_agent_tensorboard/{run_tag}",
            ]
            cmd.extend(variant_flags)
            cmds.append((run_tag, cmd))
    return cmds


def build_hier_commands(args, seeds: Iterable[int], group: str) -> List[Tuple[str, List[str]]]:
    """Build command list for hierarchical MAPPO ablations."""
    variants = [
        ("baseline", []),
        ("no_obs_norm", ["--no-obs-norm"]),
        ("no_reward_norm", ["--no-reward-norm"]),
        ("no_stability", ["--no-stability-tricks"]),
    ]

    cmds: List[Tuple[str, List[str]]] = []
    for seed in seeds:
        for variant, variant_flags in variants:
            run_tag = f"{group}_hier_{variant}_seed{seed}"
            cmd = [
                sys.executable,
                "mini_CAGE/baseline/train_hierarchical_mappo.py",
                "--total-timesteps", str(args.hier_timesteps),
                "--red-policy", args.red_policy,
                "--learning-rate", str(args.hier_lr),
                "--n-rollout-steps", str(args.hier_rollout_steps),
                "--batch-size", str(args.hier_batch_size),
                "--n-epochs", str(args.hier_n_epochs),
                "--entropy-coef", str(args.hier_entropy_coef),
                "--target-kl", str(args.hier_target_kl),
                "--seed", str(seed),
                "--device", args.device,
                "--run-name", run_tag,
            ]
            cmd.extend(variant_flags)
            cmds.append((run_tag, cmd))
    return cmds


def run_or_print(commands: List[Tuple[str, List[str]]], execute: bool) -> None:
    for idx, (name, cmd) in enumerate(commands, start=1):
        quoted = " ".join(shlex.quote(part) for part in cmd)
        print(f"[{idx:03d}] {name}\n{quoted}\n")
        if execute:
            subprocess.run(cmd, check=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch ablation runs for MiniCAGE training scripts.")
    parser.add_argument("--algo", choices=["multi", "hier", "both"], default="both")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--red_policy", choices=["bline", "meander"], default="bline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run", action="store_true", help="Execute commands. Without this flag, only print plan.")
    parser.add_argument("--group", type=str, default=None, help="Optional run group prefix.")

    # Multi-agent recommended high-performance baseline params
    parser.add_argument("--multi_timesteps", type=int, default=1_000_000)
    parser.add_argument("--multi_lr", type=float, default=3e-4)
    parser.add_argument("--multi_n_envs", type=int, default=8)
    parser.add_argument("--multi_n_steps", type=int, default=128)
    parser.add_argument("--multi_batch_size", type=int, default=256)
    parser.add_argument("--multi_n_epochs", type=int, default=10)
    parser.add_argument("--multi_entropy_coef", type=float, default=0.01)
    parser.add_argument("--multi_min_entropy_coef", type=float, default=0.001)
    parser.add_argument("--multi_target_kl", type=float, default=0.012)
    parser.add_argument("--multi_non_executed_weight", type=float, default=0.0)

    # Hierarchical recommended high-performance baseline params
    parser.add_argument("--hier_timesteps", type=int, default=1_000_000)
    parser.add_argument("--hier_lr", type=float, default=3e-4)
    parser.add_argument("--hier_rollout_steps", type=int, default=2048)
    parser.add_argument("--hier_batch_size", type=int, default=256)
    parser.add_argument("--hier_n_epochs", type=int, default=10)
    parser.add_argument("--hier_entropy_coef", type=float, default=0.05)
    parser.add_argument("--hier_target_kl", type=float, default=0.02)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    group = args.group or dt.datetime.now().strftime("abl_%Y%m%d_%H%M%S")

    all_commands: List[Tuple[str, List[str]]] = []
    if args.algo in ("multi", "both"):
        all_commands.extend(build_multi_commands(args, seeds, group))
    if args.algo in ("hier", "both"):
        all_commands.extend(build_hier_commands(args, seeds, group))

    mode = "RUN" if args.run else "PLAN"
    print(f"Mode: {mode}")
    print(f"Group: {group}")
    print(f"Total commands: {len(all_commands)}\n")
    run_or_print(all_commands, execute=args.run)


if __name__ == "__main__":
    main()
