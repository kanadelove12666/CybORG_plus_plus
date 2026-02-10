"""
Evaluate SB3-trained single-agent model.
Usage:
    python eval_sb3_model.py --model ./ppo_models/SB3_PPO_1000000/ppo_mini_cage_bline_xxx.zip --episodes 100
"""
import sys
sys.path.insert(0, 'mini_CAGE')

import numpy as np
from stable_baselines3 import PPO
from single_agent_gym_wrapper import MiniCageBlue
import argparse


def evaluate_sb3(model_path: str, num_episodes: int = 100, red_policy: str = "bline"):
    """Evaluate SB3-trained model."""
    print(f"Evaluating SB3 model: {model_path}")
    print(f"vs {red_policy} for {num_episodes} episodes...")

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = MiniCageBlue(red_policy=red_policy, max_steps=100, remove_bugs=True)

    episode_rewards = []
    episode_lengths = []
    win_rates = []
    service_availabilities = []
    red_success_episodes = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        steps = 0
        service_up = 0
        red_succeeded = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            # Track service availability from info
            if info.get("red_success", 0) == 0:  # Red didn't succeed this step
                service_up += 1

            # Track if red achieved impact this episode
            if info.get("red_success", 0) == 1:
                red_succeeded = True

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

        # Win if reward > -150 (better than React-Restore baseline)
        is_win = ep_reward > -150
        win_rates.append(1 if is_win else 0)

        # Service availability
        service_availabilities.append(service_up / steps if steps > 0 else 0)

        # Track red success
        red_success_episodes.append(1 if red_succeeded else 0)

        if ep % 10 == 0:
            print(f"  Episode {ep}/{num_episodes}: {ep_reward:.2f}")

    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    mean_win = np.mean(win_rates)
    mean_svc = np.mean(service_availabilities)
    mean_red_succ = np.mean(red_success_episodes)

    print("\n" + "-"*50)
    print("| rollout/                |             |")
    print(f"|    ep_len_mean          | {np.mean(episode_lengths):.1f}        |")
    print(f"|    ep_rew_mean          | {mean_r:.1f}       |")
    print("| eval/                   |             |")
    print(f"|    win_rate             | {mean_win:.2%}      |")
    print(f"|    service_availability | {mean_svc:.2%}      |")
    print(f"|    red_success_rate     | {mean_red_succ:.2%}      |")
    print("| adversary/              |             |")
    print(f"|    red_policy           | {red_policy:<11} |")
    print(f"|    episodes             | {num_episodes:<11} |")
    print("-"*50)

    return {
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "win_rate": float(mean_win),
        "mean_service_availability": float(mean_svc),
        "red_success_rate": float(mean_red_succ),
        "model_path": model_path,
        "red_policy": red_policy,
        "episodes": num_episodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model")
    parser.add_argument("--red", type=str, default="bline", choices=["bline", "meander"])
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    evaluate_sb3(args.model, args.episodes, args.red)
