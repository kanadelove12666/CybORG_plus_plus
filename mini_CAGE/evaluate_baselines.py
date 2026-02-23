import numpy as np
import argparse
from single_agent_gym_wrapper import MiniCageBlue
from baseline_agents import RandomAgent, HeuristicAgent

def evaluate_agent(agent, env, num_episodes=100):
    """评估单个智能体"""
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
    }

def run_all_evaluations(red_policy='bline', num_episodes=100):
    """评估所有基线并生成对比表格"""
    agents = {
        'Random': RandomAgent(),
        'Heuristic': HeuristicAgent(),
    }

    print(f"Evaluating on {red_policy} red policy, {num_episodes} episodes")
    print("=" * 70)
    print(f"{'Agent':<15} | {'Mean Reward':>12} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 70)

    results_table = {}
    for name, agent in agents.items():
        env = MiniCageBlue(red_policy=red_policy, max_steps=100)
        results = evaluate_agent(agent, env, num_episodes)
        results_table[name] = results
        print(f"{name:<15} | {results['mean_reward']:>12.2f} | {results['std_reward']:>10.2f} | {results['min_reward']:>10.2f} | {results['max_reward']:>10.2f}")

    print("=" * 70)
    return results_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--red_policy', type=str, default='bline')
    args = parser.parse_args()

    run_all_evaluations(red_policy=args.red_policy, num_episodes=args.num_episodes)

if __name__ == '__main__':
    main()
