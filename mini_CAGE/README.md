# MiniCAGE: Simplified CAGE 2 CybORG Environment

`mini_CAGE` is a simplified CybORG environment focused on fast, reproducible training runs.  
This repository's thesis mainline is:
- **CTDE-MAPPO + Transformer entity encoding + action masking + stability optimization** (primary deliverable)
- Hierarchical RL (Manager-Worker) as **optional extension**, not a mainline acceptance requirement.

> Scope note: Results are reported in MiniCAGE and are **not** claimed as strict official CAGE4 evaluation alignment.

## Project Structure

```
mini_CAGE/
├── core/                           # Shared environment and observation infrastructure
│   ├── minimal.py                  # SimplifiedCAGE environment (13 hosts, 3 subnets)
│   ├── single_agent_gym_wrapper.py # Gym wrapper (MiniCageBlue)
│   ├── entity_observation_wrapper.py
│   ├── red_bline_agent.py
│   └── test_agent.py
├── baseline/                       # Baselines and non-mainline extensions
│   ├── baseline_agents.py          # RandomAgent + HeuristicAgent
│   ├── evaluate_baselines.py
│   ├── SB3_blue_training.py
│   ├── hierarchical_agents.py
│   ├── hierarchical_mappo.py
│   ├── train_hierarchical_mappo.py
│   └── HIERARCHICAL_MAPPO_README.md
├── multi_agent/                    # Mainline multi-agent MAPPO implementation
│   ├── config.py
│   ├── env.py
│   ├── models.py
│   ├── buffer.py
│   ├── trainer.py
│   └── gym_wrapper.py
├── train_multi_agent_mappo.py      # Mainline training entry
├── run_ablation_experiments.py
└── docs/
    ├── MARL_IMPLEMENTATION.md
    ├── MAPPO_STABILITY_FIX.md
    ├── HIERARCHICAL_MAPPO_FIXES.md
    ├── OPTIMIZATION_SUMMARY.md
    └── readme.md                   # Thesis mainline method writeup
```

## Installation

```bash
# Create conda environment
mamba create -n cyborg python=3.10
mamba activate cyborg

# Install dependencies
pip install numpy gymnasium torch stable-baselines3
```

## Quick Start

### Basic Environment Usage

```python
from mini_CAGE.core.single_agent_gym_wrapper import MiniCageBlue

# Create environment with B-line attacker
env = MiniCageBlue(red_policy="bline", max_steps=100)
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Baseline Agents

```python
from mini_CAGE.baseline.baseline_agents import RandomAgent, HeuristicAgent

# Random agent (performance lower bound)
random_agent = RandomAgent(num_actions=53)

# Heuristic agent (rule-based, prioritizes decoy deployment)
heuristic_agent = HeuristicAgent()

# Get action
action = agent.get_action(observation)
```

### Evaluate Baselines

```bash
# Run evaluation
python mini_CAGE/baseline/evaluate_baselines.py --num_episodes 100 --red_policy bline

# Output:
# Agent           |  Mean Reward |        Std
# Random          |     -1702.63 |     593.65
# Heuristic       |     -1053.08 |     891.31
```

### Train Hierarchical MAPPO (Baseline)

```bash
# Train with default settings
python mini_CAGE/baseline/train_hierarchical_mappo.py \
  --total-timesteps 1000000 \
  --red-policy bline \
  --learning-rate 3e-4 \
  --n-rollout-steps 2048 \
  --batch-size 256 \
  --n-epochs 10 \
  --entropy-coef 0.05 \
  --target-kl 0.02

# Typical stable range:
# Reward: around -75 (best observed about -62 in historical run)
```

### Train Multi-Agent MAPPO (Mainline, Recommended)

```bash
# Train 5 collaborative agents with communication
python mini_CAGE/train_multi_agent_mappo.py \
  --total_timesteps 1000000 \
  --red_policy bline \
  --learning_rate 3e-4 \
  --n_envs 8 \
  --n_steps 128 \
  --batch_size 256 \
  --n_epochs 10 \
  --entropy_coef 0.01 \
  --min_entropy_coef 0.001 \
  --target_kl 0.012 \
  --non_executed_weight 0.0

# Typical stable range:
# Reward: around -20 to -30 (best observed around -20)
```

Monitor training:
```bash
tensorboard --logdir multi_agent_tensorboard/
```

### Run Ablations (Independent Script)

This repo now provides a standalone launcher:

```bash
# Plan only (print commands, do not run)
python mini_CAGE/run_ablation_experiments.py --algo both --seeds 0,1,2

# Execute all commands sequentially
python mini_CAGE/run_ablation_experiments.py --algo both --seeds 0,1,2 --run
```

Ablations included by default:
- Multi-agent: `baseline`, `no_transformer`, `no_action_mask`, `no_stability`
- Hierarchical: `baseline`, `no_obs_norm`, `no_reward_norm`, `no_stability`

## Environment Details

### Observation Space

- **Shape:** `(78,)` or `(13, 6)` entity format
- **Per-host features (6-dim):**
  - `scan_detected`: Scan activity count
  - `exploit_detected`: Exploit activity count
  - `is_privileged`: Privileged access count
  - `is_removed`: Removal count
  - `scan_count`: Historical scan info
  - `num_decoys`: Number of decoys placed

### Action Space

- **Type:** `Discrete(53)`
- **Mapping:**
  - `0`: sleep
  - `1-13`: analyse(host_i)
  - `14-26`: decoy(host_i)
  - `27-39`: remove(host_i)
  - `40-52`: restore(host_i)

### Network Topology

- **13 Hosts:** def, ent0-2, ophost0-2, opserv, user0-4
- **3 Subnets:** Enterprise, Operational, User
- **Red Policies:** B-line (fixed attack chain), Meander (random walk)

## Baseline Performance (MiniCAGE, B-line)

| Agent | Mean Reward | Std | Notes |
|-------|-------------|-----|-------|
| RandomAgent | -1702.63 | 593.65 | Random action selection |
| HeuristicAgent | -1053.08 | 891.31 | Decoy-prioritized rules |
| PPO-MLP | ~-94 (last) | - | Stable Baselines3 |
| Hierarchical-MAPPO | ~-72 (last) | - | Single-agent hierarchical |
| **Multi-Agent MAPPO** | **~-28 (last)** | - | **Mainline method** |

## Key Features

### CTDE-MAPPO Implementation
- Centralized Training with Decentralized Execution
- Critic uses global state, Actor uses local observation
- GAE advantage estimation (γ=0.99, λ=0.95)

### Multi-Agent Collaboration (MARL)
- **5 Independent Agents**: Each controls a subnet of hosts
- **8-bit Communication**: Agents broadcast messages to coordinate
- **CTDE Paradigm**: Shared Critic enables team learning
- **Host Assignment**: Agent-based network segmentation

| Agent | Hosts | Subnet |
|-------|-------|--------|
| 0 | ent0, ent1 | Enterprise A |
| 1 | def, ent2 | Enterprise Core |
| 2 | ophost0-2, opserv | Operational |
| 3 | user0-2 | User A |
| 4 | user3-4 | User B |

### Hierarchical RL Baseline
- Manager: Selects sub-goals (Investigate/Isolate/Restore/Decoy)
- Worker: Executes atomic actions
- Temporal abstraction reduces effective episode length

### Transformer Encoder
- Entity-based observation processing
- Multi-head self-attention for host relationships
- Position encoding for network topology

### Training Stability
- Observation normalization (RunningMeanStd)
- Reward scaling
- LayerNorm (instead of BatchNorm)
- Gradient clipping (max_norm=0.5)
- Learning rate annealing

## References

1. **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
2. **CTDE**: Foerster et al., "Counterfactual Multi-Agent Policy Gradients", AAAI 2018
3. **CybORG++**: Emerson et al., "CybORG++: An Enhanced Gym for the Development of Autonomous Cyber Agents", 2024
4. **CAGE 2**: [github.com/cage-challenge/CybORG](https://github.com/cage-challenge/CybORG/tree/cage-challenge-2)

## Documentation

- [Method Writeup](docs/readme.md) - Thesis mainline method and formalization
- [MARL Implementation](docs/MARL_IMPLEMENTATION.md) - Multi-agent RL architecture and training details
- [Baseline Overview](baseline/README.md) - Baseline directory, scope, and entrypoints
- [Hierarchical Baseline Notes](baseline/HIERARCHICAL_MAPPO_README.md) - Hierarchical baseline details

## Speed Comparison

| Episodes | CAGE 2 (s) | MiniCAGE (s) | Speedup |
|----------|------------|--------------|---------|
| 1 | 1.16 | 0.12 | ~15x |
| 10 | 7.52 | 0.12 | ~65x |
| 100 | 113.62 | 0.13 | ~950x |
| 1000 | 998.87 | 1.35 | ~800x |

## License

MIT License - See [LICENSE](../LICENSE)
