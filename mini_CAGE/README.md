# MiniCAGE: Simplified CAGE 2 CybORG Environment

`mini_CAGE` is a simplified version of the CAGE 2 CybORG environment with a focus on greater execution speed and parallelisable runs. This repository includes a complete **CTDE-MAPPO** implementation for autonomous cyber defense agent training.

## Project Structure

```
mini_CAGE/
├── minimal.py                      # SimplifiedCAGE environment (13 hosts, 3 subnets)
├── single_agent_gym_wrapper.py     # Gym wrapper (MiniCageBlue)
├── entity_observation_wrapper.py   # Transformer encoder + action masking
├── hierarchical_mappo.py           # CTDE-MAPPO algorithm (Actor + Critic)
├── hierarchical_agents.py          # Manager-Worker hierarchical agents
├── train_hierarchical_mappo.py     # Training script (single-agent MAPPO)
├── baseline_agents.py              # RandomAgent + HeuristicAgent
├── evaluate_baselines.py           # Baseline evaluation script
├── SB3_blue_training.py            # Stable Baselines3 PPO baseline
├── multi_agent/                    # Multi-Agent RL (MARL) implementation
│   ├── config.py                   # Agent assignment, dimensions, hyperparameters
│   ├── env.py                      # Multi-agent environment wrapper
│   ├── models.py                   # Actor, Critic, MessageEncoder networks
│   ├── buffer.py                   # Multi-agent buffer + GAE computation
│   ├── trainer.py                  # Multi-agent MAPPO trainer
│   └── gym_wrapper.py              # Gym interface wrapper
├── train_multi_agent_mappo.py      # Multi-agent training script
└── docs/
    ├── PROJECT_STATUS.md           # Project status and roadmap
    ├── MARL_IMPLEMENTATION.md      # MARL implementation details
    └── 毕设题目研究内容与意义说明.md  # Thesis proposal (Chinese)
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
from single_agent_gym_wrapper import MiniCageBlue

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
from baseline_agents import RandomAgent, HeuristicAgent

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
python evaluate_baselines.py --num_episodes 100 --red_policy bline

# Output:
# Agent           |  Mean Reward |        Std
# Random          |     -1702.63 |     593.65
# Heuristic       |     -1053.08 |     891.31
```

### Train Hierarchical MAPPO

```bash
# Train with default settings
python train_hierarchical_mappo.py

# Training converges in ~72 epochs
# Reward: -126 → -72
```

### Train Multi-Agent MAPPO

```bash
# Train 5 collaborative agents with communication
python train_multi_agent_mappo.py

# Custom parameters
python train_multi_agent_mappo.py \
    --total_timesteps 500000 \
    --n_envs 16 \
    --red_policy bline \
    --learning_rate 1e-4

# Monitor training
tensorboard --logdir multi_agent_tensorboard/
```

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

## Baseline Performance (100 episodes, B-line)

| Agent | Mean Reward | Std | Notes |
|-------|-------------|-----|-------|
| RandomAgent | -1702.63 | 593.65 | Random action selection |
| HeuristicAgent | -1053.08 | 891.31 | Decoy-prioritized rules |
| PPO-MLP | ~-80 | - | Stable Baselines3 |
| Hierarchical-MAPPO | ~-72 | - | Single-agent hierarchical |
| **Multi-Agent MAPPO** | **~-25** | - | **5 agents with communication** |

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

### Hierarchical RL
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

- [MARL Implementation](docs/MARL_IMPLEMENTATION.md) - Multi-agent RL architecture and training details
- [Project Status](docs/PROJECT_STATUS.md) - Development roadmap and current status

## Speed Comparison

| Episodes | CAGE 2 (s) | MiniCAGE (s) | Speedup |
|----------|------------|--------------|---------|
| 1 | 1.16 | 0.12 | ~15x |
| 10 | 7.52 | 0.12 | ~65x |
| 100 | 113.62 | 0.13 | ~950x |
| 1000 | 998.87 | 1.35 | ~800x |

## License

MIT License - See [LICENSE](../LICENSE)
