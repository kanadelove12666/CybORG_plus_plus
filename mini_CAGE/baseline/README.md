# Baselines

`mini_CAGE/baseline/` 存放非主线方法、对照实验脚本和历史扩展实现。

当前主线方法不在这里。
主线仍然是 `mini_CAGE/train_multi_agent_mappo.py` 和 `mini_CAGE/multi_agent/` 中的 CTDE-MAPPO 多智能体方法。

## Directory Layout

```text
baseline/
├── baseline_agents.py          # Random / Heuristic rule-based agents
├── evaluate_baselines.py       # Rule-based baseline evaluation entry
├── SB3_blue_training.py        # Stable-Baselines3 PPO single-agent baseline
├── hierarchical_agents.py      # Hierarchical policy components
├── hierarchical_mappo.py       # Early hierarchical MAPPO implementation
├── train_hierarchical_mappo.py # Hierarchical baseline training entry
├── HIERARCHICAL_MAPPO_README.md
└── README.md
```

## What Belongs Here

- 简单对照组：随机、启发式、SB3 PPO
- 非主线算法：当前是 hierarchical MAPPO
- 为论文或实验表格服务、但不属于最终主方法的实现

## What Does Not Belong Here

- 主线多智能体方法
- 通用环境逻辑
- 共享观测封装或红方脚本

这些内容分别放在：

- `mini_CAGE/core/`
- `mini_CAGE/multi_agent/`
- `mini_CAGE/train_multi_agent_mappo.py`

## Quick Start

在仓库根目录执行，并先激活环境：

```bash
eval "$(conda shell.zsh hook)"
mamba activate cyborg
```

规则基线评估：

```bash
python mini_CAGE/baseline/evaluate_baselines.py \
  --num_episodes 10 \
  --red_policy bline
```

SB3 PPO baseline：

```bash
python mini_CAGE/baseline/SB3_blue_training.py
```

Hierarchical baseline：

```bash
python mini_CAGE/baseline/train_hierarchical_mappo.py \
  --total-timesteps 1000000 \
  --red-policy bline
```

## Notes

- `HIERARCHICAL_MAPPO_README.md` 只覆盖分层 baseline，不代表整个 `baseline/` 目录。
- 后续新增 baseline 时，优先保持“一个 baseline 一个独立入口脚本 + 在本目录登记说明”的模式。
- 如果将来 baseline 数量继续增长，建议进一步拆成：
  - `baseline/rule_based/`
  - `baseline/single_agent/`
  - `baseline/hierarchical/`
  - `baseline/shared/`
