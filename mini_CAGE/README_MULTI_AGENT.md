# Multi-Agent CybORG Framework

基于 CybORG++ MiniCAGE 的多智能体强化学习框架，实现毕业设计草案中的分层多智能体架构。

## 核心功能

### 1. 多智能体环境 (`multi_agent_gym_wrapper.py`)

- **MultiAgentCage**: 多智能体网络防御环境
  - 3个蓝方智能体（每个子网一个）
  - 局部观测 + 全局状态
  - 智能体间通信（8-bit消息）
  - 支持 IPPO（独立PPO）和 CTDE（集中训练分散执行）模式

- **子网划分**:
  - Subnet 0 (Enterprise): ent0, ent1, ent2 (3 hosts)
  - Subnet 1 (Operational): ophost0-2, opserv (4 hosts)
  - Subnet 2 (User): user0-4 (5 hosts)

### 2. 分层智能体 (`hierarchical_agents.py`)

- **ManagerNetwork**: 高层策略网络
  - 每 N 步选择子目标
  - 子目标空间: [investigate_subnet, isolate_threats, restore_services, deploy_decoys, maintain_posture]

- **WorkerNetwork**: 低层策略网络
  - 每个时间步执行原子动作
  - 输入: 局部观测 + 子目标嵌入

- **HierarchicalAgent**: 完整分层智能体
  - 支持观测填充/裁剪以适应不同子网维度

### 3. MAPPO 训练 (`mappo_training.py`)

- **MAPPOTrainer**:
  - 支持 IPPO 和 CTDE 两种模式
  - GAE (Generalized Advantage Estimation)
  - 共享critic（CTDE模式）
  - 多智能体rollout收集

- **训练示例**:
```bash
# CTDE模式训练
python mappo_training.py --mode ctde --timesteps 1000000 --red bline

# IPPO模式训练
python mappo_training.py --mode ippo --timesteps 1000000 --red meander
```

### 4. 评估与可视化 (`multi_agent_evaluation.py`)

- **评估指标**:
  - 累积奖励 (Cumulative Reward)
  - 胜率 (Win Rate)
  - 服务可用性 (Service Availability)
  - 平均修复时间 (MTTR)

- **可视化**:
  - 学习曲线
  - 动作分布
  - 胜率变化

## 快速开始

### 安装依赖

```bash
pip install numpy torch gymnasium matplotlib seaborn
# 如果使用SB3基线对比
pip install stable-baselines3
```

### 测试环境

```bash
python test_multi_agent.py
```

### 训练多智能体

```python
from mappo_training import train_mappo

# CTDE模式
train_mappo(
    mode="ctde",
    total_timesteps=1_000_000,
    red_policy="bline",
    save_dir="./mappo_models",
)
```

### 评估模型

```bash
python multi_agent_evaluation.py \
    --model-path ./mappo_models/ctde_final.pt \
    --mode ctde \
    --red bline \
    --episodes 100
```

### 对比单智能体 vs 多智能体

```python
from multi_agent_evaluation import evaluate_single_vs_multi

results = evaluate_single_vs_multi(
    single_agent_path="./ppo_models/single_agent.zip",
    multi_agent_path="./mappo_models/ctde_final.pt",
    num_episodes=50,
)
```

## 架构对比

| 特性 | Single Agent (SB3) | IPPO | MAPPO (CTDE) | Hierarchical |
|-----|-------------------|------|--------------|--------------|
| 智能体数量 | 1 | 3 | 3 | 3 |
| 观测 | 全局 (78-dim) | 局部 (46-54 dim) | 局部 + 全局 | 局部 + 子目标 |
| Critic | 独立 | 独立 | 共享 | 双层 |
| 通信 | 无 | 可选 | 可选 | 子目标 |
| 训练难度 | 低 | 中 | 中 | 高 |
| 协作能力 | 无 | 有限 | 强 | 强 |

## 文件结构

```
mini_CAGE/
├── multi_agent_gym_wrapper.py   # 多智能体环境包装器
├── hierarchical_agents.py        # 分层智能体架构
├── mappo_training.py             # MAPPO训练框架
├── multi_agent_evaluation.py     # 评估与可视化
├── test_multi_agent.py           # 测试脚本
├── single_agent_gym_wrapper.py   # 单智能体包装器（原始）
├── SB3_blue_training.py          # SB3训练脚本（原始）
└── README_MULTI_AGENT.md         # 本文档
```

## 实验建议

### 1. 基线对比

- 单智能体 PPO (SB3)
- IPPO (独立多智能体)
- MAPPO (CTDE)
- 分层 MAPPO

### 2. 消融实验

- 有无通信
- 有无共享critic
- 不同子目标频率
- 不同网络规模

### 3. 对抗测试

- B-line Agent (快速攻击)
- Meander Agent (随机漫步)
- 混合策略

## 参考文献

1. Singh et al. (2024) - Hierarchical Multi-agent RL for Cyber Network Defense
2. Yu et al. (2021) - The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
3. CybORG++: An Enhanced Gym for Autonomous Cyber Agents

## 开发计划

- [x] 多智能体环境
- [x] 分层架构
- [x] MAPPO训练
- [x] 评估脚本
- [ ] 图神经网络 (GNN) 状态表示
- [ ] Transformer 策略网络
- [ ] 学习通信协议
- [ ] 对抗训练
