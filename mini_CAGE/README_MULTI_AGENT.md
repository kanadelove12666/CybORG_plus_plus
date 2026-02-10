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
pip install numpy torch gymnasium matplotlib seaborn stable-baselines3
```

### 1. 测试环境

```bash
# 测试多智能体框架
python test_multi_agent.py

# 测试单智能体环境
python -c "from single_agent_gym_wrapper import MiniCageBlue; env = MiniCageBlue(); print('OK')"
```

### 2. 训练模型

**单智能体 (SB3)**:
```bash
python SB3_blue_training.py
# 权重保存到: ppo_models/SB3_PPO_1000000/
```

**多智能体 (MAPPO)**:
```bash
# IPPO模式
python quick_train_mappo.py --mode ippo --steps 1000000

# CTDE模式
python quick_train_mappo.py --mode ctde --steps 1000000

# 权重保存到: mappo_models/
```

### 3. 评估对比

#### 3.1 评估 SB3 单智能体（训练权重）

```bash
python eval_sb3_model.py \
    --model ./ppo_models/SB3_PPO_1000000/你的模型文件.zip \
    --red bline \
    --episodes 100
```

#### 3.2 评估 MAPPO 多智能体（训练权重）

```bash
python multi_agent_evaluation.py \
    --model-path ./mappo_models/ippo_final.pt \
    --mode ippo \
    --red bline \
    --episodes 100
```

#### 3.3 原版规则基线（无训练权重）

```bash
# 在原版环境中运行
cd Debugged_CybORG/CybORG
python CybORG/Evaluation/evaluation.py

# 快速测试（非交互式）
python -c "
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRestoreAgent
import inspect

path = str(inspect.getfile(CybORG))[:-10] + '/Shared/Scenarios/Scenario2.yaml'
cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
agent = BlueReactRestoreAgent()

obs = cyborg.reset().observation
total = 0
for i in range(100):
    action = agent.get_action(obs, cyborg.get_action_space('Blue'))
    result = cyborg.step('Blue', action)
    total += result.reward
print(f'Reward: {total:.2f}')
"
```

### 4. 评估输出格式

所有评估脚本输出统一格式的结果：

```
--------------------------------------------------
| rollout/                |             |
|    ep_len_mean          | 100.0       |
|    ep_rew_mean          | -102.4      |
| eval/                   |             |
|    service_availability | 100.00%     |
| adversary/              |             |
|    red_agent            | bline       |
|    episodes             | 100         |
--------------------------------------------------
```

**指标说明**:
- `ep_rew_mean`: 平均回合奖励（越高越好，CybORG为负值）
- `service_availability`: OpServer可用时间比例
- 基线参考: React-Restore (-156), React-Decoy (-69), Sleep (-1141)

## 方法对比

### 训练 vs 规则

| 类型 | 实现 | 权重 | 评估方式 | 典型奖励 |
|-----|------|-----|---------|---------|
| **原版规则** | 手工代码 | 无 | `evaluation.py` 或命令行 | -156 (React-Restore) |
| **SB3单智能体** | 神经网络 | `.zip` | `eval_sb3_model.py` | ~-100 |
| **MAPPO多智能体** | 神经网络 | `.pt` | `multi_agent_evaluation.py` | ~-100 |

### 架构对比

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
├── Core Implementation
│   ├── minimal.py                  # MiniCAGE核心环境
│   ├── single_agent_gym_wrapper.py # 单智能体Gym包装
│   ├── multi_agent_gym_wrapper.py  # 多智能体Gym包装
│   ├── test_agent.py               # 红/蓝Agent基类
│   └── red_bline_agent.py          # B-line红方实现
│
├── Training
│   ├── SB3_blue_training.py        # 单智能体训练(SB3)
│   ├── mappo_training.py           # 多智能体训练(MAPPO)
│   ├── quick_train_mappo.py        # 快速训练脚本
│   └── hierarchical_agents.py      # 分层智能体架构
│
├── Evaluation
│   ├── eval_sb3_model.py           # 评估SB3训练权重
│   ├── multi_agent_evaluation.py   # 评估MAPPO训练权重
│   └── test_multi_agent.py         # 框架测试
│
└── Documentation
    ├── README_MULTI_AGENT.md       # 本文档
    ├── IMPLEMENTATION_DETAILS.md   # 实现细节
    └── TRAINING_ANALYSIS.md        # 训练分析
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
