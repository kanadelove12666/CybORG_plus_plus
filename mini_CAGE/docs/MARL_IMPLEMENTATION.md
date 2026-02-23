# CAGE 4 多智能体协作（MARL）实现文档

## 概述

本文档记录了 CAGE 4 真正的多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）的实现过程，包括框架设计、遇到的问题及解决方案。

---

## 1. 框架设计

### 1.1 架构对比

| 维度 | 单智能体 PPO | 多智能体 MAPPO (本实现) |
|------|-------------|------------------------|
| 智能体数量 | 1个蓝队代理 | **5个独立蓝队智能体** |
| 策略网络 | 单一Actor-Critic | 每个agent独立Actor + 共享Critic |
| 通信方式 | 无 | **8-bit消息通道** |
| 奖励信号 | 单独奖励 | 共享团队奖励 |
| 训练范式 | 标准PPO | CTDE (集中训练、分散执行) |

### 1.2 主机分配

13个主机分配给5个智能体，按网络区域划分：

```
Agent 0: ent0, ent1          (企业网 A - 2台)
Agent 1: def, ent2           (企业核心 - 2台)
Agent 2: ophost0-2, opserv   (运维区 - 4台)
Agent 3: user0-2             (用户区 A - 3台)
Agent 4: user3-4             (用户区 B - 2台)
```

### 1.3 文件结构

```
mini_CAGE/multi_agent/
├── __init__.py         # 模块初始化
├── config.py           # 主机分配、维度计算、超参数
├── env.py              # SimplifiedMultiAgentCAGE 环境
├── models.py           # Actor, Critic, MessageEncoder
├── buffer.py           # MultiAgentBuffer + GAE计算
├── trainer.py          # MultiAgentMAPPOTrainer
└── gym_wrapper.py      # Gym接口封装

mini_CAGE/
└── train_multi_agent_mappo.py  # 训练入口脚本
```

### 1.4 核心组件

#### 1.4.1 通信通道 (CommunicationChannel)

```python
class CommunicationChannel:
    """8-bit消息通道，支持智能体间广播通信"""
    def __init__(self, n_agents=5, message_bits=8):
        self.messages = np.zeros((n_agents, message_bits))

    def send_message(self, agent_id, message):
        self.messages[agent_id] = message

    def receive_messages(self, agent_id):
        # 接收所有其他智能体的消息
        return self.messages[np.arange(self.n_agents) != agent_id].flatten()
```

#### 1.4.2 Actor 网络

每个智能体有独立的Actor网络，输入包含：
- 局部观测 (local_obs): 只看到自己负责的主机
- 全局摘要 (global_summary): 压缩的全局状态信息
- 收到的消息 (messages): 其他智能体的通信

```python
obs = concat([local_obs, global_summary, messages])  # 输入
action = Actor(obs)  # 输出: 离散动作
message = MessageEncoder(obs)  # 输出: 8-bit消息
```

#### 1.4.3 共享Critic

Critic使用全局状态进行价值估计（CTDE的核心）：

```python
global_state = concat([true_state, decoy_info, impact_info])  # 65维
value = CentralizedCritic(global_state, all_messages)
```

---

## 2. 实现中遇到的问题

### 2.1 Episode Reset Bug

**现象**: `ep_len_mean = 1.0`，每个step都被当作新的episode

**原因**:
- 使用全局步数计数器 `self.current_step`
- 当某个环境done时，没有重置该环境的步数
- 导致done状态持续为True

**解决方案**:
```python
# Before (错误)
self.current_step += 1
done = (self.current_step >= self.max_steps)

# After (正确)
self.episode_steps = np.zeros(num_envs, dtype=np.int32)  # 每环境独立

def step(self, ...):
    self.episode_steps += 1
    done = (self.episode_steps >= self.max_steps)

    # 部分重置
    if done.any():
        env_indices = np.where(done)[0]
        self.env.reset(env_indices=env_indices)
        self.episode_steps[env_indices] = 0
```

### 2.2 Value Loss 不稳定

**现象**: `value_loss` 从 779 跳到 3658，训练不稳定

**原因**:
1. 使用MSE loss对异常值敏感
2. GAE bootstrap值计算不正确，使用了错误的状态

**解决方案**:

1. **使用Huber Loss**:
```python
# Before
value_loss = F.mse_loss(values, returns)

# After
value_loss = F.smooth_l1_loss(values.squeeze(-1), batch_returns)
```

2. **修复Bootstrap值传递**:
```python
# Before (错误)
last_values = self.critic(buffer.global_states[-1])  # 使用rollout最后一步的状态

# After (正确)
# 在collect_rollout中追踪最后一步之后的状态
last_global_state = ...  # rollout结束后的状态
# 传递给update()用于正确的bootstrap
update_stats = trainer.update(last_obs, last_global_state, last_messages, progress)
```

### 2.3 环境接口不匹配

**现象**: `ValueError: too many values to unpack (expected 4)`

**原因**: `env.step()` 返回5个值（含truncated），但wrapper只解包4个

**解决方案**:
```python
# Before
agent_obs, reward, done, info = self.env.step(agent_actions)

# After
agent_obs, reward, done, truncated, info = self.env.step(agent_actions)
```

### 2.4 部分重置支持

**现象**: 调用 `env.reset(env_indices=...)` 时报错 `unexpected keyword argument`

**原因**: Gym wrapper没有实现 `env_indices` 参数的透传

**解决方案**:
```python
# gym_wrapper.py
def reset(self, env_indices=None):
    agent_obs, info = self.env.reset(env_indices=env_indices)
    if env_indices is None:
        self.episode_steps.fill(0)
    else:
        self.episode_steps[env_indices] = 0
    return agent_obs, info
```

---

## 3. 训练结果

### 3.1 性能对比

| 指标 | 初始值 | 最终值 (100K步) | 提升 |
|------|--------|-----------------|------|
| ep_rew_mean | -1804 | **-25** | 98.6% ↑ |
| value_loss | ~30 | **0.58** | 稳定 |
| ep_len_mean | 1.0 | **100** | 修复 |
| entropy | 2.38 | 2.21 | 收敛 |

### 3.2 超参数配置

```python
# 与 train_hierarchical_mappo.py 对齐
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_EPOCHS = 10
N_ENVS = 8
N_STEPS = 128
BATCH_SIZE = 256
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# 多智能体特有
N_AGENTS = 5
MESSAGE_BITS = 8
MESSAGE_COEF = 0.1
```

---

## 4. Reward 对齐验证

### 4.1 验证方法

检查reward从底层环境到上层的传递链：

```
SimplifiedCAGE (minimal.py)
    └── step() 返回 reward_dict['Blue']
            ↓
单智能体 (single_agent_gym_wrapper.py:103)
    └── reward = reward_dict['Blue'][0][0]
            ↓
多智能体 (multi_agent/env.py:403)
    └── shared_reward = reward_dict['Blue'].flatten()
```

### 4.2 结论

**Reward计算100%对齐**：
- 底层环境相同 (`SimplifiedCAGE`)
- 红队策略相同 (`B_line_minimal`)
- Reward提取相同 (`reward_dict['Blue']`)

---

## 5. 使用方法

### 5.1 训练

```bash
cd mini_CAGE
mamba activate cyborg

# 默认训练 (1M步)
python train_multi_agent_mappo.py

# 自定义参数
python train_multi_agent_mappo.py \
    --total_timesteps 500000 \
    --n_envs 16 \
    --red_policy bline \
    --learning_rate 1e-4
```

### 5.2 监控

```bash
# TensorBoard
tensorboard --logdir multi_agent_tensorboard/
```

### 5.3 模型位置

```
multi_agent_mappo_models/
├── checkpoint_100352.pt
└── ...
```

---

## 6. 后续工作

- [ ] 与单智能体baseline对比评测
- [ ] 消融实验：消息通道的作用
- [ ] 支持更复杂的红队策略
- [ ] 实现分层MARL (Hierarchical MAPPO)

---

## 参考资料

- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
- [CTDE Paradigm](https://arxiv.org/abs/1911.10635)
- PROJECT_STATUS.md - 项目整体规划

---

*文档创建时间: 2026-02-23*
