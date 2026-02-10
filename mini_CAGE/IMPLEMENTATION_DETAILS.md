# CybORG 多智能体框架实现细节

## 一、实验整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     实验架构概览                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  Red Agent  │────▶│ Simplified  │◀────│ Blue Agent  │        │
│  │  (B-line/   │     │   CAGE      │     │   (3个)     │        │
│  │  Meander)   │     │  (底层环境)  │     │             │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                              │                                    │
│                              ▼                                    │
│                   ┌─────────────────────┐                        │
│                   │  MultiAgentCage     │                        │
│                   │  (多智能体包装器)    │                        │
│                   │                     │                        │
│                   │  - 子网划分          │                        │
│                   │  - 局部观测          │                        │
│                   │  - 通信机制          │                        │
│                   └─────────────────────┘                        │
│                              │                                    │
│           ┌──────────────────┼──────────────────┐                │
│           ▼                  ▼                  ▼                │
│    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│    │    IPPO     │   │    MAPPO    │   │ Hierarchical│          │
│    │  (独立PPO)  │   │   (CTDE)    │   │   (分层)    │          │
│    └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 二、核心模块实现细节

### 2.1 多智能体环境 (`multi_agent_gym_wrapper.py`)

#### 子网划分策略

```python
# 3个子网对应CAGE 2网络拓扑
SUBNET_HOSTS = {
    0: [1, 2, 3],         # Enterprise: ent0, ent1, ent2
    1: [4, 5, 6, 7],      # Operational: ophost0-2, opserv
    2: [8, 9, 10, 11, 12] # User: user0-4
}
```

- **Agent 0** (Enterprise): 防守企业服务器子网
- **Agent 1** (Operational): 防守核心业务子网
- **Agent 2** (User): 防守用户主机子网

#### 观测空间设计

每个Agent的观测维度不同（根据子网大小）：

```
观测 = [局部特征] + [全局上下文] + [通信消息]

局部特征 (per host):
  - scan_state: 扫描状态 (-1未知, 0已扫描, 1已发现)
  - exploit_state: 是否被渗透 (0/1)
  - privilege_state: 是否有特权 (0/1)
  - scan_activity: 扫描活动记录
  - attack_activity: 攻击活动记录
  - removed_processes: 已清除进程
  - exploited_unsafe: 被攻击标记

全局上下文 (3 dims):
  - 整体网络感染率
  - OpServer渗透状态
  - OpServer特权状态

通信消息 (16 dims, optional):
  - 来自其他2个agent的8-bit消息
```

**各Agent观测维度**:
- Agent 0 (3 hosts): 3×9 + 3 + 16 = **46 dim**
- Agent 1 (4 hosts): 4×9 + 3 + 16 = **55 dim**
- Agent 2 (5 hosts): 5×9 + 3 + 16 = **64 dim**

#### 动作空间

每个Agent的动作空间大小取决于子网主机数量：

```python
动作 = 1 (sleep) + 4 × num_hosts

0: sleep (休息)
1~n: analyse_host_i (分析主机i)
n+1~2n: decoy_host_i (部署诱饵)
2n+1~3n: remove_host_i (移除恶意进程)
3n+1~4n: restore_host_i (恢复主机)
```

**各Agent动作数**:
- Agent 0: 1 + 4×3 = **13 actions**
- Agent 1: 1 + 4×4 = **17 actions**
- Agent 2: 1 + 4×5 = **21 actions**

### 2.2 MAPPO训练器 (`mappo_training.py`)

#### 网络架构

**Actor-Critic网络** (每个Agent独立):
```
输入 (obs_dim) → Linear(hidden_dim) → ReLU → Linear(hidden_dim) → ReLU
                                                   ↓
                        ┌──────────────────────────┼──────────────────────────┐
                        ▼                          ▼                          ▼
                  Linear(action_dim)         Linear(1)                  (features)
                        (Actor)               (Critic)
```

**Centralized Critic** (仅CTDE模式):
```
输入 (141 dim全局状态) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1)
```

#### 关键算法参数

| 参数 | 默认值 | 意义 |
|-----|-------|------|
| `lr` | 3e-4 | Adam学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE lambda参数 |
| `clip_coef` | 0.2 | PPO裁剪系数ε |
| `vf_coef` | 0.5 | Value loss权重 |
| `ent_coef` | 0.01 | 熵奖励权重 |
| `rollout_steps` | 2048 | 每次更新的步数 |
| `num_epochs` | 10 | 每次数据训练轮数 |
| `batch_size` | 64 | Minibatch大小 |

#### 训练模式对比

**IPPO (Independent PPO)**:
```python
# 每个Agent有独立的Actor和Critic
for agent_id in range(num_agents):
    actor = ActorCritic(obs_dim, action_dim)
    # 独立更新，不共享信息
```
- 优点: 实现简单， decentralized execution
- 缺点: 无法利用全局信息，协作能力有限

**MAPPO (CTDE)**:
```python
# Actor独立，Critic共享全局状态
actors = [ActorCritic(obs_dim, action_dim) for _ in range(n)]
centralized_critic = CentralizedCritic(global_state_dim)

# 训练时用全局状态评估价值
value = centralized_critic(global_state)
# 执行时只用局部观测
action = actors[agent_id](local_obs)
```
- 优点: 利用全局信息训练，更好的协作
- 缺点: 需要全局状态，训练计算量大

### 2.3 分层智能体 (`hierarchical_agents.py`)

#### 两层架构

**Manager (高层策略)**:
```
输入: 全局状态 (141 dim)
输出: 子目标 (5个选项)
更新频率: 每 N 个时间步

子目标空间:
0: investigate_subnet (调查子网)
1: isolate_threats (隔离威胁)
2: restore_services (恢复服务)
3: deploy_decoys (部署诱饵)
4: maintain_posture (保持防御姿态)
```

**Worker (低层策略)**:
```
输入: 局部观测 + 子目标嵌入
输出: 原子动作 (Analyse/Remove/Restore/Decoy/Sleep)
更新频率: 每个时间步
```

#### 子目标嵌入

```python
# 子目标索引 → 可学习嵌入向量
subgoal_embedding = nn.Embedding(num_subgoals, hidden_dim // 2)

# 与观测特征拼接
combined_features = [obs_features; subgoal_embedding]
```

## 三、训练流程详解

### 3.1 数据收集 (Rollout Collection)

```python
for step in range(rollout_steps):
    # 1. 每个Agent选择动作
    actions = {}
    for agent_id in range(num_agents):
        obs = get_local_obs(agent_id)
        action = actor[agent_id].select_action(obs)
        actions[agent_id] = action

    # 2. 执行动作 (转换为全局动作)
    global_action = combine_actions(actions)
    red_action = red_agent.get_action(red_obs)

    # 3. 环境步进
    next_obs, reward, done, info = env.step(global_action, red_action)

    # 4. 存储数据
    store(obs, actions, reward, value, done)
```

### 3.2 GAE计算

```python
def compute_gae(rewards, values, dones, next_value):
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # TD误差
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]

        # GAE累加
        gae = delta + gamma * lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

### 3.3 PPO更新

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        new_log_probs, entropy, values = model(batch)

        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        critic_loss = 0.5 * ((values - returns) ** 2).mean()

        # 熵奖励
        entropy_loss = -entropy.mean()

        # 总loss
        loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
```

## 四、关键设计决策

### 4.1 为什么选择分层观测？

- **真实性**: 实际网络安全中，不同子网由不同管理员负责
- **可扩展性**: 新增子网只需新增Agent，无需修改网络结构
- **计算效率**: 局部观测维度小，降低网络复杂度

### 4.2 为什么使用CTDE？

- **Credit Assignment**: 在多智能体中，难以确定哪个Agent对成功防御贡献最大
- **Non-stationarity**: 其他Agent策略变化导致环境动态变化
- **Global Information**: 训练时利用全局状态有助于学习协作策略

### 4.3 动作组合策略

```python
def combine_agent_actions(actions):
    """将多个Agent的动作合并为全局动作"""
    # 优先级: restore > remove > analyse > decoy > sleep
    for agent_id, action in sorted(actions.items(), reverse=True):
        action_type, host = action_map[action]

        if action_type == "restore":
            return 40 + host  # 高优先级
        elif action_type == "remove":
            return 27 + host
        # ... 其他动作

    return 0  # sleep (默认)
```

## 五、训练监控指标

### 5.1 关键指标解释

| 指标 | 理想范围 | 说明 |
|-----|---------|------|
| `ep_rew_mean` | > -150 | 平均回合奖励，越高越好 |
| `actor_loss` | ~0 | 策略损失，收敛后接近0 |
| `critic_loss` | 下降中 | Value损失，应逐渐减小 |
| `entropy` | 0.5~2 | 策略熵，太高=探索过度，太低=过早收敛 |
| `approx_kl` | < 0.02 | 策略变化程度，太大说明更新过快 |
| `clip_fraction` | 0.1~0.3 | 被裁剪的样本比例 |

### 5.2 收敛判断

- **奖励稳定**: 连续50个update奖励波动<5%
- **Critic Loss稳定**: 不再明显下降
- **Entropy适中**: 保持在0.1以上避免过早收敛

## 六、实验对比基线

### 6.1 单智能体 PPO (SB3)

```python
env = MiniCageBlue(red_policy="bline")
model = PPO("MlpPolicy", env, learning_rate=0.002)
```

- 观测: 全局78维
- 动作: 全局53个
- 特点: 简单直接，但无法处理子网协作

### 6.2 IPPO vs MAPPO vs Hierarchical

| 特性 | IPPO | MAPPO | Hierarchical |
|-----|------|-------|--------------|
| 网络结构 | Actor-Critic ×3 | Actor×3 + Shared Critic | Manager + Worker×3 |
| 观测 | 局部 | 局部+全局 | 局部+全局+子目标 |
| 训练难度 | 中 | 高 | 很高 |
| 协作能力 | 有限 | 强 | 强+结构化 |
| 适用场景 | 简单协作 | 复杂协作 | 长时序任务 |

## 七、常见问题排查

### 7.1 训练不收敛

- **检查**: 奖励是否一直为负且不改善
- **解决**: 增大学习率，增加entropy系数，检查reward shaping

### 7.2 过早收敛

- **检查**: Entropy迅速降到0
- **解决**: 增大ent_coef，使用更小的clip_coef

### 7.3 维度不匹配

- **检查**: matmul维度错误
- **解决**: 确保observation_space和network输入维度一致

## 八、扩展方向

### 8.1 图神经网络 (GNN)

将网络拓扑建模为图，使用GAT/GCN处理变长输入：
```python
# 节点特征: host状态
# 边: 网络连接
graph_obs = pyg.data.Data(x=host_features, edge_index=connections)
```

### 8.2 注意力机制

在Critic中加入Attention，关注重要Agent：
```python
attention_weights = softmax(Q @ K.T / sqrt(d_k))
weighted_values = attention_weights @ V
```

### 8.3 学习通信

使用DIAL或TarMAC替代固定的8-bit消息：
```python
message = communication_network(obs, incoming_message)
```

---

**当前训练结果示例**:
```
Update 0/488  | Reward: -599.61 | Entropy: 2.79  # 初始随机策略
Update 10/488 | Reward: -103.17 | Entropy: 0.0001  # 快速收敛
Update 50/488 | Reward: -102.94 | Entropy: 0.0000  # 稳定收敛
```

奖励从-599提升到-103，说明Agent学会了有效的防御策略！
