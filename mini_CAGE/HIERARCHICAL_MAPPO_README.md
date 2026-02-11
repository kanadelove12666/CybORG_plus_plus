# Hierarchical MAPPO for CybORG++

分层多智能体PPO (Hierarchical MAPPO) 训练框架，用于CybORG网络安全防御环境。

> **重要更新 (2026-02-11)**: 已修复8个关键bug，包括worker value复制、explained_variance计算错误、bootstrap value获取错误等。详见[HIERARCHICAL_MAPPO_FIXES.md](HIERARCHICAL_MAPPO_FIXES.md)。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                    Hierarchical Policy                      │
├─────────────────────────┬───────────────────────────────────┤
│      Manager (高层)      │         Worker (低层)              │
│  - 选择关注的主机        │  - 执行具体的防御动作              │
│  - 13个host选择          │  - 5个动作: sleep/analyse/decoy/   │
│  - 全局观测 (78-dim)     │         remove/restore            │
│  - 学习长期策略          │  - 接收manager的goal embedding    │
└─────────────────────────┴───────────────────────────────────┘
```

## 实现特性

### 核心算法
- [x] **MAPPO (Multi-Agent PPO)**: CTDE (Centralized Training, Decentralized Execution)
- [x] **分层策略**: Manager选择host → Worker选择action
- [x] **GAE (Generalized Advantage Estimation)**: λ=0.95
- [x] **PPO Clip**: ε=0.2

### 现代RL优化 (已集成)
- [x] **分离优化器**: Policy和Value使用独立Adam优化器
- [x] **LayerNorm**: 所有隐藏层使用Layer Normalization
- [x] **正交初始化**: Orthogonal initialization (gain=√2 for hidden, 0.01 for output)
- [x] **观测归一化**: Running Mean/Std normalization (SB3风格)
- [x] **Return归一化**: 使用discounted return的std归一化reward
- [x] **线性学习率衰减**: 从初始值线性降至0
- [x] **KL Early Stopping**: 当KL > 0.015时提前停止

### 超参数 (修复后 - 推荐配置)
```python
learning_rate = 3e-4      # 降低学习率提高稳定性 (原为0.002)
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
n_epochs = 10             # 增加epoch数提高样本效率 (原为6)
batch_size = 256          # 更大批量稳定梯度 (原为64)
entropy_coef = 0.05       # 更高熵奖励促进探索 (原为0.01)
value_coef = 0.5
max_grad_norm = 0.5
target_kl = 0.02          # 稍高KL阈值减少early stopping (原为0.015)
hidden_dim = 128
```

## 文件结构

```
mini_CAGE/
├── train_hierarchical_mappo.py   # 主训练脚本 (唯一需要的文件)
├── single_agent_gym_wrapper.py   # MiniCageBlue环境包装器
├── SB3_blue_training.py          # SB3 PPO基线 (对比用)
└── HIERARCHICAL_MAPPO_README.md  # 本文档
```

## 使用方法

### 基础训练

```bash
cd /Users/huangyinzhen/Documents/GitHub/CybORG_plus_plus
mamba activate cyborg

python mini_CAGE/train_hierarchical_mappo.py \
    --total-timesteps 1000000 \
    --red-policy bline
```

### 完整参数

```bash
python mini_CAGE/train_hierarchical_mappo.py \
    --total-timesteps 1000000 \
    --red-policy bline \
    --learning-rate 0.002 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --n-epochs 6 \
    --batch-size 256 \
    --entropy-coef 0.05 \
    --n-rollout-steps 2048 \
    --use-tensorboard \
    --run-name "my_experiment"
```

### 关键参数说明 (修复后)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--red-policy` | bline | Red agent策略 (bline/meander) |
| `--total-timesteps` | 1000000 | 总训练步数 |
| `--learning-rate` | 3e-4 | 学习率 (修复: 从0.002降低) |
| `--batch-size` | 256 | Mini-batch大小 (修复: 从64增大) |
| `--entropy-coef` | 0.05 | 熵系数 (修复: 从0.01增大) |
| `--n-epochs` | 10 | PPO epoch数 (修复: 从6增大) |
| `--target-kl` | 0.02 | KL阈值 (修复: 从0.015增大) |
| `--n-rollout-steps` | 2048 | 每次更新的步数 |
| `--use-tensorboard` | False | 启用TensorBoard日志 |
| `--no-lr-schedule` | False | 禁用学习率衰减 |

## 监控训练

### TensorBoard

```bash
tensorboard --logdir ./hierarchical_mappo_tensorboard
```

查看指标:
- `rollout/ep_rew_mean`: 平均episode奖励
- `train/value_loss`: Value function损失
- `train/explained_variance`: Critic解释方差
- `train/approx_kl`: KL散度
- `train/clip_fraction`: Clip比例

### 预期训练指标

| 阶段 | Value Loss | Explained Var | Reward | Clip Fraction | 状态 |
|------|------------|---------------|--------|---------------|------|
| 初始 | 20-50 | ~0 | -150 ~ -200 | 0.1-0.2 | 正常 |
| 100 iter | 5-15 | 0.1-0.3 | -120 ~ -150 | 0.1-0.2 | 学习中 |
| 300 iter | 1-5 | 0.3-0.6 | -80 ~ -120 | 0.05-0.15 | 收敛中 |
| 500+ iter | 0.5-2 | **0.5-0.8** | **-50 ~ -80** | 0.02-0.1 | **良好** |

**实际训练效果** (1M steps, B-line red agent):
- `explained_variance`: **0.5-0.8** (Critic正常学习)
- `value_loss`: **0.5-2** (稳定的值函数)
- `reward`: **-72** (优于预期的-80)
- `clip_fraction`: **0.02-0.1** (稳定的策略更新)

## 网络架构

### Manager Network
```
Input (78-dim)
    ↓
MLP [128, 128] + LayerNorm + Tanh
    ↓
┌─────────────────┬─────────────────┐
│ Policy Head     │ Value Head      │
│ (13 hosts)      │ (1 value)       │
└─────────────────┴─────────────────┘
```

### Worker Network
```
Host Observation (6-dim) + Goal Embedding (16-dim)
    ↓
MLP [128, 128] + LayerNorm + Tanh
    ↓
┌─────────────────┬─────────────────┐
│ Policy Head     │ Value Head      │
│ (5 actions)     │ (1 value)       │
└─────────────────┴─────────────────┘
```

## 动作映射

```python
# Manager输出: host_index (0-12)
# Worker输出: action_type (0-4)

action_mapping = {
    0: 0,        # sleep (无host)
    1: 1 + i,    # analyse_host[i]
    2: 14 + i,   # decoy_host[i]
    3: 27 + i,   # remove_host[i]
    4: 40 + i    # restore_host[i]
}
```

## 保存的模型

模型保存在 `hierarchical_mappo_models/` 目录:
- `{run_name}_iter_{N}.pt`: 每10个iteration的检查点
- `{run_name}_iter_final.pt`: 最终模型

## 加载预训练模型

```python
from train_hierarchical_mappo import HierarchicalMAPPOTrainer, make_env

env = make_env(red_policy="bline")
trainer = HierarchicalMAPPOTrainer(env=env)
trainer.load_checkpoint("path/to/model.pt")
```

## 故障排除

### Value Loss过高 (>10)
- 检查explained_variance是否>0.3
- 增大batch_size到512
- 降低learning_rate到0.001

### KL ≈ 0 (策略停滞)
- 增大entropy_coef到0.1
- 降低n_epochs到4
- 检查advantage normalization

### Reward不改善
- 确认red_policy是否正确设置
- 检查observation normalization是否启用
- 尝试不同的random seed

## 与SB3对比 (修复后)

本实现基于现代MARL最佳实践调整超参数:

| 参数 | SB3 PPO | 修复前 | 修复后 | 说明 |
|------|---------|--------|--------|------|
| Learning Rate | 0.002 | 0.002 | **3e-4** | 降低学习率提高稳定性 |
| Gamma | 0.99 | 0.99 | 0.99 | - |
| GAE Lambda | 0.95 | 0.95 | 0.95 | - |
| Clip Range | 0.2 | 0.2 | 0.2 | - |
| N Epochs | 6 | 6 | **10** | 更多epoch提高样本效率 |
| Batch Size | 64 | 64 | **256** | 更大batch稳定梯度 |
| Entropy Coef | 0.01 | 0.01 | **0.05** | 更高熵奖励促进探索 |
| Target KL | 0.015 | 0.015 | **0.02** | 减少不必要的early stop |

## 修复的Bug列表

1. **Worker Value复制** - Worker value现在独立计算而非复制manager value
2. **Explained Variance计算** - 使用当前值而非旧值计算
3. **Bootstrap Value获取** - 使用last_obs而非reset()获取
4. **PPO Ratio组合** - 分离manager/worker目标而非相乘
5. **Value Clipping** - 使用简单MSE而非clipped MSE
6. **Observation Normalization** - 修复第一次迭代不触发的问题
7. **Device不匹配** - 添加device=obs.device到torch.arange
8. **KL计算** - 使用abs mean更准确估计KL

## 训练结果

使用优化后的超参数训练1M steps，在B-line红队策略下：
- **最终Reward**: -72 (显著优于未修复版本的-126~-169)
- **Critic学习**: explained_variance稳定在0.5-0.8
- **策略稳定性**: clip_fraction控制在0.02-0.1

详细修复说明见 [HIERARCHICAL_MAPPO_FIXES.md](HIERARCHICAL_MAPPO_FIXES.md)

## 引用

```bibtex
@software{cyborgpp_hierarchical_mappo,
  title={Hierarchical MAPPO for CybORG++},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/CybORG_plus_plus}
}
```

## License

MIT License - 详见项目根目录 LICENSE 文件
