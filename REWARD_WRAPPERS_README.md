# Custom Reward Wrappers

This document describes the new reward wrapper implementations for WOFOST Gym.

## Available Reward Wrappers

### 0. Default WSO Reward (Dense Reward - Daily WSO)

**Purpose**: Provides reward at every step based on current WSO value (default behavior).

**Behavior**:
- Each step receives reward = **current WSO**
- Dense feedback signal at every timestep
- Encourages maintaining high WSO throughout the episode
- Default reward function when no wrapper is specified

**Normalization**:
- Automatically normalized by `NormalizeReward` wrapper
- reward_range: `[0, 10000]` (set by wrapper default)
- Normalized formula: `normalized_reward = WSO / 10000`
- Typical normalized range: `[0, ~1.26]` (max observed WSO ~12590)

**Usage in train.sh**:
```bash
REWARD_TYPE=""  # Empty string for default reward
REWARD_SUFFIX=""  # No suffix for default reward
```

**Usage with env_config**:
```python
from agrimanager.env.wofost_gym import WOFOSTEnvConfig

config = WOFOSTEnvConfig(
    env_reward=None  # or don't specify env_reward
)
```

### 1. RewardFinalWSOWrapper (Sparse Reward - Final WSO)

**Purpose**: Provides reward only at episode termination based on final WSO value.

**Behavior**:
- All intermediate steps receive **0** reward
- Final step (when terminated or truncated) receives the **final WSO** value
- Encourages the agent to maximize final yield rather than short-term gains

**Normalization**:
- Automatically normalized by `NormalizeReward`
- reward_range: `[0, 1000]`
- Normalized formula: `normalized_reward = final_wso / 1000`
- With observed final WSO roughly in `[0, ~7300]`, normalized rewards can exceed `1.0`

**Limitations**:
- WSO often declines after reaching peak maturity (avg 77% loss observed)
- May not reflect true crop yield potential if episode ends after peak

**Usage in train.sh**:
```bash
REWARD_TYPE="RewardFinalWSOWrapper"
REWARD_SUFFIX="_final_wso"
```

**Usage with env_config**:
```python
from agrimanager.env.wofost_gym import WOFOSTEnvConfig

config = WOFOSTEnvConfig(
    env_reward="RewardFinalWSOWrapper"
)
```

### 2. RewardPeakWSOWrapper (Sparse Reward - Peak WSO) **[RECOMMENDED]**

**Purpose**: Provides reward based on the maximum WSO achieved during the episode.

**Behavior**:
- All intermediate steps receive **0** reward
- Tracks maximum WSO throughout the episode
- Final step receives the **peak WSO** value (not final WSO)
- Better reflects agricultural management quality

**Advantages**:
- Not affected by WSO decline after maturity
- Rewards maximizing crop yield potential
- More appropriate for crops where WSO declines post-peak (observed avg 77% decline)

**Usage in train.sh**:
```bash
REWARD_TYPE="RewardPeakWSOWrapper"
REWARD_SUFFIX="_peak_wso"
```

**Usage with env_config**:
```python
from agrimanager.env.wofost_gym import WOFOSTEnvConfig

config = WOFOSTEnvConfig(
    env_reward="RewardPeakWSOWrapper"
)
```

### 3. RewardWSODeltaWrapper (Dense Reward - WSO Delta)

**Purpose**: Provides dense feedback at every step based on WSO change.

**Behavior**:
- Each step receives reward = **WSO(t) - WSO(t-1)**
- Positive reward when WSO increases
- Negative or zero reward when WSO decreases or stays the same
- Provides frequent feedback to guide learning

**Usage in train.sh**:
```bash
REWARD_TYPE="RewardWSODeltaWrapper"
REWARD_SUFFIX="_wso_delta"
```

**Usage with env_config**:
```python
from agrimanager.env.wofost_gym import WOFOSTEnvConfig

config = WOFOSTEnvConfig(
    env_reward="RewardWSODeltaWrapper"
)
```

## Comparison

| Wrapper | Reward Type | When Rewarded | Reward Value | Return Formula |
|---------|-------------|---------------|--------------|----------------|
| **Daily WSO** (Default) | Dense | Every step | Current WSO | G = Σ γ^t · WSO(t) |
| **RewardFinalWSOWrapper** | Sparse | Episode end | Final WSO | G = γ^T · WSO(T) |
| **RewardPeakWSOWrapper** ⭐ | Sparse | Episode end | Peak WSO | G = γ^T · max_t WSO(t) |
| **RewardWSODeltaWrapper** | Dense | Every step | WSO(t) - WSO(t-1) | G = Σ γ^t · [WSO(t) - WSO(t-1)] |

**Notation**:
- G: Expected return
- γ: Discount factor (0.999)
- t: Time step
- T: Final time step (episode end)
- WSO(t): WSO at time step t

## Implementation Details

### RewardFinalWSOWrapper

The implementation overrides the `step()` method to check for episode termination:
- `terminated`: Crop has finished growing
- `truncated`: Episode has reached maximum timesteps

Only when either condition is True, the final WSO is given as reward.

### RewardPeakWSOWrapper

The implementation tracks the maximum WSO throughout the episode:
- On `reset()`: Initialize `peak_wso = 0.0`
- On each `step()`: Update `peak_wso = max(peak_wso, current_wso)`
- On termination/truncation: Return `peak_wso` as reward
- Intermediate steps receive 0 reward

### RewardWSODeltaWrapper

The implementation tracks `prev_wso` across steps:
- On `reset()`: Initialize `prev_wso = 0.0`
- On each `step()`: Calculate `reward = current_wso - prev_wso`, then update `prev_wso = current_wso`

## Notes

- All wrappers are compatible with Multi_NPK_Env for multi-farm scenarios
- All wrappers handle None/NaN WSO values gracefully
- Reward ranges (based on empirical observations from baselines):
  - **Daily WSO** (Default): [0, 10000] (borrowed from original code)
  - **RewardFinalWSOWrapper**: [0, 1000] (normalized as `final_wso / 1000`; observed final WSO range: [0, ~7300])
  - **RewardPeakWSOWrapper**: [0, 15000] (observed peak WSO range: [0, ~12590])
  - **RewardWSODeltaWrapper**: [-1500, 500] (observed WSO delta range: [-1056, 363])
