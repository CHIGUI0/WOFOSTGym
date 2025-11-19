# WOFOSTGym Environment Configuration Guide

This guide explains the various configuration options available when training RL agents in WOFOSTGym environments.

## Table of Contents
- [Environment Template Selection (`env_id`)](#environment-template-selection-env_id)
- [Agromanagement Configuration](#agromanagement-configuration)
- [Site Variation vs Crop Variety](#site-variation-vs-crop-variety)
- [Randomization Options](#randomization-options)
- [Limited vs Potential Production](#limited-vs-potential-production)
- [Single-Farm vs Multi-Farm Environments](#single-farm-vs-multi-farm-environments)
- [NPK Environment Parameters](#npk-environment-parameters)
- [Training Recommendations](#training-recommendations)

---

## Environment Template Selection (`env_id`)

The environment ID (`--env-id`) determines the action space, observation space, and resource constraints.

### Environment Naming Convention

Format: `[prefix-][crop_type-]resource_limitations-v0`

### Resource Limitation Types

| env_id | Full Name | Limitations | Use Case |
|--------|-----------|-------------|----------|
| `lnpkw-v0` | Limited N/P/K/Water | N, P, K, Water | **Most realistic** - Complete nutrient and water management |
| `lnpk-v0` | Limited N/P/K | N, P, K only | Irrigated agriculture - Focus on fertilization |
| `ln-v0` | Limited Nitrogen | N only | Simplified scenario - Nitrogen management only |
| `lnw-v0` | Limited N + Water | N, Water | Rainfed agriculture - N fertilizer + irrigation |
| `lw-v0` | Limited Water | Water only | Irrigation scheduling only |
| `pp-v0` | Potential Production | **None** | Theoretical maximum yield - No management needed |

### Crop Types

| Prefix | Description | Growing Season | Weather Years |
|--------|-------------|----------------|---------------|
| (none) | Annual crops (24 crops) | Single season (e.g., wheat, maize) | 1984-2019 (36 years) |
| `perennial-` | Perennial crops (jujube, pear) | Multi-year | 1984-2019 (36 years) |
| `grape-` | Grape (dedicated env) | Multi-year | 1984-2019 (36 years) |
| `multi-` | Multi-farm | Multiple farms simultaneously | 1984-2023 (40 years) |

### Action Types

| Prefix | Actions Available |
|--------|-------------------|
| (default) | Fertilization + Irrigation only |
| `plant-` | Planting + Fertilization + Irrigation + Harvesting |
| `harvest-` | Fertilization + Irrigation + Harvesting |

### Soil Models

| Prefix | Soil Model |
|--------|------------|
| (default) | Single-layer water balance |
| `l` prefix | Multi-layered water balance (e.g., `llnpkw-v0`) |

### Examples

```bash
# Basic annual crop with full resource management
python3 train_agent.py --env-id lnpkw-v0

# Multi-farm environment
python3 train_agent.py --env-id multi-lnpk-v0

# Planting decisions included
python3 train_agent.py --env-id plant-lnpkw-v0

# Layered soil model
python3 train_agent.py --env-id llnpkw-v0
```

---

## Agromanagement Configuration

The agromanagement file (`--agro-file`) specifies all physical inputs for the simulation.

### File Location
`env_config/agro/{crop}_agro.yaml`

### Key Parameters

```yaml
AgroManagement:
  SiteCalendar:
    latitude: 50.0              # Location latitude (affects weather)
    longitude: 5.0              # Location longitude
    year: 1984                  # Simulation year
    site_name: oregon           # Site name (links to site YAML)
    site_variation: oregon_1    # Site variation (soil/climate variant)
    site_start_date: 1985-02-01 # Simulation start date
    site_end_date: 1985-10-01   # Simulation end date

  CropCalendar:
    crop_name: wheat            # Crop type
    crop_variety: wheat_1       # Crop variety/cultivar
    crop_start_date: 1985-02-01 # Planting/emergence date
    crop_start_type: sowing     # sowing | emergence
    crop_end_date: 1985-10-01   # Harvest/maturity date
    crop_end_type: maturity     # harvest | maturity | max_duration
    max_duration: 365           # Maximum growing season (days)
```

### Available Configurations

**Sites**: `oregon`, `washington`, `china`, `moregon`
- Each site has multiple variations (e.g., `oregon_1`, `oregon_2`, `oregon_3`)

**Annual Crops (24 crops)**:
```
wheat, maize, rice, potato, soybean, cotton, sunflower, sugarbeet,
barley, cassava, chickpea, cowpea, fababean, groundnut, millet,
mungbean, pigeonpea, rapeseed, seed_onion, sorghum, sugarcane,
sweetpotato, tobacco
```

**Perennial Crops (3 crops)**:
```
grape, jujube, pear
```
- Weather years: 1984-2019 (36 years, same as annual crops)
- Use `perennial-` prefix for env_id (e.g., `perennial-lnpkw-v0`)
- Grape has dedicated environments: `grape-lnpkw-v0` (no prefix needed)

- Each crop has multiple varieties (e.g., `wheat_1` through `wheat_9`)

---

## Site Variation vs Crop Variety

### Site Variation (Environmental Conditions)

**What it represents**: Soil and climate conditions (the farm itself)

**File location**: `env_config/site/{site_name}.yaml`

**Example parameters**:
```yaml
oregon_1:  # Standard soil
  CO2: 360.0                    # Atmospheric CO2 (ppm)
  SMFCF: 0.460                  # Field capacity (cm³/cm³)
  RDMSOL: 120.0                 # Maximum rooting depth (cm)
  RNABSORPTION: 0.9             # N absorption rate to subsoil

oregon_2:  # More water-retentive soil
  CO2: 340.0
  SMFCF: 0.530                  # Higher field capacity
  RDMSOL: 170.0                 # Deeper soil

oregon_3:  # Poor nutrient retention
  RNABSORPTION: 0.1             # Low N absorption (high runoff)
  RNSOILMAX: 10                 # Low max N absorption rate
```

**Impact**: Different environmental growing conditions for the same crop

### Crop Variety (Plant Genetics)

**What it represents**: Plant genetic characteristics (the seed variety)

**File location**: `env_config/crop/{crop_name}.yaml`

**Example parameters**:
```yaml
wheat_1:  # Early flowering variety
  TSUM1: 543                    # Temperature sum to anthesis (°C·day)
  TSUM2: 1194                   # Temperature sum to maturity (°C·day)
  DLO: 16.8                     # Optimum day length (hr)
  VERNBASE: 10.0                # Base vernalization requirement (days)

wheat_2:  # Late flowering variety
  TSUM1: 853                    # Later flowering
  TSUM2: 975                    # Faster maturation
  DLO: 16.3
  VERNBASE: 9.0
```

**Impact**: Different growth patterns and resource requirements

### Key Distinction

| Aspect | Site Variation | Crop Variety |
|--------|----------------|--------------|
| **Represents** | 🌍 Soil/climate conditions | 🌾 Plant genetics |
| **Analogy** | Different farms/fields | Different seed varieties |
| **Controls** | Soil nutrients, water retention, CO₂ | Growth rate, phenology, nutrient demand |
| **Example** | "Sandy soil farm" vs "Clay soil farm" | "Early-maturing wheat" vs "Late-maturing wheat" |

---

## Randomization Options

Control environment diversity during training to improve generalization.

### Available Flags

```python
random_reset: bool = False      # Reset to random weather year
train_reset: bool = False       # Reset to specific training years
domain_rand: bool = False       # Randomize WOFOST parameters each reset
crop_rand: bool = False         # Randomize parameters once at initialization
```

### 1. `random_reset` - Weather Year Randomization

**What it randomizes**: Weather year selection

**Selection pool**:
- Annual crop environments: 1984-2019 (36 years)
- Perennial crop environments: 1984-2019 (36 years)
- Multi-farm environments: 1984-2023 (40 years)

**When applied**: Every `env.reset()`

**Example**:
```bash
python3 train_agent.py --agent-type PPO \
  --npk.random-reset True
```

**Effect**: Each episode uses different historical weather data while keeping soil and crop parameters constant.

### 2. `train_reset` - Limited Year Range

**What it randomizes**: Weather year from restricted set

**Selection pool**: 1990-1994 (5 years only)

**When applied**: Every `env.reset()`

**Example**:
```bash
python3 train_agent.py --agent-type PPO \
  --npk.train-reset True
```

**Note**: Overrides `random_reset` if both are enabled.

### 3. `domain_rand` - Domain Randomization

**What it randomizes**: ALL float-type WOFOST parameters (100+ parameters)

**Distribution**: Normal distribution with configurable scale

**When applied**: Every `env.reset()`

**Parameters affected**:
- **Crop parameters**: TSUM1, TSUM2, AMAXTB, EFFTB, FRTB, NMAXLV_TB, etc.
- **Site parameters**: NAVAILI, PAVAILI, SMFCF, SM0, RDMSOL, NSOILBASE, etc.

**Formula**:
```python
new_value = original_value + original_value * N(0, scale)
```

**Example**:
```bash
python3 train_agent.py --agent-type PPO \
  --npk.random-reset True \
  --npk.domain-rand True \
  --npk.scale 0.15
```

**Example effect** (scale=0.1):
```python
TSUM1 = 543 → 543 ± 54.3 → approximately [489, 597]
SMFCF = 0.46 → 0.46 ± 0.046 → approximately [0.414, 0.506]
```

### 4. `crop_rand` - Initialization Randomization

**What it randomizes**: Same as `domain_rand` but uniform distribution

**Distribution**: Uniform distribution

**When applied**: Once at environment initialization only

**Use case**: Generate diverse expert demonstration data

**Formula**:
```python
new_value = original_value + U(-original_value*scale, original_value*scale)
```

**Example**:
```bash
python3 train_agent.py --agent-type PPO \
  --npk.crop-rand True \
  --npk.scale 0.1
```

### Randomization Comparison

| Feature | random_reset | train_reset | domain_rand | crop_rand |
|---------|--------------|-------------|-------------|-----------|
| **Randomizes** | Weather year | Weather year | WOFOST params | WOFOST params |
| **Pool size** | 36-40 years | 5 years | ∞ (continuous) | ∞ (continuous) |
| **Distribution** | Discrete | Discrete | Normal | Uniform |
| **Frequency** | Every reset | Every reset | Every reset | Once at init |
| **Use case** | Multi-year generalization | Limited training data | Strong generalization | Data generation |

### Recommended Training Progression

```bash
# Stage 1: Fixed environment (algorithm debugging)
python3 train_agent.py --agent-type PPO \
  --save-folder logs/stage1/ \
  --PPO.total-timesteps 500000

# Stage 2: Weather variability
python3 train_agent.py --agent-type PPO \
  --save-folder logs/stage2/ \
  --npk.random-reset True \
  --PPO.total-timesteps 2000000

# Stage 3: Full domain randomization (production-ready)
python3 train_agent.py --agent-type PPO \
  --save-folder logs/stage3/ \
  --npk.random-reset True \
  --npk.domain-rand True \
  --npk.scale 0.1 \
  --PPO.total-timesteps 5000000
```

---

## Limited vs Potential Production

The "Limited" designation refers to **soil resource availability**, NOT action space constraints.

### Key Distinction

❌ **NOT**: Limit on how much fertilizer/water the agent can apply
✅ **YES**: Soil resources (N/P/K/Water) are finite and can be depleted

### Potential Production (PP)

**Soil module**: `SoilModuleWrapper_PP`

**Characteristics**:
- ✅ Unlimited N/P/K/Water supply in soil
- ✅ Crops never experience nutrient or water stress
- ❌ No management actions allowed
- 📊 Yield limited only by light, temperature, and variety

**Action space**: `Discrete(1)` - Only "do nothing"

**Observation example**:
```python
{
    'NAVAIL': 999,    # Always unlimited
    'PAVAIL': 999,    # Always unlimited
    'KAVAIL': 999,    # Always unlimited
    'SM': 0.45,       # Always optimal
    'WSO': 8000       # Theoretical maximum yield
}
```

**Use case**: Baseline for potential yield assessment

### Limited NPKW Production (LNPKW)

**Soil module**: `SoilModuleWrapper_LNPKW`

**Characteristics**:
- ⚠️ Finite N/P/K pools (start at ~0-10 kg/ha)
- ⚠️ Resources deplete over time (crop uptake + losses)
- ⚠️ Maximum pool limits (e.g., NMAX=50 kg/ha)
- ⚠️ Water stress occurs if soil moisture drops
- ✅ Agent CAN apply fertilizer/irrigation (unlimited application ability)
- ✅ Applied nutrients/water have recovery efficiency < 1.0

**Action space**: `Discrete(1 + 3*num_fert + num_irrig)`
- Example: 1 + 12 + 4 = 17 actions

**Observation example**:
```python
# Time step 1 (initial)
{
    'NAVAIL': 5.0,     # Limited soil N
    'PAVAIL': 3.0,     # Limited soil P
    'KAVAIL': 8.0,     # Limited soil K
    'SM': 0.35,        # Suboptimal moisture
    'WSO': 100         # Low yield due to stress
}

# Agent applies N fertilizer (4 kg/ha)

# Time step 2 (after fertilization)
{
    'NAVAIL': 7.8,     # Increased (5 + 4*0.7 recovery)
    'PAVAIL': 2.5,     # Depleting (crop uptake)
    'KAVAIL': 7.2,     # Depleting
    'SM': 0.33,        # Drying
    'WSO': 150         # Improved yield
}

# Time step 10 (without management)
{
    'NAVAIL': 0.5,     # Nearly depleted!
    'PAVAIL': 0.2,     # Severely limited
    'SM': 0.25,        # Drought stress
    'WSO': 180         # Yield plateauing
}
# Agent must decide: Apply N? Apply P? Irrigate?
```

### Soil Dynamics in LNPKW

From `env_config/site/oregon.yaml`:
```yaml
NAVAILI: 0.0           # Initial available N (kg/ha)
PAVAILI: 0.0           # Initial available P (kg/ha)
KAVAILI: 0.0           # Initial available K (kg/ha)
NMAX: 50.0             # Maximum N pool (kg/ha)
PMAX: 50.0             # Maximum P pool (kg/ha)
KMAX: 50.0             # Maximum K pool (kg/ha)
NSOILBASE: 10.0        # Base soil N supply via mineralization (kg/ha)
NSOILBASE_FR: 0.025    # Daily release fraction (2.5%/day)
```

**Nutrient dynamics**:
1. Starts low (NAVAILI=0)
2. Background supply from mineralization (NSOILBASE × NSOILBASE_FR daily)
3. Crop uptake reduces pool
4. Fertilization adds to pool (with recovery efficiency)
5. Losses via leaching/runoff
6. Cannot exceed NMAX

### Environment Comparison

| Aspect | PP Environment | LNPKW Environment |
|--------|----------------|-------------------|
| **Soil N/P/K** | Infinite | Finite (0-50 kg/ha) |
| **Nutrient depletion** | Never | Yes, continuous |
| **Yield without management** | 8000 kg/ha | 1500 kg/ha |
| **Yield with optimal management** | 8000 kg/ha | 7500 kg/ha |
| **Agent actions** | None (only observe) | Fertilize + Irrigate |
| **Management value** | 0 | +6000 kg/ha |
| **Training difficulty** | ⭐ | ⭐⭐⭐⭐⭐ |
| **Realistic** | ❌ | ✅ |

### Recommended Environment by Use Case

| Research Goal | Recommended Environment | Reason |
|---------------|------------------------|---------|
| Algorithm debugging | `pp-v0` | Fast convergence, verify code |
| Irrigation optimization | `lw-v0` | Focus on water management |
| Fertilization strategy | `lnpk-v0` | Focus on N/P/K optimization |
| Integrated management | `lnpkw-v0` | **Most realistic and valuable** |
| Rainfed agriculture | `lnw-v0` | N fertilizer + limited irrigation |

---

## Single-Farm vs Multi-Farm Environments

### Single-Farm Environment

**Class**: `NPK_Env`

**Characteristics**:
- Manages **1 farm**
- 1 WOFOST crop model
- Agent action affects single farm
- Observation = single farm state

**Example env_ids**: `lnpkw-v0`, `harvest-lnpk-v0`, `plant-ln-v0`

**Observation space dimension**:
```
1 + len(output_vars) + len(weather_vars) * forecast_length
Example: 1 + 11 + 3 = 15 dimensions
```

### Multi-Farm Environment

**Class**: `Multi_NPK_Env`

**Characteristics**:
- Manages **multiple farms** simultaneously (default: 5)
- Each farm has independent WOFOST model
- **Same weather for all farms**
- **Same action applied to all farms**
- Observation = concatenated state from all farms
- Reward = sum of yields from all farms

**Example env_ids**: `multi-lnpkw-v0`, `multi-harvest-lnpk-v0`

**Observation space dimension**:
```
1 + len(individual_vars) * num_farms + len(shared_vars) + len(weather_vars) * forecast_length
Example (5 farms): 1 + 7*5 + 4 + 3 = 43 dimensions
```

**Shared vs Individual Features**:
```python
# Individual features (per-farm states)
individual_vars = ['DVS', 'LAI', 'WSO', 'SM', 'NAVAIL', 'PAVAIL', 'KAVAIL']

# Shared features (cumulative across all farms)
shared_vars = ['TOTN', 'TOTP', 'TOTK', 'TOTIRRIG']
```

**Example**:
```bash
# Single farm
python3 train_agent.py --env-id lnpkw-v0

# Multi-farm (5 farms by default)
python3 train_agent.py --env-id multi-lnpkw-v0

# Multi-farm with custom number
python3 train_agent.py --env-id multi-lnpkw-v0 \
  --npk.num-farms 10 \
  --npk.domain-rand True  # Each farm has different parameters
```

### Weather Data Years

| Environment Type | Weather Years | Total Years |
|-----------------|---------------|-------------|
| Annual crops (e.g., `lnpkw-v0`) | 1984-2019 | 36 years |
| Perennial crops (e.g., `perennial-lnpkw-v0`) | 1984-2019 | 36 years |
| Multi-farm (e.g., `multi-lnpk-v0`) | 1984-2023 | 40 years |

### Comparison

| Aspect | Single-Farm | Multi-Farm |
|--------|-------------|------------|
| **Use case** | Precision agriculture (single field) | Large-scale farming (multiple fields) |
| **Complexity** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Observation dim** | ~15 | ~43 (for 5 farms) |
| **Training difficulty** | Moderate | Hard |
| **Realism** | Individual field management | Farm/cooperative management |
| **Action impact** | Single field | All fields simultaneously |

---

## NPK Environment Parameters

Key parameters in `NPK_Args` that affect environment behavior.

### Action Space Configuration

```python
num_fert: int = 4              # Number of fertilization levels
num_irrig: int = 4             # Number of irrigation levels
fert_amount: float = 2.0       # Fertilizer amount per level (kg/ha)
irrig_amount: float = 0.5      # Irrigation amount per level (cm)
```

**Action space size**:
```
Total actions = 1 (no-op) + 3*num_fert (N,P,K) + num_irrig
Example: 1 + 3*4 + 4 = 17 actions
```

**Fertilizer levels**:
```
Action 1: Apply 1 * fert_amount = 2 kg N/ha
Action 2: Apply 2 * fert_amount = 4 kg N/ha
Action 3: Apply 3 * fert_amount = 6 kg N/ha
Action 4: Apply 4 * fert_amount = 8 kg N/ha
(Similar for P and K)
```

### Efficiency Parameters

```python
n_recovery: float = 0.7        # Nitrogen recovery efficiency (70%)
p_recovery: float = 0.7        # Phosphorus recovery efficiency (70%)
k_recovery: float = 0.7        # Potassium recovery efficiency (70%)
irrig_effec: float = 0.7       # Irrigation efficiency (70%)
harvest_effec: float = 1.0     # Harvest efficiency (100%)
```

**Impact**: Applied fertilizer × recovery = actual soil addition
```python
# Agent applies 10 kg N
# Soil receives: 10 * 0.7 = 7 kg N
# Lost: 3 kg (runoff, volatilization, etc.)
```

### Observation Configuration

```python
output_vars: list = [          # Crop state variables to observe
    'DVS',      # Development stage [0-2]
    'LAI',      # Leaf area index [m²/m²]
    'SM',       # Soil moisture [cm³/cm³]
    'TAGP',     # Total above ground production [kg/ha]
    'NAVAIL',   # Available nitrogen [kg/ha]
    'PAVAIL',   # Available phosphorus [kg/ha]
    'KAVAIL',   # Available potassium [kg/ha]
    'WSO',      # Weight storage organs (yield) [kg/ha]
    # ... more variables
]

weather_vars: list = [         # Weather variables to observe
    'IRRAD',    # Solar radiation [kJ/m²/day]
    'TEMP',     # Temperature [°C]
    'RAIN',     # Rainfall [cm/day]
]

forecast_length: int = 1       # Weather forecast horizon (days)
forecast_noise: list = [0, 0.2] # Forecast noise [min, max]
```

### Intervention Timing

```python
intvn_interval: int = 1        # Days between agent decisions
```

**Example**:
- `intvn_interval=1`: Agent makes decision daily
- `intvn_interval=7`: Agent makes decision weekly

### Randomization Parameters

```python
seed: int = 0                  # Random seed
scale: float = 0.1             # Domain randomization scale
random_reset: bool = False     # Random year reset
train_reset: bool = False      # Training years reset
domain_rand: bool = False      # Domain randomization
crop_rand: bool = False        # Crop initialization randomization
```

### Example Configuration

```bash
python3 train_agent.py --agent-type PPO \
  --env-id lnpkw-v0 \
  --npk.num-fert 8 \           # More fertilization options
  --npk.num-irrig 6 \          # More irrigation options
  --npk.fert-amount 1.5 \      # Smaller increments
  --npk.n-recovery 0.75 \      # Better N efficiency
  --npk.forecast-length 3 \    # 3-day weather forecast
  --npk.intvn-interval 7 \     # Weekly decisions
  --npk.random-reset True \    # Random years
  --npk.domain-rand True \     # Parameter randomization
  --npk.scale 0.15             # Randomization strength
```

---

## Training Recommendations

### Environment Selection by Experience Level

**Beginners**:
```bash
# Start with simple nitrogen-only management
python3 train_agent.py --env-id ln-v0
```

**Intermediate**:
```bash
# Add water management
python3 train_agent.py --env-id lnw-v0 \
  --npk.random-reset True
```

**Advanced**:
```bash
# Full resource management with domain randomization
python3 train_agent.py --env-id lnpkw-v0 \
  --npk.random-reset True \
  --npk.domain-rand True \
  --npk.scale 0.15
```

### Recommended Parameter Combinations

**Fast prototyping** (low computational cost):
```bash
python3 train_agent.py --env-id ln-v0 \
  --npk.num-fert 4 \
  --npk.num-irrig 0 \
  --npk.intvn-interval 7 \
  --PPO.total-timesteps 500000
```

**Balanced training**:
```bash
python3 train_agent.py --env-id lnpkw-v0 \
  --npk.num-fert 6 \
  --npk.num-irrig 5 \
  --npk.random-reset True \
  --npk.intvn-interval 3 \
  --PPO.total-timesteps 2000000
```

**Production-ready** (maximum generalization):
```bash
python3 train_agent.py --env-id lnpkw-v0 \
  --npk.num-fert 8 \
  --npk.num-irrig 8 \
  --npk.random-reset True \
  --npk.domain-rand True \
  --npk.scale 0.15 \
  --npk.intvn-interval 1 \
  --PPO.total-timesteps 5000000
```

### Generalization Strategy

**Stage 1**: Fixed environment (verify algorithm)
```bash
python3 train_agent.py --env-id lnpkw-v0 \
  --agro-file wheat_agro.yaml \
  --PPO.total-timesteps 500000
```

**Stage 2**: Weather variability
```bash
python3 train_agent.py --env-id lnpkw-v0 \
  --npk.random-reset True \
  --PPO.total-timesteps 2000000
```

**Stage 3**: Full randomization
```bash
python3 train_agent.py --env-id lnpkw-v0 \
  --npk.random-reset True \
  --npk.domain-rand True \
  --npk.scale 0.1 \
  --PPO.total-timesteps 5000000
```

**Stage 4**: Multi-crop evaluation
```bash
# Train on wheat
python3 train_agent.py --env-id lnpkw-v0 \
  --agro-file wheat_agro.yaml \
  --npk.domain-rand True

# Test on maize
python3 eval_agent.py --env-id lnpkw-v0 \
  --agro-file maize_agro.yaml \
  --load-model logs/wheat_model.pt
```

### Common Pitfalls

❌ **Don't**: Train on PP environment
- Agent learns nothing (no actions needed)

❌ **Don't**: Use only `crop_rand` without `domain_rand`
- Parameters fixed after initialization, no diversity during training

❌ **Don't**: Set `scale` too high (>0.3)
- Unrealistic parameter values, unstable training

❌ **Don't**: Use very large `intvn_interval` (>14 days)
- Miss critical intervention windows

✅ **Do**: Start simple, gradually increase complexity

✅ **Do**: Enable `random_reset` for realistic training

✅ **Do**: Use `domain_rand` with moderate `scale` (0.1-0.15)

✅ **Do**: Monitor key metrics: NAVAIL, SM, WSO, total reward

---

## Quick Reference

### Complete Training Command Example

```bash
python3 train_agent.py \
  --agent-type PPO \
  --save-folder logs/wheat_lnpkw/ \
  \
  --env-id lnpkw-v0 \
  --agro-file wheat_agro.yaml \
  \
  --npk.num-fert 6 \
  --npk.num-irrig 5 \
  --npk.fert-amount 2.0 \
  --npk.irrig-amount 0.5 \
  --npk.n-recovery 0.7 \
  --npk.p-recovery 0.7 \
  --npk.k-recovery 0.7 \
  --npk.irrig-effec 0.7 \
  --npk.forecast-length 3 \
  --npk.intvn-interval 3 \
  \
  --npk.random-reset True \
  --npk.domain-rand True \
  --npk.scale 0.12 \
  --npk.seed 42 \
  \
  --PPO.learning-rate 2.5e-4 \
  --PPO.total-timesteps 3000000 \
  --PPO.num-steps 650 \
  --PPO.gamma 0.999
```

### Environment Comparison Matrix

| Feature | pp-v0 | ln-v0 | lnw-v0 | lnpk-v0 | lnpkw-v0 | multi-lnpkw-v0 |
|---------|-------|-------|--------|---------|----------|----------------|
| N limited | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| P limited | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| K limited | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Water limited | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| Actions | 1 | 5 | 9 | 13 | 17 | 17 |
| Obs dims | 15 | 15 | 15 | 15 | 15 | 43 |
| Difficulty | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Realism | Low | Medium | High | High | **Highest** | Highest |

---

## Additional Resources

- **WOFOST Documentation**: http://pcse.readthedocs.io
- **NASA POWER Weather Data**: https://power.larc.nasa.gov/
- **Environment Registration**: `pcse_gym/pcse_gym/__init__.py`
- **Environment Implementation**: `pcse_gym/pcse_gym/envs/wofost_annual.py`
- **Configuration Files**: `env_config/agro/`, `env_config/site/`, `env_config/crop/`

---

## Troubleshooting

**Q**: Training is too slow
- **A**: Reduce `total_timesteps`, increase `intvn_interval`, use simpler env (e.g., `ln-v0`)

**Q**: Agent doesn't learn
- **A**: Check if using `pp-v0` (no actions needed), verify reward signal, reduce action space size

**Q**: Poor generalization to new years
- **A**: Enable `random_reset` and `domain_rand` during training

**Q**: Environment crashes with parameter errors
- **A**: Check `scale` value (<0.3), verify YAML files exist, check year range in weather data

**Q**: What's a good baseline reward?
- **A**: Depends on environment and crop. For wheat `lnpkw-v0`: no-management ≈ 1500 kg/ha, optimal ≈ 7500 kg/ha

---
