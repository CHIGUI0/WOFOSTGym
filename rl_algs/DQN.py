"""
Code to train a DQN Agent
Docs: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
Modified by: Will Solow, 2024
"""

import random
from argparse import Namespace
import time
from dataclasses import dataclass
import wandb
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy


@dataclass
class Args(RL_Args):
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 650
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    checkpoint_frequency: int = 500
    """How often to save the agent during training"""
    use_simple_exp_name: bool = True
    """If True, use exp_name directly; if False, use DQN/{env_id}__{exp_name} format"""


class DQN(nn.Module, Agent):
    def __init__(self, env: gym.Env, state_fpath: str = None, **kwargs: dict) -> None:
        super().__init__()
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

        if state_fpath is not None:
            assert isinstance(
                state_fpath, str
            ), f"`state_fpath` must be of type `str` but is of type `{type(state_fpath)}`"
            try:
                self.load_state_dict(torch.load(state_fpath, weights_only=True))
            except:
                msg = f"Error loading state dictionary from {state_fpath}"
                raise Exception(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_action(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Returns action from network. Helps with compatibility
        """
        return torch.argmax(self.network(x), dim=-1)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def train(kwargs: Namespace) -> None:
    """
    DQN Training Function
    """
    args = kwargs.DQN
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    if args.use_simple_exp_name and args.exp_name:
        # Use custom exp_name directly
        run_name = args.exp_name
    else:
        # Use default format: DQN/{env_id}__{exp_name}
        run_name = f"DQN/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer, device, envs = setup(kwargs, args, run_name)

    q_network = DQN(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = DQN(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # WSO tracking variables (for single env)
    episode_sum_wso = 0.0
    episode_max_wso = -np.inf
    episode_last_wso = 0.0

    obs, _ = envs.reset(seed=args.seed)
    for global_step in tqdm(range(args.total_timesteps), desc="DQN Training", unit="step"):

        if global_step % args.checkpoint_frequency == 0:
            torch.save(q_network.state_dict(), f"{kwargs.save_folder}{run_name}/agent.pt")
            if kwargs.track:
                wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Update WSO tracking from growth info
        if "growth" in infos:
            growth_info = infos["growth"]
            # Get the most recent WSO value (last key in growth dict)
            for key in reversed(list(growth_info.keys())):
                if isinstance(key, str) and key.startswith("_"):
                    continue
                values = growth_info.get(key)
                if values is not None:
                    wso_val = float(np.atleast_1d(values)[0])
                    episode_last_wso = wso_val
                    episode_sum_wso += wso_val
                    if wso_val > episode_max_wso:
                        episode_max_wso = wso_val
                    break

        # Handle episode completion
        episode_done = False
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_return = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    episode_done = True
                    break
        elif "episode" in infos:
            # Alternative format used by some gymnasium versions
            ep_info = infos["episode"]
            completed_mask = ep_info.get("_r")
            if completed_mask is None:
                completed_mask = ep_info.get("_l")
            if completed_mask is None:
                completed_indices = range(len(np.atleast_1d(ep_info["r"])))
            else:
                mask = np.array(completed_mask, dtype=bool)
                completed_indices = np.flatnonzero(mask)

            if len(completed_indices) > 0:
                returns_arr = np.asarray(ep_info["r"])
                lengths_arr = np.asarray(ep_info["l"])
                ep_return = float(returns_arr[completed_indices[0]])
                ep_length = float(lengths_arr[completed_indices[0]])
                episode_done = True

        if episode_done:
            # Get final WSO stats
            max_wso = episode_max_wso if np.isfinite(episode_max_wso) else episode_last_wso
            final_wso = episode_last_wso
            sum_wso = episode_sum_wso

            print(f"global_step={global_step}, episodic_return={ep_return:.2f}, "
                  f"max_wso={max_wso:.2f}, final_wso={final_wso:.2f}, sum_wso={sum_wso:.2f}")

            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar("episode/max_wso", max_wso, global_step)
            writer.add_scalar("episode/final_wso", final_wso, global_step)
            writer.add_scalar("episode/sum_wso", sum_wso, global_step)

            if kwargs.track:
                wandb.log({
                    "charts/episodic_return": ep_return,
                    "charts/episodic_length": ep_length,
                    "episode/max_wso": max_wso,
                    "episode/final_wso": final_wso,
                    "episode/sum_wso": sum_wso
                }, step=global_step)

            # Reset WSO tracking for next episode
            episode_sum_wso = 0.0
            episode_max_wso = -np.inf
            episode_last_wso = 0.0

        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % args.checkpoint_frequency == 0:
            writer.add_scalar("charts/average_reward", eval_policy(q_network, envs, kwargs, device), global_step)

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % args.checkpoint_frequency == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()
    writer.close()
