"""
Code to train a PPO Agent
Docs: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
Modified by: Will Solow, 2024
"""

import wandb
import time
from dataclasses import dataclass

from argparse import Namespace
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Optional
from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy
from tqdm import tqdm


@dataclass
class Args(RL_Args):
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 650
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    checkpoint_frequency: int = 500
    """How often to save the agent during training"""
    use_simple_exp_name: bool = True
    """If True, use exp_name directly; if False, use PPO/{env_id}__{exp_name} format"""

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer: nn.Module, std: np.ndarray = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module, Agent):
    def __init__(self, envs: gym.Env, state_fpath: str = None, **kwargs: dict) -> None:
        super().__init__()
        self.env = envs
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
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

    def get_action(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Helper function to get action for compatibility with generating data
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.from_numpy(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train(kwargs: Namespace) -> None:
    """
    PPO Training Function
    """

    args = kwargs.PPO
    if args.use_simple_exp_name and args.exp_name:
        # Use custom exp_name directly
        run_name = args.exp_name
    else:
        # Use default format: PPO/{env_id}__{exp_name}
        run_name = f"PPO/{kwargs.env_id}__{args.exp_name}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer, device, envs = setup(kwargs, args, run_name)

    agent = PPO(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Track episode statistics for batch-level aggregation
    batch_episode_returns = []  # Cumulative returns of completed episodes in current batch
    batch_episode_lengths = []  # Lengths of completed episodes in current batch
    batch_episode_max_wso = []
    batch_episode_final_wso = []
    batch_episode_sum_wso = []

    episode_sum_wso = np.zeros(args.num_envs)
    episode_max_wso = np.full(args.num_envs, -np.inf)
    episode_last_wso = np.zeros(args.num_envs)

    def update_wso_stats(growth_info: dict) -> None:
        """
        Extract the latest WSO value per environment from the info dict and update trackers.
        The info dict contains entries per date plus masks (key prefixed with '_') indicating
        which envs the value applies to. We walk the keys in reverse order so we only consume
        the most recent observation per environment for this step.
        """

        if not isinstance(growth_info, dict):
            return

        updated_envs = np.zeros(args.num_envs, dtype=bool)
        keys = list(growth_info.keys())
        for key in reversed(keys):
            if isinstance(key, str) and key.startswith("_"):
                continue

            values = growth_info.get(key)
            if values is None:
                continue
            values_arr = np.asarray(values)
            mask_arr = None
            mask_key = f"_{key}"
            if mask_key in growth_info:
                mask_arr = np.asarray(growth_info[mask_key])

            for env_idx in range(args.num_envs):
                if updated_envs[env_idx]:
                    continue
                if env_idx >= len(values_arr):
                    continue
                if mask_arr is not None:
                    if env_idx >= len(mask_arr) or not mask_arr[env_idx]:
                        continue

                wso_val = float(values_arr[env_idx])
                episode_last_wso[env_idx] = wso_val
                episode_sum_wso[env_idx] += wso_val
                if wso_val > episode_max_wso[env_idx]:
                    episode_max_wso[env_idx] = wso_val
                updated_envs[env_idx] = True

            if updated_envs.all():
                break

    # Create progress bar - track batches/iterations
    pbar = tqdm(total=args.num_iterations, desc="Training Progress", unit="batch")

    for iteration in range(1, args.num_iterations + 1):
        if global_step % args.checkpoint_frequency == 0:
            torch.save(agent.state_dict(), f"{kwargs.save_folder}{run_name}/agent.pt")
            if kwargs.track:
                wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Track WSO statistics per environment
            update_wso_stats(infos.get("growth"))

            episode_updates: list[tuple[int, float, float]] = []

            if "final_info" in infos:
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        ep_info = info["episode"]
                        episode_updates.append((env_idx, ep_info["r"], ep_info["l"]))
            elif "episode" in infos:
                ep_info = infos["episode"]
                completed_mask = ep_info.get("_r")
                if completed_mask is None:
                    completed_mask = ep_info.get("_l")
                if completed_mask is None:
                    # No explicit mask (SyncVectorEnv or gymnasium < 1.1) - assume all entries valid
                    completed_indices = range(len(np.atleast_1d(ep_info["r"])))
                else:
                    mask = np.array(completed_mask, dtype=bool)
                    completed_indices = np.flatnonzero(mask)

                returns_arr = np.asarray(ep_info["r"])
                lengths_arr = np.asarray(ep_info["l"])

                for env_idx in completed_indices:
                    episode_updates.append((env_idx, float(returns_arr[env_idx]), float(lengths_arr[env_idx])))

            for env_idx, ep_return, ep_length in episode_updates:
                max_wso = episode_max_wso[env_idx]
                if not np.isfinite(max_wso):
                    max_wso = episode_last_wso[env_idx]
                final_wso = episode_last_wso[env_idx]
                sum_wso = episode_sum_wso[env_idx]

                # Store episode stats for batch-level aggregation
                batch_episode_returns.append(ep_return)
                batch_episode_lengths.append(ep_length)
                batch_episode_max_wso.append(max_wso)
                batch_episode_final_wso.append(final_wso)
                batch_episode_sum_wso.append(sum_wso)

                # Reset trackers for next episode
                episode_max_wso[env_idx] = -np.inf
                episode_sum_wso[env_idx] = 0.0
                episode_last_wso[env_idx] = 0.0

                print(
                    f"iteration={iteration}, env={env_idx}, episode_return={ep_return:.2f}, "
                    f"episode_length={ep_length}, episode_max_wso={max_wso:.2f}, "
                    f"episode_final_wso={final_wso:.2f}, episode_sum_wso={sum_wso:.2f}",
                    flush=True,
                )
            if global_step % args.checkpoint_frequency == 0:
                # Skip eval_policy for AsyncVectorEnv due to pickling issues
                # AsyncVectorEnv uses multiprocessing which requires serialization
                if not isinstance(envs, gym.vector.AsyncVectorEnv):
                    avg_reward = eval_policy(agent, envs, kwargs, device)
                    writer.add_scalar("charts/average_reward", avg_reward, iteration)
                    if kwargs.track:
                        wandb.log({"eval/average_reward": avg_reward}, step=iteration)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Calculate batch-level episode statistics
        avg_episode_return = None
        avg_episode_length = None
        avg_episode_max_wso = None
        avg_episode_final_wso = None
        avg_episode_sum_wso = None
        num_episodes_completed = len(batch_episode_returns)

        if num_episodes_completed > 0:
            avg_episode_return = np.mean(batch_episode_returns)
            avg_episode_length = np.mean(batch_episode_lengths)
            avg_episode_max_wso = np.mean(batch_episode_max_wso)
            avg_episode_final_wso = np.mean(batch_episode_final_wso)
            avg_episode_sum_wso = np.mean(batch_episode_sum_wso)

            # Clear buffers for next batch
            batch_episode_returns.clear()
            batch_episode_lengths.clear()
            batch_episode_max_wso.clear()
            batch_episode_final_wso.clear()
            batch_episode_sum_wso.clear()

        # TensorBoard logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
        writer.add_scalar("losses/value_loss", v_loss.item(), iteration)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), iteration)
        writer.add_scalar("losses/entropy", entropy_loss.item(), iteration)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iteration)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), iteration)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), iteration)
        writer.add_scalar("losses/explained_variance", explained_var, iteration)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), iteration)
        writer.add_scalar("charts/elapsed_time", time.time() - start_time, iteration)
        if avg_episode_return is not None:
            writer.add_scalar("episode/avg_return", avg_episode_return, iteration)
            writer.add_scalar("episode/avg_length", avg_episode_length, iteration)
            writer.add_scalar("episode/avg_max_wso", avg_episode_max_wso, iteration)
            writer.add_scalar("episode/avg_final_wso", avg_episode_final_wso, iteration)
            writer.add_scalar("episode/avg_sum_wso", avg_episode_sum_wso, iteration)
            writer.add_scalar("episode/num_completed", num_episodes_completed, iteration)

        # Wandb logging - log once per batch/iteration
        if kwargs.track:
            log_dict = {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/value_loss": v_loss.item(),
                "train/policy_loss": pg_loss.item(),
                "train/entropy": entropy_loss.item(),
                "train/old_approx_kl": old_approx_kl.item(),
                "train/approx_kl": approx_kl.item(),
                "train/clipfrac": np.mean(clipfracs),
                "train/explained_variance": explained_var,
                "metrics/SPS": int(global_step / (time.time() - start_time)),
                "metrics/elapsed_time": time.time() - start_time,
                "metrics/global_step": global_step
            }

            # Add episode statistics if we have completed episodes
            if avg_episode_return is not None:
                log_dict.update({
                    "episode/avg_return": avg_episode_return,
                    "episode/avg_length": avg_episode_length,
                    "episode/avg_max_wso": avg_episode_max_wso,
                    "episode/avg_final_wso": avg_episode_final_wso,
                    "episode/avg_sum_wso": avg_episode_sum_wso,
                    "episode/num_completed": num_episodes_completed
                })

            wandb.log(log_dict, step=iteration)

        # Update progress bar with current metrics (update by 1 batch)
        pbar.update(1)
        postfix_dict = {
            "v_loss": f"{v_loss.item():.4f}",
            "p_loss": f"{pg_loss.item():.4f}",
            "SPS": int(global_step / (time.time() - start_time))
        }
        if avg_episode_return is not None:
            postfix_dict["avg_ret"] = f"{avg_episode_return:.1f}"
            postfix_dict["n_eps"] = num_episodes_completed
        pbar.set_postfix(postfix_dict)

    pbar.close()
    envs.close()
    writer.close()
