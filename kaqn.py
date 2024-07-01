import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import time
import random

import gymnasium as gym
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from kan import KAN
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from buffer import ReplayBuffer
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

POLICY_UPDATE_FREQUENCY = 2
EXPLORATION_NOISE = 0.1

# Define the Actor class, this will return the action to take
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(np.array(env.observation_space.shape[0]).prod(), 256, device=self.device)
        self.fc2 = nn.Linear(256, 256, device=self.device)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape[0]), device=self.device)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

# Define the Q_MLP class, this will return the Q value
class Q_MLP(nn.Module):
    def __init__(self, observation_space, action_space, width=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(observation_space + action_space, width, device=self.device)
        # self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(width, 1, device=self.device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def kan_train(
    net,
    target,
    actor,
    target_actor,
    data,
    optimizer,
    actor_optimizer,
    gamma=0.99,
    episode=0,
    lamb=0.0,
    lamb_l1=1.0,
    lamb_entropy=2.0,
    lamb_coef=0.0,
    lamb_coefdiff=0.0,
    small_mag_threshold=1e-16,
    small_reg_factor=1.0,
):
    def reg(acts_scale):
        def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = 0.0
        for i in range(len(acts_scale)):
            vec = acts_scale[i].reshape(
                -1,
            )

            p = vec / torch.sum(vec)
            l1 = torch.sum(nonlinear(vec))
            entropy = -torch.sum(p * torch.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(net.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(net.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(
                torch.mean(torch.abs(torch.diff(net.act_fun[i].coef)), dim=1)
            )
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    observations, actions, next_observations, rewards, terminations = data

    with torch.no_grad():
        next_q_values = net(next_observations)
        next_actions = next_q_values.argmax(dim=1)
        next_q_values_target = target(next_observations)
        target_max = next_q_values_target[range(len(next_q_values)), next_actions]
        td_target = rewards.flatten() + gamma * target_max * (
            1 - terminations.flatten()
        )

    # We emulate the gather function by doing for loops
    old_val = torch.zeros(len(observations))
    obs = net(observations)
    dict_actions = {}
    for i in range(len(observations)):
        # if the action is not in the dictionary, we add it with the corresponding value in the network
        if actions[i] not in dict_actions:
            dict_actions[actions[i]] = obs[i]
        else:
            # if the action is in the dictionary, we sum the values
            dict_actions[actions[i]] += obs[i]
    # We convert the dictionary to a tensor
    list_actions = list(dict_actions.values())
    # In order to have the same shape as td_target, we sum the values along the first dimension
    list_actions = [torch.sum(action) for action in list_actions]
    old_val = torch.stack(list_actions)
    # Now we squeeze to remove single dimensions
    #old_val = old_val.squeeze()
    loss = nn.functional.mse_loss(td_target, old_val)
    reg_ = reg(net.acts_scale)
    loss = loss + lamb * reg_
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def mlp_train(
    net,
    target,
    actor,
    target_actor,
    data,
    optimizer,
    actor_optimizer,
    gamma=0.99,
    episode=0
):
    
    observations, actions, next_observations, rewards, terminations = data # data is already a sample from the buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observations = observations.to(device)
    actions = actions.to(device)
    next_observations = next_observations.to(device)
    rewards = rewards.to(device)
    terminations = terminations.to(device)
    
    with torch.no_grad():
        next_state_actions = actor(next_observations)
        qf1_next_target = target(next_observations, next_state_actions)
        next_q_value = rewards.flatten() + (1 - terminations.flatten()) * gamma * (qf1_next_target).view(-1)
        
    old_val = net(observations, actions).view(-1)
    loss = F.mse_loss(old_val, next_q_value)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if episode % POLICY_UPDATE_FREQUENCY == 0:
        actor_loss = -net(observations, actor(observations)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            aux1 = 0.01 * param.data
            aux2 = (1 - 0.01) * target_param.data
            target_param.data.copy_(aux1 + aux2)
        for param, target_param in zip(net.parameters(), target.parameters()):
            aux1 = 0.01 * param.data
            aux2 = (1 - 0.01) * target_param.data
            target_param.data.copy_(aux1 + aux2)
        
    return loss.item()


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(config.seed)
    env = gym.make(config.env_id)
    if config.method == "KAN":
        # If env has discrete action space, then we leave action_space.n as it is
        # If it is not, we change it to action_space.shape[0]
        if env.action_space.dtype == int:
            q_network = KAN(
                width=[env.observation_space.shape[0], config.width, env.action_space.n],
                grid=config.grid,
                k=3,
                bias_trainable=False,
                sp_trainable=False,
                sb_trainable=False,
            )
            target_network = KAN(
                width=[env.observation_space.shape[0], config.width, env.action_space.n],
                grid=config.grid,
                k=3,
                bias_trainable=False,
                sp_trainable=False,
                sb_trainable=False,
            )
            actor = Actor(env).to(device)
            target_actor = Actor(env).to(device)
        else:
            q_network = KAN(
                width=[env.observation_space.shape[0], config.width, env.action_space.shape[0]],
                grid=config.grid,
                k=3,
                bias_trainable=False,
                sp_trainable=False,
                sb_trainable=False,
            )
            target_network = KAN(
                width=[env.observation_space.shape[0], config.width, env.action_space.shape[0]],
                grid=config.grid,
                k=3,
                bias_trainable=False,
                sp_trainable=False,
                sb_trainable=False,
            )
            actor = Actor(env).to(device)
            target_actor = Actor(env).to(device)
        train = kan_train
    elif config.method == "MLP":
        # If env has discrete action space, then we leave action_space.n as it is
        # If it is not, we change it to action_space.shape[0]
        if env.action_space.dtype == int:
            q_network = Q_MLP(env.observation_space.shape[0], env.action_space.n, config.width).to(device)
            target_network = Q_MLP(env.observation_space.shape[0], env.action_space.n, config.width).to(device)
            actor = Actor(env).to(device)
            target_actor = Actor(env).to(device)
        else:
            q_network = Q_MLP(env.observation_space.shape[0], env.action_space.shape[0], config.width).to(device)
            target_network = Q_MLP(env.observation_space.shape[0], env.action_space.shape[0], config.width).to(device)
            actor = Actor(env).to(device)
            target_actor = Actor(env).to(device)
        train = mlp_train
    else:
        raise Exception(
            f"Method {config.method} don't exist, choose between MLP and KAN."
        )

    target_network.load_state_dict(q_network.state_dict())
    target_actor.load_state_dict(actor.state_dict())

    run_name = f"{config.method}_{config.env_id}_{config.seed}_{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{run_name}.csv", "w") as f:
        f.write("episode,length\n")

    optimizer = torch.optim.Adam(q_network.parameters(), config.learning_rate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.learning_rate)

    buffer = ReplayBuffer(capacity=config.replay_buffer_capacity,
                          observation_dim=env.observation_space.shape[0],
                          action_dim=env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0])

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    pbar_position = 0 if HydraConfig.get().mode == HydraConfig.get().mode.RUN else HydraConfig.get().job.num

    for episode in tqdm(range(config.n_episodes), desc=f"{run_name}", position=pbar_position):
        observation, info = env.reset()
        observation = torch.from_numpy(observation)
        finished = False
        episode_length = 0
        while not finished:
            if episode < config.warm_up_episodes:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(observation).to(device))
                    actions += torch.normal(0, actor.action_scale * EXPLORATION_NOISE)
                    action = actions.cpu().numpy()
                
            next_observation, reward, terminated, truncated, info = env.step(action)
            if config.env_id == "CartPole-v1":
                reward = -1 if terminated else 0
            next_observation = torch.from_numpy(next_observation)

            # If the action space is not discrete, we need to convert the action to a tensor
            if env.action_space.dtype != int:
                action = torch.from_numpy(action).float()

            buffer.add(observation, action, next_observation, reward, terminated)

            observation = next_observation
            finished = terminated or truncated
            episode_length += 1
        with open(f"results/{run_name}.csv", "a") as f:
            f.write(f"{episode},{episode_length}\n")
        if len(buffer) >= config.batch_size:
            for _ in range(config.train_steps):
                loss = train(
                    net=q_network,
                    target=target_network,
                    actor=actor,
                    target_actor=target_actor,
                    data=buffer.sample(config.batch_size),
                    optimizer=optimizer,
                    actor_optimizer=actor_optimizer,
                    gamma=config.gamma,
                    episode=episode
                )
            writer.add_scalar("episode_length", episode_length, episode)
            writer.add_scalar("loss", loss, episode)
            writer.add_scalar("reward", reward, episode)
            if (
                episode % 25 == 0
                and config.method == "KAN"
                and episode < int(config.n_episodes * (1 / 2))
            ):
                q_network.update_grid_from_samples(buffer.observations[: len(buffer)])
                target_network.update_grid_from_samples(
                    buffer.observations[: len(buffer)]
                )

            if episode % config.target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())


if __name__ == "__main__":
    main()
