import pprint
import math
import gym
import torch
import random
import os
import yaml
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from agent import Agent
from actor import *
from kan import KAN
from fasterkan.fasterkan import FasterKAN, FasterKANvolver
from tqdm import tqdm
from continuous_cartpole import *

def run_episode(
        episode_index, 
        env, 
        agent, 
        do_training=True, 
        deterministic=True, 
        rendering=True, 
        random_probability=[]
        ):
    
    observation, info = env.reset()
    observation = torch.from_numpy(observation)
    
    loss_value, actor_loss_value = None, None
    episode_length = 0
    net_reward = 0

    while True:
        if do_training and episode_index < agent.config.warm_up_episodes:
            action = env.action_space.sample()
        else:
            action = agent.act(observation, deterministic, random_probability)

        next_observation, reward, terminated, truncated, info = env.step(action)

        if env.action_space.dtype != int:
            action = torch.from_numpy(action).float()
        else:
            action = torch.from_numpy(action)
            action.apply_(lambda x: int(x>0.5))

        next_observation = torch.from_numpy(next_observation)
        agent.buffer.add(observation, action, next_observation, reward, terminated)

        net_reward += reward

        if do_training and episode_index >= agent.config.warm_up_episodes:
            loss_value, actor_loss_value = agent.train(episode_index)
            
        observation = next_observation

        if terminated or truncated:
            break

        episode_length += 1

        if rendering:
            env.render()
    
    if math.remainder(episode_index+1, 10) == 0 or not do_training:
        print("Reward:", net_reward)
        print("Loss:", loss_value)
        print("Actor Loss:", actor_loss_value)
        print("Episode length:", episode_length)

    with open(f"{agent.config.results_dir}/{agent.run_name}.csv", "a") as f:
        f.write(f"{episode_index},{episode_length},{net_reward},{loss_value}\n")

    if do_training and loss_value is not None:
        agent.writer.add_scalar("episode_length", episode_length, episode_index)
        agent.writer.add_scalar("loss", loss_value, episode_index)
        agent.writer.add_scalar("reward", net_reward, episode_index)

        if agent.current_loss != 9999:
            agent.previous_loss = agent.current_loss
        agent.current_loss = loss_value
        agent.check_early_stopping()

        if (
            math.remainder(episode_index+1, agent.config.grid_update_frequency) == 0
            and agent.config.method == "KAN"
            and episode_index < int(agent.config.train_n_episodes * (1 / 2))
        ):
            combined_input = torch.cat([agent.buffer.observations[: len(agent.buffer)], agent.buffer.actions[: len(agent.buffer)]], axis=1).to(agent.device)
            agent.q_network.update_grid_from_samples(combined_input)
            agent.target_network.update_grid_from_samples(combined_input)

    return agent

def train(env, agent, deterministic= True, do_training=True, rendering=True):
    agent.reset_for_train()
    for episode in tqdm(range(agent.config.train_n_episodes), desc=f"{agent.run_name}"):
        if agent.terminate_training: break
        agent = run_episode(episode, env, agent, deterministic=deterministic, do_training=do_training, rendering=rendering)
        if math.remainder(episode+1, 10) == 0:
            # print("Saving model...")
            torch.save(agent.target_actor.state_dict(), f"{agent.config.models_dir}/{agent.run_name}.pt")

    if not agent.terminate_training:
        torch.save(agent.target_actor.state_dict(), f"{agent.config.models_dir}/{agent.run_name}.pt")
        # agent.target_actor.save_ckpt(f"{agent.config.env_id}_{agent.config.method}.pt")
    if agent.config.method == "KAN":
        agent.target_network.plot()
        plt.savefig(f"{agent.config.plots_dir}/{agent.run_name}.png")
    
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = dotdict(config)
        pprint.pprint(config)
        
    if config.device == "mps" and torch.backends.mps.is_available():
        device_name = "mps"
    elif config.device == "cuda" and torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    print(f"Running on device: {device_name}")
    device = torch.device(device_name)

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    set_all_seeds(config.seed)

    print(f"Running environment - {config.env_id}")
    if config.env_id in ["ContinuousCartPoleEnv"]:
        env = ContinuousCartPoleEnv()
    else:
        env = gym.make(config.env_id, render_mode="rgb_array")

    if env.action_space.dtype == int:
        network_in = env.observation_space.shape[0] + env.action_space.n
    else:
        network_in = env.observation_space.shape[0] + env.action_space.shape[0]
    print(f"Observation space: {env.observation_space}\nAction space: {env.action_space}\nNetwork in: {network_in}")
    
    if config.method == "KAN":
        q_network = KAN(
			width=[network_in, config.width, 1],
			grid=config.grid,
			k=3,
			bias_trainable=False,
			sp_trainable=False,
			sb_trainable=False,
            device=device_name
		)
        target_network = KAN(
			width=[network_in, config.width, 1],
			grid=config.grid,
			k=3,
			bias_trainable=False,
			sp_trainable=False,
			sb_trainable=False,
            device=device_name
		)
    elif config.method == "FasterKAN":
        q_network = FasterKAN(
			layers_hidden=[network_in, config.width, 1],
			num_grids=config.grid,
			exponent=3,
            train_grid = True
			# bias_trainable=False,
			# sp_trainable=False,
			# sb_trainable=False,
            # device=device_name
		).to(device)
        target_network = FasterKAN(
			layers_hidden=[network_in, config.width, 1],
			num_grids=config.grid,
			exponent=3,
            train_grid = True
			# bias_trainable=False,
			# sp_trainable=False,
			# sb_trainable=False,
            # device=device_name
		).to(device)
    elif config.method == "MLP":
        q_network = nn.Sequential(
            nn.Linear(network_in, config.width, device=device),
            nn.ReLU(),
            nn.Linear(config.width, 1, device=device),
        )
        target_network = nn.Sequential(
            nn.Linear(network_in, config.width, device=device),
            nn.ReLU(),
            nn.Linear(config.width, 1, device=device),
        )
    else:
        raise Exception(
            f"Method {config.method} don't exist."
        )

    if config.train:
        print("Training initiated...")
        actor = Policy_MLP(env, device)
        target_actor = Policy_MLP(env, device)
        agent = Agent(env, q_network, target_network, actor, target_actor, device, config)
        train(env, agent, deterministic=False, do_training=True, rendering=False)

    if config.test:
        print("Test initiated...")
        actor = Policy_MLP(env, device)
        target_actor = Policy_MLP(env, device)
        agent2 = Agent(env, q_network, target_network, actor, target_actor, device, config)
        agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/{agent.run_name}.pt"))
        # agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/MLP_CartPole-v1_0_1720385960.pt"))

        for i in range(config.test_n_episodes):
            run_episode(i, env, agent2, deterministic=True, do_training=False, rendering=False)

if __name__ == "__main__":
    main()