import pprint
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
from tqdm import tqdm

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
    
    loss_value = None
    episode_length = 0
    net_reward = 0

    while True:
        if do_training and episode_index < agent.config.warm_up_episodes:
            action = env.action_space.sample()
        else:
            action = agent.act(observation, deterministic, random_probability)
        
        next_observation, reward, terminated, truncated, info = env.step(action)
        net_reward += reward
        
        next_observation = torch.from_numpy(next_observation)

        if env.action_space.dtype != int:
            action = torch.from_numpy(action).float()

        if do_training:
            loss_value = agent.train(observation, action, next_observation, reward, terminated)
            
        observation = next_observation

        if terminated or truncated:
            break

        episode_length += 1

        if rendering:
            env.render()

    print("Reward:", net_reward)
    with open(f"results/{agent.run_name}.csv", "a") as f:
        f.write(f"{episode_index},{episode_length},{net_reward}\n")

    if do_training and loss_value is not None:
        agent.writer.add_scalar("episode_length", episode_length, episode_index)
        agent.writer.add_scalar("loss", loss_value, episode_index)
        agent.writer.add_scalar("reward", net_reward, episode_index)

        if (
            episode_index % 5 == 0
            and agent.config.method == "KAN"
            and episode_index < int(agent.config.train_n_episodes * (1 / 2))
        ):
            combined_input = torch.cat([agent.buffer.observations[: len(agent.buffer)], agent.buffer.actions[: len(agent.buffer)]], axis=1)
            agent.q_network.update_grid_from_samples(combined_input)
            agent.target_network.update_grid_from_samples(combined_input)

        if episode_index+1 % agent.config.policy_update_frequency == 0:
            agent.soft_update()

    return agent

def train(env, agent, deterministic= True, do_training=True, rendering=True):
    for episode in tqdm(range(agent.config.train_n_episodes), desc=f"{agent.run_name}"):
        agent = run_episode(episode, env, agent, deterministic=deterministic, do_training=do_training, rendering=rendering)
        if episode+1 % 10 == 0:
            torch.save(agent.target_actor.state_dict(), f"{agent.config.env_id}_{agent.config.method}.pt")

    torch.save(agent.target_actor.state_dict(), f"{agent.run_name}.pt")
    # agent.target_actor.save_ckpt(f"{agent.config.env_id}_{agent.config.method}.pt")
    if agent.config.method == "KAN":
        agent.target_network.plot()
        plt.savefig(f"{agent.run_name}.png")
    
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


# @hydra.main(config_path=".", config_name="config", version_base=None)
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

    set_all_seeds(config.seed)
    env = gym.make(config.env_id, render_mode='human')

    network_in = env.observation_space.shape[0] + env.action_space.shape[0]
    
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
    elif config.method == "MLP":
        q_network = nn.Sequential(
            nn.Linear(network_in, config.width),
            nn.ReLU(),
            nn.Linear(config.width, 1),
        )
        target_network = nn.Sequential(
            nn.Linear(network_in, config.width),
            nn.ReLU(),
            nn.Linear(config.width, 1),
        )
    else:
        raise Exception(
            f"Method {config.method} don't exist, choose between MLP and KAN."
        )
    
    actor = Policy_MLP(env, device)
    target_actor = Policy_MLP(env, device)
    agent = Agent(env, q_network, target_network, actor, target_actor, device, config)

    if config.train:
        train(env, agent, deterministic=False, do_training=True, rendering=True)

    if config.test:
        # agent.target_actor.load_state_dict(torch.load(f"{agent.run_name}.pt"))
        agent.target_actor.load_state_dict(torch.load(f"KAN_Ant-v4_0_1720045570.pt"))

        for i in range(config.test_n_episodes):
            run_episode(i, env, agent, deterministic=True, do_training=False, rendering=True)

if __name__ == "__main__":
    main()