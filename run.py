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
from efficient_kan.kan import KAN as EfficientKAN
from fasterkan.fasterkan import FasterKAN, FasterKANvolver
from fastkan.fastkan import FastKAN
from tqdm import tqdm
from continuous_cartpole import *
from morphing_agents.mujoco.ant.env import MorphingAntEnv


def run_episode(
        episode_index, 
        env, 
        agent, 
        do_training=True, 
        deterministic=True, 
        rendering=False,
        log=True
        ):
    
    observation, info = env.reset()
    observation = torch.from_numpy(observation)
    
    loss_value, actor_loss_value = None, None
    episode_length = 0
    net_reward = 0

    first_steps = 0

    while True:
        if do_training and episode_index < agent.config.warm_up_episodes:
            action = env.action_space.sample()
        elif not do_training and first_steps< 5:
            # print(f"first steps random {first_steps}")
            action = env.action_space.sample()
            first_steps += 1
        else:
            action = agent.act(observation, deterministic, noise_std=0.2)

        if do_training and episode_index >= agent.config.warm_up_episodes:
            if agent.config.noise_mean!=0 or agent.config.noise_std!=0:
                noise = np.random.normal(agent.config.noise_mean, agent.config.noise_std, action.shape)
                action = action + noise
                action = np.clip(action, agent.env.action_space.low, agent.env.action_space.high)
            
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

    if log:
        with open(f"{agent.config.results_dir}/{agent.run_name}.csv", "a") as f:
            f.write(f"{episode_index},{episode_length},{net_reward},{loss_value}\n")

    if do_training and loss_value is not None:
        agent.writer.add_scalar("episode_length", episode_length, episode_index)
        agent.writer.add_scalar("loss", loss_value, episode_index)
        agent.writer.add_scalar("reward", net_reward, episode_index)

        if agent.current_loss != 9999:
            agent.previous_loss = agent.current_loss
        agent.current_loss = loss_value
        # agent.check_early_stopping()

        if (
            math.remainder(episode_index+1, agent.config.grid_update_frequency) == 0
            and episode_index < int(agent.config.train_n_episodes * (1 / 2))
        ):
            if agent.config.method == "KAN":
                combined_input = torch.cat([agent.buffer.observations[: len(agent.buffer)], agent.buffer.actions[: len(agent.buffer)]], axis=1).to(agent.device)
                agent.q_network.update_grid_from_samples(combined_input)
                agent.target_network.update_grid_from_samples(combined_input)
            elif agent.config.method == "EfficientKAN":
                combined_input = torch.cat([agent.buffer.observations[: len(agent.buffer)], agent.buffer.actions[: len(agent.buffer)]], axis=1).to(agent.device)
                agent.q_network(combined_input, update_grid=True)
                agent.target_network(combined_input, update_grid=True)

    return agent, net_reward

def train(env, agent, deterministic= True, do_training=True, rendering=False):
    agent.reset_for_train()
    for episode in tqdm(range(agent.config.train_n_episodes), desc=f"{agent.run_name}"):
        if agent.terminate_training: break

        agent, _ = run_episode(episode, env, agent, deterministic=deterministic, do_training=do_training, rendering=rendering)

        if math.remainder(episode+1, 10) == 0:
            agent.save_model()
        if math.remainder(episode+1, 50) == 0:
            print("\nTesting model...")
            test(env, agent, 5, deterministic=True, rendering=rendering, log=False)
            agent.save_model(f"_{episode}")

    if not agent.terminate_training:
        agent.save_model()

    if agent.config.method in ["KAN", "EfficientKAN"]:
        agent.target_network.plot()
        plt.savefig(f"{agent.config.plots_dir}/{agent.run_name}.png")

def test(env, agent, episodes, deterministic= True, rendering=False, log=True):
    rewards = []
    for i in range(episodes):
        _, reward = run_episode(i, env, agent, deterministic=deterministic, do_training=False, rendering=rendering, log=log)
        rewards.append(reward)
    average_reward = sum(rewards) / max(1, len(rewards))
    print(f"Average reward over {episodes} episodes: {average_reward}")
    
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
    elif config.env_id in ["MorphingAnt"]:
        env = MorphingAntEnv(num_legs=config.num_legs, 
                             expose_design=False, 
                             render_flag=config.render
                             )
    elif config.env_id in ["Ant-v4"]:
        if config.render:
            render_mode = "human"
        else:
            render_mode = None
        env = gym.make(config.env_id, 
                    #    healthy_reward=0.01,
                       render_mode=render_mode)
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
    elif config.method == "EfficientKAN":
        q_network = EfficientKAN(
			layers_hidden=[network_in, config.width, 1],
			grid_size=config.grid,
			spline_order=3,
			# bias_trainable=False,
			# sp_trainable=False,
			# sb_trainable=False,
            # device=device_name
		).to(device)
        target_network = EfficientKAN(
			layers_hidden=[network_in, config.width, 1],
			grid_size=config.grid,
			spline_order=3,
			# bias_trainable=False,
			# sp_trainable=False,
			# sb_trainable=False,
            # device=device_name
		).to(device)
    elif config.method == "FastKAN":
        q_network = FastKAN(
			layers_hidden=[network_in, config.width, 1],
			num_grids=config.grid,
			# exponent=3,
            # train_grid = True
			# bias_trainable=False,
			# sp_trainable=False,
			# sb_trainable=False,
            # device=device_name
		).to(device)
        target_network = FastKAN(
			layers_hidden=[network_in, config.width, 1],
			num_grids=config.grid,
			# exponent=3,
            # train_grid = True
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
        train(env, agent, deterministic=False, do_training=True, rendering=config.render)

    if config.test:
        print("Test initiated...")
        actor = Policy_MLP(env, device)
        target_actor = Policy_MLP(env, device)
        agent2 = Agent(env, q_network, target_network, actor, target_actor, device, config)
        agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/{agent.run_name}_actor.pt"))
        # agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/MLP_MorphingAnt_0_1722121326_actor.pt"))

        test(env, agent2, config.test_n_episodes, deterministic=True, rendering=config.render, log=True)

if __name__ == "__main__":
    main()
