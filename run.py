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
from tqdm import tqdm
from continuous_cartpole import *
from wrapper import TransformObservation
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from morphing_agents.mujoco.ant.env import MorphingAntEnv


def run_episode(
        episode_index, 
        env, 
        agent, 
        do_training=True, 
        deterministic=True, 
        rendering=True, 
        random_probability=[],
        log=True
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
        agent.check_early_stopping()

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

    return agent

def train(env, agent, deterministic= True, do_training=True, rendering=True):
    agent.reset_for_train()
    for episode in tqdm(range(agent.config.train_n_episodes), desc=f"{agent.run_name}"):
        if agent.terminate_training: break
        agent = run_episode(episode, env, agent, deterministic=deterministic, do_training=do_training, rendering=rendering)
        if math.remainder(episode+1, 10) == 0:
            # print("Saving model...")
            torch.save(agent.target_actor.state_dict(), f"{agent.config.models_dir}/{agent.run_name}.pt")
        if math.remainder(episode+1, 100) == 0:
            print("Testing model...")
            run_episode(episode, env, agent, deterministic=True, do_training=False, rendering=False, log=False)

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

def obervation_noise_concat(observation, number_of_dimensions, noise_level=0.1):
    """Generates a noise dimension for each of number_of_dimensions and concatenates them to the original observation.

    Args:
        observation: original obsevation from the env.step() function
        number_of_dimensions (int): number of dimensions to add to the observation
        noise_level (float, optional): noise level. Defaults to 0.1.
    """
    #print("Adding noise to observation:", observation)
    
    array_of_noise = np.random.normal(0, noise_level, number_of_dimensions)
    #print("Noise added:", array_of_noise)
    return np.concatenate([observation, array_of_noise])

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
        env = MorphingAntEnv(num_legs=config.num_legs, expose_design=False, centered=False)
    else:
        env = gym.make(config.env_id, render_mode="rgb_array", terminate_when_unhealthy=True)
    
    # The lambda function will apply the observation_noise_concat function to the observation
    dimension_wrapper_number = config.dimension_wrapper_number
    env = TransformObservation(env, lambda obs: obervation_noise_concat(obs, dimension_wrapper_number, noise_level=0.1))

    if env.action_space.dtype == int:
        network_in = (env.observation_space.shape[0] + dimension_wrapper_number) + env.action_space.n
    else:
        network_in = (env.observation_space.shape[0] + dimension_wrapper_number) + env.action_space.shape[0]
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
        actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        target_actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        agent = Agent(env, q_network, target_network, actor, target_actor, device, config, dimension_wrapper_number=dimension_wrapper_number)
        train(env, agent, deterministic=False, do_training=True, rendering=False)

    if config.test:
        print("Test initiated...")
        actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        target_actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        agent2 = Agent(env, q_network, target_network, actor, target_actor, device, config, dimension_wrapper_number=dimension_wrapper_number)
        agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/{agent.run_name}.pt"))
        # agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/EfficientKAN_Ant-v4_2_1721655487.pt"))

        for i in range(config.test_n_episodes):
            run_episode(i, env, agent2, deterministic=True, do_training=False, rendering=False)

if __name__ == "__main__":
    main()