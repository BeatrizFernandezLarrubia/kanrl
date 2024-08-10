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
import pandas as pd
from agent import Agent
from actor import *
from kan import KAN
from efficient_kan.kan import KAN as EfficientKAN
from fasterkan.fasterkan import FasterKAN, FasterKANvolver
from fastkan.fastkan import FastKAN
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
        rendering=False,
        log=True
        ):
    """
    Run a single episode with the given configuration and return reward. Perform early stopping if necessary. Write the results to tensorboard. Update the networks periodically.
    """
    
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
            action = env.action_space.sample()
            first_steps += 1
        else:
            action = agent.act(observation, deterministic)

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

    return agent, net_reward

def train(env, agent, deterministic= True, do_training=True, rendering=False):
    """
    Run n episodes and train the network by periodically updating the networks in the agent.
    """
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
    """
    Run n episodes and return average reward and the reward for each episode.
    """
    rewards = []
    for i in range(episodes):
        _, reward = run_episode(i, env, agent, deterministic=deterministic, do_training=False, rendering=rendering, log=log)
        rewards.append(reward)
    average_reward = sum(rewards) / max(1, len(rewards))
    print(f"Average reward over {episodes} episodes: {average_reward}")
    return average_reward, rewards
    
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

def set_parameters_by_config(config):
    """
    Set the parameters and create objects required for the agent according to the config file.
    """
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
                             render_flag=config.render,
                             healthy_reward=config.healthy_reward
                             )
    elif config.env_id in ["Ant-v4"]:
        if config.render:
            render_mode = "human"
        else:
            render_mode = None
        env = gym.make(config.env_id, 
                       healthy_reward=config.healthy_reward,
                       render_mode=render_mode)
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
			spline_order=3
		).to(device)
        target_network = EfficientKAN(
			layers_hidden=[network_in, config.width, 1],
			grid_size=config.grid,
			spline_order=3
		).to(device)
    elif config.method == "FastKAN":
        q_network = FastKAN(
			layers_hidden=[network_in, config.width, 1],
			num_grids=config.grid
		).to(device)
        target_network = FastKAN(
			layers_hidden=[network_in, config.width, 1],
			num_grids=config.grid
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
    return {
        "device": device,
        "device_name": device_name,
        "q_network": q_network,
        "target_network": target_network,
        "dimension_wrapper_number": dimension_wrapper_number,
        "network_in": network_in,
        "env": env
    }

def main():
    """
    Read the config file and perform train and/or test operations.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = dotdict(config)
        
    parameters = set_parameters_by_config(config)
    device = parameters["device"]
    env = parameters["env"]
    q_network = parameters["q_network"]
    target_network = parameters["target_network"]
    dimension_wrapper_number = parameters["dimension_wrapper_number"]

    if config.train:
        print("Training initiated...")
        actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        target_actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        agent = Agent(env, q_network, target_network, actor, target_actor, device, config, dimension_wrapper_number=dimension_wrapper_number)
        train(env, agent, deterministic=False, do_training=True, rendering=config.render)

    if config.test:
        print("Test initiated...")
        actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        target_actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
        agent2 = Agent(env, q_network, target_network, actor, target_actor, device, config, dimension_wrapper_number=dimension_wrapper_number)
        agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/{agent.run_name}_actor.pt"))
        # agent2.actor.load_state_dict(torch.load(f"{config.models_dir}/EfficientKAN_Ant-v4_2_1721655487.pt"))

        test(env, agent2, config.test_n_episodes, deterministic=True, rendering=config.render, log=True)

def test_bulk(file_name, num_legs, es_epoch):
    """
    Test model over n episodes and calculate mean and std of the rewards.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = dotdict(config)
        config["num_legs"] = num_legs
        config["method"] = file_name.split("_")[0]
        pprint.pprint(config)
        
    parameters = set_parameters_by_config(config)
    device = parameters["device"]
    env = parameters["env"]
    q_network = parameters["q_network"]
    target_network = parameters["target_network"]
    dimension_wrapper_number = parameters["dimension_wrapper_number"]

    actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
    target_actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
    agent = Agent(env, q_network, target_network, actor, target_actor, device, config, dimension_wrapper_number=dimension_wrapper_number)

    if es_epoch!=0:
        agent.actor.load_state_dict(torch.load(f"{config.models_dir}/{file_name}_actor_{es_epoch}.pt", map_location=torch.device('cpu')))
    else:
        agent.actor.load_state_dict(torch.load(f"{config.models_dir}/{file_name}_actor.pt", map_location=torch.device('cpu')))

    _, rewards_list = test(env, agent, config.test_n_episodes, deterministic=True, rendering=config.render, log=True)
    rewards_list = np.array(rewards_list)

    mean_reward = np.mean(rewards_list)
    std_reward = np.std(rewards_list)

    return mean_reward, std_reward

def test_bulk_driver():
    """
    Driver function to test multiple models over n episodes and calculate mean and std of the rewards.
    """
    test_list = [
            ["EfficientKAN_MorphingAnt_0_1722319702", 3, 0.2, 999],
            ["EfficientKAN_MorphingAnt_0_1721917306", 4, 0.2, 0],
            ["EfficientKAN_MorphingAnt_0_1722203246", 5, 0.2, 749],
            ["EfficientKAN_MorphingAnt_0_1722204441", 6, 0.2, 999],
            ["EfficientKAN_MorphingAnt_0_1722203855", 7, 0.2, 749],
            ["MLP_MorphingAnt_0_1722342048", 3, 0.2, 999],
            ["MLP_MorphingAnt_0_1722121326", 4, 0.2, 999],
            ["MLP_MorphingAnt_0_1722154175", 5, 0.2, 799],
            ["MLP_MorphingAnt_0_1722247036", 6, 0.2, 499],
            ["MLP_MorphingAnt_0_1722391642", 7, 0.2, 999]
    ]
    # test_list = [
    #         ["EfficientKAN_MorphingAnt_0_1721726361", 4, 0.0, 699],
    #         ["EfficientKAN_MorphingAnt_0_1721857925", 4, 0.2, 0],
    #         ["EfficientKAN_MorphingAnt_0_1722028514", 4, 0.4, 749],
    #         ["MLP_MorphingAnt_0_1722296073", 4, 0.0, 199],
    #         ["MLP_MorphingAnt_0_1722319667", 4, 0.2, 999],
    #         ["MLP_MorphingAnt_0_1722365973", 4, 0.4, 449],
    #         # ["EfficientKAN_MorphingAnt_0_1721726361", 4, 0.0, 999],
    #         # ["EfficientKAN_MorphingAnt_0_1721857925", 4, 0.2, 99],
    #         # ["EfficientKAN_MorphingAnt_0_1722028514", 4, 0.4, 749],
    #         # ["MLP_MorphingAnt_0_1722296073", 4, 0.0, 999],
    #         # ["MLP_MorphingAnt_0_1722319667", 4, 0.2, 999],
    #         # ["MLP_MorphingAnt_0_1722365973", 4, 0.4, 999],
    #     ]
    df = pd.DataFrame(columns=["num_legs",
                               "method",
                               "noise",
                               "num_epochs",
                               "file_name",
                               "mean_reward",
                               "std_reward"])
    for meta in test_list:
        mean_reward, std_reward = test_bulk(
            meta[0], meta[1], meta[3]
        )
        reward_row = pd.DataFrame([{
            "num_legs": meta[1],
            "method": meta[0].split("_")[0],
            "noise": meta[2],
            "num_epochs": meta[3],
            "file_name": meta[0],
            "mean_reward": mean_reward,
            "std_reward":std_reward
        }])
        df = pd.concat([df, reward_row], ignore_index=True)
    print(df)
    df.to_csv("final_test/test.csv")     

def plot_kan(file_name):
    """
    Plot the KAN graphs of the model that help with visual inferences.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = dotdict(config)
        config["method"] = file_name.split("_")[0]
        pprint.pprint(config)
        
    parameters = set_parameters_by_config(config)
    device = parameters["device"]
    env = parameters["env"]
    q_network = parameters["q_network"]
    target_network = parameters["target_network"]
    dimension_wrapper_number = parameters["dimension_wrapper_number"]

    actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
    target_actor = Policy_MLP(env, device, dimension_wrapper_number=dimension_wrapper_number)
    agent = Agent(env, q_network, target_network, actor, target_actor, device, config, dimension_wrapper_number=dimension_wrapper_number)

    agent.actor.load_state_dict(torch.load(f"{config.models_dir}/{file_name}_actor.pt", map_location=torch.device('cpu')))
    agent.target_network.load_state_dict(torch.load(f"{config.models_dir}/{file_name}_network.pt", map_location=torch.device('cpu')))

    agent.target_network.plot(beta=0.5)
    plt.savefig(f"{agent.config.plots_dir}/{agent.run_name}_v2.png")

if __name__ == "__main__":
    main()
    # plot_kan("EfficientKAN_MorphingAnt_0_1721917306")
    # test_bulk_driver()
