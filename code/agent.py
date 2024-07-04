import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from buffer import ReplayBuffer
from kan import KAN

class Agent:
    def __init__(self, env, q_network, target_network, actor, target_actor, device, config):
        self.method = config.method
        self.env = env
        self.config = config
        self.device = device
    
        self.action_space_n = env.action_space.shape[0]

        self.q_network = q_network.to(device)
        self.target_network = target_network.to(device)
        self.actor = actor.to(device)
        self.target_actor = actor.to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.run_name = f"{config.method}_{config.env_id}_{config.seed}_{int(time.time())}"

        self.writer = SummaryWriter(f"runs/{self.run_name}")

        os.makedirs("results", exist_ok=True)
        with open(f"results/{self.run_name}.csv", "w") as f:
            f.write("episode,length,reward\n")

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), config.actor_learning_rate)
        
        self.buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0]
            )

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )

    def reg(self,
            acts_scale,
            gamma=0.99,
            lamb=0.0,
            lamb_l1=1.0,
            lamb_entropy=2.0,
            lamb_coef=0.0,
            lamb_coefdiff=0.0,
            small_mag_threshold=1e-16,
            small_reg_factor=1.0):
        def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
            return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

        reg_ = 0.0
        for i in range(len(acts_scale)):
            vec = acts_scale[i].reshape(-1,)
            p = vec / torch.sum(vec)
            l1 = torch.sum(nonlinear(vec))
            entropy = -torch.sum(p * torch.log2(p + 1e-4))
            reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(self.q_network.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(self.q_network.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(
                torch.mean(torch.abs(torch.diff(self.q_network.act_fun[i].coef)), dim=1)
            )
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1
        # loss = loss + lamb * reg_
        return lamb * reg_
    
    def soft_update(self):
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            aux1 = self.config.tau * param.data
            aux2 = (1 - self.config.tau) * target_param.data
            target_param.data.copy_(aux1 + aux2)
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            aux1 = self.config.tau * param.data
            aux2 = (1 - self.config.tau) * target_param.data
            target_param.data.copy_(aux1 + aux2)

    def train(self, observation, action, next_observation, reward, terminated):
        # print("Bufer comparison", len(self.buffer),self.config.batch_size)
        self.buffer.add(observation, action, next_observation, reward, terminated)
        for ts in range(self.config.train_steps):
            observations, actions, next_observations, rewards, terminations = self.buffer.sample(self.config.batch_size)
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            next_observations = next_observations.to(self.device)
            rewards = rewards.to(self.device)
            terminations = terminations.to(self.device)

            with torch.no_grad():
                next_state_actions = self.actor(next_observations)
                network_input = torch.cat([next_observations, next_state_actions], axis=1).to(self.device)
                q_next_target = self.target_network(network_input)
                next_q_values = rewards.flatten() + (1 - terminations.flatten()) * self.config.gamma * (q_next_target).view(-1)
            q_values = self.q_network(torch.cat([observations, actions], axis=1).to(self.device)).view(-1)

            loss = self.loss_function(q_values, next_q_values)
            
            if self.method == "KAN":
                loss = loss + self.reg(self.q_network.acts_scale)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            actor_loss = -self.q_network(torch.cat([observations, self.actor(observations)], axis=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if ts == self.config.train_steps-1:
                return loss.item()
            else:
                continue
    
    def act(self, observation, deterministic, random_probability=[]):
        r = np.random.uniform()
        if deterministic or r > self.config.exploration_epsilon:
            action_output = self.actor(
                observation.type(torch.float32).to(self.device).unsqueeze(0)
            ).detach().cpu().numpy().ravel()
        else:
            # Sample random action
            action_output = np.random.rand(self.action_space_n)

        return action_output

