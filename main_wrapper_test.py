from continuous_cartpole import ContinuousCartPoleEnv
from wrapper import TransformObservation
# import tensorboard
import tensorflow as tf
import numpy as np
import time


# Create the environment
env = ContinuousCartPoleEnv()

env.seed(42)

# Create the wrapper
#env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.randn(*obs.shape))

# Set the number of episodes to train
num_episodes = 100

# Set the maximum number of steps per episode
max_steps = 500

timestamp = time.strftime("%Y%m%d-%H%M%S")

# Create a summary writer for Tensorboard
writer = tf.summary.create_file_writer("logs/cartpole" + timestamp)

# Train the agent
for episode in range(num_episodes):
    # Reset the environment for each episode
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Render the environment
        env.render()

        # Choose an action
        action = env.action_space.sample()

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Update the total reward
        total_reward += reward

        # Update the state
        state = next_state

        # Check if the episode is done
        if done:
            break

    # Print the total reward for the episode
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Write the total reward to Tensorboard
    with writer.as_default():
        tf.summary.scalar("Total Reward", total_reward, step=episode)
    

# Close the environment
env.close()


def gaussian_noise(observation, noise_scale=0.1):
    return observation + noise_scale * np.random.randn(*observation.shape)