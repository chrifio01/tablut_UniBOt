"""
This module defines the DQNAgent class for training a DQN (Deep Q-Network) agent
using the tf-agents library. The agent interacts with the environment, collects
trajectories, and stores them in a replay buffer for training.

Dependencies:
- TensorFlow (`tensorflow`)
- tf-agents (`tf_agents`)
- PyYAML (`yaml`)
"""

from tf_agents.environments import tf_py_environment, TFPyEnvironment
from tf_agents.agents import DqnAgent
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
import tensorflow as tf
from model.utils.dqn_network import DQN
import yaml

# Load the configuration from the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract the DQN configuration
hidden_layers_num = config['model']['dqn']['hidden_layers_num']
neurons_per_layer = config['model']['dqn']['neurons_per_layer']


class DQNAgent:
    """
    A class to represent a DQN agent.

    Attributes
    ----------
    env : tf_py_environment.TFPyEnvironment
        The TensorFlow environment in which the agent operates.
    agent : DqnAgent
        The DQN agent.
    replay_buffer : TFUniformReplayBuffer
        The replay buffer to store trajectories.

    Methods
    -------
    __init__(env: Environment):
        Initializes the DQNAgent with the given environment.
    collect_trajectory(num_episodes=1000):
        Collects trajectories by interacting with the environment.
    """

    def __init__(self, tf_env: TFPyEnvironment):
        """
        Initializes the DQNAgent with the given environment.

        Parameters
        ----------
        env : TFPyEnvironment
            The environment in which the agent will operate.
        """

        self.env = tf_env

        q_network = DQN(
            input_tensor_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
            shape=(neurons_per_layer,) * hidden_layers_num
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        train_step = tf.Variable(0)
        update_period = 4
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0,
            decay_steps=250000 // update_period,
            end_learning_rate=0.01)

        self.agent = DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=q_network,
            optimizer=optimizer,
            target_update_period=2000,
            td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
            gamma=0.99,
            train_step_counter=train_step, epsilon_greedy=lambda: epsilon_fn(train_step)
        )

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=1000000
        )

        self.agent.initialize()

    def collect_trajectory(self, num_episodes=1000):
        """
        Collects trajectories by interacting with the environment.

        Parameters
        ----------
        num_episodes : int, optional
            The number of episodes to collect trajectories for (default is 1000).
        """
        for episode in range(num_episodes):
            time_step = self.env.reset()  # Reset the environment

            while not time_step.is_last():
                # The agent collects an action using its policy
                action_step = self.agent.collect_policy.action(time_step)
                next_time_step = self.env.step(action_step.action)  # Environment executes the action

                # Create the transition
                traj = trajectory.from_transition(time_step, action_step, next_time_step)

                # Add the transition to the replay buffer
                self.replay_buffer.add_batch(traj)

                # Update the time_step for the next loop
                time_step = next_time_step