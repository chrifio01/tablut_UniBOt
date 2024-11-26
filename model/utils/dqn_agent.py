"""
This module defines the `DQNAgent` class for implementing a Deep Q-Network (DQN) agent
using the `tf-agents` library. The agent interacts with an environment, collects
trajectories, and stores them in a replay buffer for training purposes.

Classes:
--------
DQNAgent:
    Encapsulates the logic for a DQN-based reinforcement learning agent, including
    interaction with the environment, policy evaluation, and replay buffer management.

Dependencies:
-------------
- TensorFlow (`tensorflow`)
- TensorFlow Agents (`tf_agents`)
- PyYAML (`yaml`)
"""

import tensorflow as tf
from tf_agents.environments import TFPyEnvironment
from tf_agents.agents import DqnAgent
from tf_agents.networks.q_network import QNetwork

class DQNAgent:
    """
    A class to represent a Deep Q-Network (DQN) agent for reinforcement learning.

    This class provides methods for initializing the agent, collecting trajectories 
    by interacting with the environment, and storing these trajectories in a replay 
    buffer for future training. It utilizes the `tf-agents` library for implementing 
    the agent's policies and environment interactions.

    Attributes
    ----------
    env : TFPyEnvironment
        The TensorFlow environment in which the agent operates.
    agent : DqnAgent
        The underlying DQN agent from the `tf-agents` library.
    replay_buffer : ReplayBuffer
        The replay buffer for storing collected trajectories.

    Methods
    -------
    __init__(tf_env, q_network, optimizer, *, epsilon_fn, target_update_period=2000, 
             td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"), gamma=0.99, 
             train_step_counter=tf.Variable(0)):
        Initializes the DQNAgent with the specified environment, network, and training parameters.

    collect_trajectory(replay_buffer, num_episodes=1000):
        Collects trajectories by interacting with the environment and stores them 
        in the provided replay buffer.
    """

    def __init__(
        self, 
        tf_env: TFPyEnvironment, 
        q_network: QNetwork, 
        optimizer: tf.compat.v1.train.Optimizer,
        *,
        epsilon_fn: callable,
        target_update_period: int = 2000,
        td_errors_loss_fn: tf.keras.losses.Loss = tf.keras.losses.Huber(reduction="none"),
        gamma: float = 0.99,
        train_step_counter: tf.Variable = tf.Variable(0)
        ):
        """
        Initializes the DQNAgent with the given environment.

        Parameters
        ----------
        env : TFPyEnvironment
            The environment in which the agent will operate.
        """

        self.env = tf_env

        self.agent = DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=q_network,
            optimizer=optimizer,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            train_step_counter=train_step_counter,
            epsilon_greedy=lambda: epsilon_fn(train_step_counter)
        )

        self.agent.initialize()
