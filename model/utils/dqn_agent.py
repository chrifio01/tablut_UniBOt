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

import os
import zipfile
import shutil
import tensorflow as tf
from tf_agents.environments import TFPyEnvironment
from tf_agents.agents import DqnAgent
from tf_agents.networks.q_network import QNetwork

class DQNAgent:
    """
    A class to represent a Deep Q-Network (DQN) agent for reinforcement learning.
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
        train_step_counter: tf.Variable = tf.Variable(0),
        from_pretrained: str = None  # Pass the zipped checkpoint path if loading pretrained
        ):
        """
        Initializes the DQNAgent with the given environment.

        Parameters
        ----------
        tf_env : TFPyEnvironment
            The TensorFlow environment in which the agent operates.
        q_network : QNetwork
            The Q-network to be used by the agent.
        optimizer : tf.compat.v1.train.Optimizer
            Optimizer for training the agent.
        epsilon_fn : callable
            A function to calculate epsilon for epsilon-greedy policy.
        from_pretrained : str, optional
            Path to a zipped checkpoint file to load a pretrained agent.
        """
        self.env = tf_env

        if from_pretrained:
            # Load pretrained agent from the zipped checkpoint
            self.agent = self._load(from_pretrained, tf_env, q_network, optimizer, epsilon_fn, 
                                    target_update_period, td_errors_loss_fn, gamma, train_step_counter)
        else:
            # Create a new agent
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

    def _load(self, checkpoints_dir_path: str, tf_env, q_network, optimizer, epsilon_fn, 
              target_update_period, td_errors_loss_fn, gamma, train_step_counter) -> DqnAgent:
        """
        Loads a pretrained DQN agent from a zipped checkpoint file.

        Parameters
        ----------
        checkpoints_dir_path : str
            The path to the checkpoint dir.

        Returns
        -------
        DqnAgent
            The loaded pretrained DQN agent.
        """
        # Initialize a temporary agent to restore the state
        temp_agent = DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_network,
            optimizer=optimizer,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            train_step_counter=train_step_counter,
            epsilon_greedy=lambda: epsilon_fn(train_step_counter)
        )
        temp_agent.initialize()

        # Restore the agent from the checkpoint
        checkpoint = tf.train.Checkpoint(agent=temp_agent)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir_path)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint).expect_partial()
        else:
            raise ValueError("No checkpoint found in the provided directory %s.", checkpoints_dir_path)

        return temp_agent
