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
from tf_agents.trajectories import TimeStep
from tf_agents.environments import TFPyEnvironment
from tf_agents.agents import DqnAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories import trajectory
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

    def collect_trajectory(self, replay_buffer: ReplayBuffer, num_episodes=1000):
        """
        Initializes the DQNAgent with the given environment, Q-network, and optimization settings.

        Parameters
        ----------
        tf_env : TFPyEnvironment
            The environment in which the agent will operate. This environment must be compatible 
            with TensorFlow and follow the `tf-agents` environment specifications.
        q_network : QNetwork
            The Q-network to approximate the Q-value function. This network maps observations 
            to predicted Q-values for each action.
        optimizer : tf.compat.v1.train.Optimizer
            The optimizer used to train the Q-network.
        epsilon_fn : callable
            A function that returns the epsilon value for the epsilon-greedy policy, 
            allowing exploration during training.
        target_update_period : int, optional
            The number of steps before updating the target network (default is 2000).
        td_errors_loss_fn : tf.keras.losses.Loss, optional
            The loss function for TD-error minimization (default is Huber loss with "none" reduction).
        gamma : float, optional
            The discount factor for future rewards (default is 0.99).
        train_step_counter : tf.Variable, optional
            A counter to track training steps (default is a new `tf.Variable` initialized to 0).
        """
        for _ in range(num_episodes):
            time_step: TimeStep = self.env.reset()  # Reset the environment

            while not time_step.is_last():
                # The agent collects an action using its policy
                action_step = self.agent.collect_policy.action(time_step)
                next_time_step = self.env.step(action_step.action)  # Environment executes the action

                # Create the transition
                traj = trajectory.from_transition(time_step, action_step, next_time_step)

                # Add the transition to the replay buffer
                replay_buffer.add_batch(traj)

                # Update the time_step for the next loop
                time_step = next_time_step
