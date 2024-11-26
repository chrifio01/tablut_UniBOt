"""
replay_memory.py

This module defines the ReplayMemory class, which manages the replay buffer for 
storing and sampling experience tuples in reinforcement learning.

The replay memory facilitates the training of agents by storing environment 
transitions (state, action, reward, next state) in a replay buffer. These 
transitions can then be sampled to train reinforcement learning models like 
Deep Q-Networks (DQN).

Classes:
    ReplayMemory: A wrapper around TF-Agents' replay buffer for collecting 
                  and storing transitions during training.

Dependencies:
    - tf_agents.replay_buffers.TFUniformReplayBuffer
    - tf_agents.environments.PyEnvironment
    - tf_agents.trajectories.trajectory
    - tf_agents.agents.dqn.DqnAgent
"""

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import trajectory
from tf_agents.agents.dqn.dqn_agent import DqnAgent

class ReplayMemory:
    """
    A class to manage replay memory for reinforcement learning using TF-Agents.

    This class encapsulates the functionality of a replay buffer, enabling 
    the storage and retrieval of experience tuples for training deep reinforcement 
    learning agents.

    Attributes:
        _agent (DqnAgent): The reinforcement learning agent.
        _environment (PyEnvironment): The environment used to collect experiences.
        _memory_capacity (int): Maximum number of transitions the buffer can store.
        _buffer (TFUniformReplayBuffer): The underlying replay buffer for storing experiences.
    """

    def __init__(self, agent: DqnAgent, environment: PyEnvironment, memory_capacity: int):
        """
        Initializes the replay memory with a specified capacity.

        Args:
            agent (DqnAgent): The agent whose experiences will be stored.
            environment (PyEnvironment): The environment used for collecting experiences.
            memory_capacity (int): The maximum number of transitions to store in the buffer.
        """
        self._agent = agent
        self._environment = environment
        self._memory_capacity = memory_capacity
        
        self._buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=self._environment.batch_size,
            max_length=self._memory_capacity
        )

    def collect_step(self, policy):
        """
        Collects a single transition step using the provided policy and stores it in the buffer.

        This method retrieves the current time step from the environment, selects an action 
        using the given policy, applies the action to the environment, and records the resulting 
        transition (time step, action, next time step) as a trajectory in the replay buffer.

        Args:
            policy (tf_policy.TFPolicy): The policy used to determine the agent's actions.
        """
        time_step = self._environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self._environment.step(action_step.action)
        
        # Ensure the trajectory matches the expected batch size
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        self._buffer.add_batch(traj)
