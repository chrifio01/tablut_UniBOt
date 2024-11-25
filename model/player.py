"""
    Our model module.
"""

import os

from tf_agents.trajectories import time_step as ts

from environment import Environment
from model.utils.dqn_agent import DQNAgent
from shared.consts import INITIAL_STATE
from shared.history import History
from shared.random_player import RandomPlayer
from shared.utils import strp_state, Color, State, Action
from .utils.replay_memory import ReplayMemory
from environment.utils import state_to_tensor, ActionDecoder

from .utils.dqn_agent import DQNAgent
from shared.loggers import logger, training_logger
from tf_agents.policies import random_tf_policy
import tensorflow as tf
from .utils.dqn_network import DQN
from shared.utils import parse_yaml, AbstractPlayer

_config_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.yaml"
)

_hyperparams_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "hyperparams.yaml"
)

CONFIG = parse_yaml(_config_file_path)
HYPER_PARAMS = parse_yaml(_hyperparams_file_path)


class DQNPlayer(AbstractPlayer):
    
    def __init__(self, color: Color, *, training_mode: bool = False):
        super().__init__()
        if training_mode:
            logger.disabled = True
        self._name = "DQNPlayer"
        self._color = color
        
        self._current_state = strp_state(INITIAL_STATE)
        history = History(matches={})
        trainer = self
        opponent = RandomPlayer(color=Color.BLACK)
        observation_spec_shape = CONFIG['env']['observation_spec']["shape"]
        action_spec_shape = CONFIG['env']['action_spec']["shape"]
        discount_factor = HYPER_PARAMS['env']['discount_factor']

        self._env = Environment(
            current_state=self._current_state,
            history=history,
            trainer=trainer,
            observation_spec_shape=observation_spec_shape,
            action_spec_shape=action_spec_shape,
            discount_factor=discount_factor,
            opponent=opponent,
        )
        self._env = self._env.to_TFPy()
        
        shape = (CONFIG["model"]["dqn"]["neurons_per_layer"], CONFIG["model"]["dqn"]["hidden_layers_num"])
        
        self._q_network = DQN(
            input_tensor_spec=self._env.observation_spec(),
            action_spec=self._env.action_spec(),
            shape=shape
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=HYPER_PARAMS["training"]["optimizer"]["learning_rate"])
        
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=HYPER_PARAMS["training"]["epsilon_fn"]["initial_learning_rate"],
            decay_steps=HYPER_PARAMS["training"]["epsilon_fn"]["decay_steps"],
            end_learning_rate=HYPER_PARAMS["training"]["epsilon_fn"]["end_learning_rate"]
        )

        self._agent = DQNAgent(
            self._env,
            self._q_network,
            optimizer,
            epsilon_fn=epsilon_fn,
            gamma=HYPER_PARAMS["training"]["gamma"],
            target_update_period=HYPER_PARAMS["training"]["target_update_period"],
            )
        
        self._replay_buffer = ReplayMemory(
            self._agent.agent,
            self._env,
            memory_capacity=CONFIG["replay_buffer"]["capacity"],
            batch_size=CONFIG["replay_buffer"]["batch_size"],
            )
    
    def fit(self, state: State, *args, **kwargs) -> Action:
        """
        Predict the best action for a given state using the DQN.

        This method processes the input game state, assigns it to the environment's 
        current time step, and uses the policy to predict the best action.

        Args:
            state (State): The current game state.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Action: The action chosen by the DQN based on predicted Q-values.
        """
        assert self._color is not None, "Player color must be set before calling fit."

        # Convert the state into a tensor suitable for the model
        observation = state_to_tensor(state, self._color)

        # Ensure observation has a batch dimension
        batched_observation = tf.expand_dims(observation, axis=0)  # Add batch dimension

        # Use the agent's policy to predict the action from the current environment state
        action_step = self._agent.agent.policy.action(
            ts.restart(batched_observation)  # Wrap in TimeStep with batch dimension
        )

        # Decode the selected action into a format usable by the environment
        action = ActionDecoder.decode(action_step.action.numpy()[0], state)

        return action
    
    def train(self):
        training_logger.debug("Initializing for training...")
        num_iterations = HYPER_PARAMS["training"]["iterations"]
        collect_steps_per_iteration = HYPER_PARAMS["training"]["collect_steps_per_iteration"]
        log_interval = HYPER_PARAMS["training"]["log_interval"]

        # Collect initial data
        training_logger.debug("Collecting initial data...")
        for _ in range(HYPER_PARAMS["training"]["initial_dataset_size"]):
            self._replay_buffer.collect_step(
                random_tf_policy.RandomTFPolicy(
                    self._env.time_step_spec(), self._env.action_spec()
                )
            )
        training_logger.debug("Initial data collection complete.")

        # Create dataset
        training_logger.debug("Creating replay buffer dataset...")
        dataset = self._replay_buffer._buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=CONFIG["replay_buffer"]["batch_size"],
            num_steps=2
        ).prefetch(3)     

        iterator = iter(dataset)
        training_logger.debug("Dataset created successfully.")

        # Train the agent
        training_logger.debug("Starting training...")
        for i in range(num_iterations):
            training_logger.debug(f"Iteration {i}...")
            for _ in range(collect_steps_per_iteration):
                self._replay_buffer.collect_step(self._agent.agent.collect_policy)

            # Sample experience and log shapes
            experience, _ = next(iterator)

            # Ensure tensors are float32 before training
            train_loss = self._agent.agent.train(experience).loss
            step = self._agent.agent.train_step_counter.numpy()

            if step % log_interval == 0:
                training_logger.debug(f"Step {step}: Loss = {train_loss}")
            
    def test(self, num_episodes = 1000):
        """
        Tests the DQN agent in the given environment.

        Parameters
        ----------
        agent : DQNAgent
            The DQN agent to be tested.
        env : tf_py_environment.TFPyEnvironment
            The environment in which to test the agent.
        num_episodes : int, optional
            The number of episodes to test the agent for (default is 100).

        Returns
        -------
        list
            A list of total rewards for each episode.
        """
        total_rewards = []

        for _ in range(num_episodes):
            time_step = self._env.reset()
            episode_reward = 0

            while not time_step.is_last():
                action_step = self._agent.agent.policy.action(time_step)
                time_step = self._env.step(action_step.action)
                episode_reward += time_step.reward.numpy()

            total_rewards.append(episode_reward)

        return total_rewards
