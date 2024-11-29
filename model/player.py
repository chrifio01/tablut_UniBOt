"""
    Our model module.
"""

import os

import tensorflow as tf
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import random_tf_policy

from environment import Environment, state_to_tensor, ActionDecoder
from shared import INITIAL_STATE, History, RandomPlayer, strp_state, Color, State, Action, parse_yaml, AbstractPlayer, logger, training_logger, env_logger, MoveChecker
from shared.consts import WIN_REWARD, INVALID_ACTION_PUNISHMENT, LOSS_REWARD, DRAW_REWARD
from shared.heuristic import heuristic
from .utils import ReplayMemory, DQNAgent, DQN
from environment.utils import TablutCustomTFPolicy


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
    """
    A reinforcement learning agent for playing Tablut using the Deep Q-Network (DQN) algorithm.

    The `DQNPlayer` class integrates various components required to train, evaluate, 
    and test a DQN agent within the Tablut game environment. It supports training 
    from scratch, evaluating policy performance, and loading pretrained agents from checkpoints. 
    Experience replay and checkpointing mechanisms enable effective training and model management.

    Attributes:
    ----------
    _name : str
        The name of the player ("DQNPlayer").
    _color : Color
        The color of the player (either `Color.WHITE` or `Color.BLACK`).
    _current_state : State
        The current state of the game, initialized to the starting state.
    _env : Environment
        The Tablut environment encapsulating game logic, interactions, and agent dynamics.
    _q_network : DQN
        The Q-network that predicts Q-values for state-action pairs.
    _agent : DQNAgent
        The DQN agent responsible for training, evaluation, and policy management.
    _replay_buffer : ReplayMemory
        The replay memory for storing and sampling experiences during training.

    Methods:
    -------
    __init__(color, training_mode=False, disable_env_logger=False, from_pretrained=None):
        Initializes the DQNPlayer with its environment, Q-network, and DQN agent. 
        Supports training from scratch or loading pretrained models from a checkpoint.

    fit(state):
        Predicts the optimal action for a given game state using the trained DQN.

    evaluate(num_episodes):
        Evaluates the agent's policy over a specified number of episodes, 
        returning the average reward and other performance metrics.

    train():
        Trains the DQN agent using experience replay and periodic policy updates.
        The training progress is checkpointed for later restoration.

    test(num_episodes):
        Tests the trained agent in the environment for a given number of episodes, 
        measuring its performance and logging the outcomes.

    Loading Pretrained Agents:
    -------------------------
    The `DQNPlayer` class can load a pretrained DQN agent from a zipped checkpoint file
    by passing the `from_pretrained` argument when initializing the player. The agent's 
    state, including the Q-network and policy, is fully restored for inference or further training.

    Example Usage:
    --------------
    Training from Scratch:
        ```python
        player = DQNPlayer(color=Color.WHITE, training_mode=True)
        player.train()  # Train the agent
        action = player.fit(current_state)  # Predict the best action for the current state
        ```

    Loading a Pretrained Agent:
        ```python
        pretrained_path = "/path/to/pretrained_agent.zip"
        player = DQNPlayer(color=Color.BLACK, from_pretrained=pretrained_path)
        action = player.fit(current_state)  # Use the pretrained agent to predict the best action
        ```
    
    Evaluating Policy Performance:
        ```python
        average_reward = player.evaluate(num_episodes=10)
        print(f"Average Reward: {average_reward}")
        ```
    """
    
    def __init__(self, color: Color, *, training_mode: bool = False, disable_env_logger = False, from_pretrained: str = None):
        super().__init__()
        if training_mode:
            logger.disabled = True
        env_logger.disabled = disable_env_logger
        
        self._name = "DQNPlayer"
        self._color = color
        
        self._current_state = strp_state(INITIAL_STATE)
        history = History(matches={})
        trainer = self
        if self.color:
            opponent = RandomPlayer(color=Color.BLACK if self.color == Color.WHITE else Color.WHITE)
        else:
            opponent = RandomPlayer(color=None)
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
        
        shape = CONFIG["model"]["dqn"]["shape"]
        
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

        if from_pretrained:
            self._agent = DQNAgent(
                self._env,
                self._q_network,
                optimizer,
                epsilon_fn=epsilon_fn,
                from_pretrained=from_pretrained
            )
        
            logger.debug("Agent successfully loaded from pretrained model %s", from_pretrained)
        else:
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
            memory_capacity=CONFIG["replay_buffer"]["capacity"]
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
        try:
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
            MoveChecker.is_valid_move(state, action)
            return action
        except Exception as e:
            logger.error("Error during calculating move: %s", e)
            poss_moves = list(MoveChecker.gen_possible_moves(state))
            values =[]
            for move in poss_moves:
                values.append(heuristic(state, move))
            best_val = max(values)
            best_move = poss_moves[values.index(best_val)]
            return best_move
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent's performance by running it in the environment for a fixed number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to run for evaluation.

        Returns:
            float: The average reward obtained during evaluation.
        """
        training_logger.debug("Starting policy evaluation...")

        total_reward = 0.0
        for episode in range(num_episodes):
            time_step = self._env.reset()
            episode_reward = 0.0

            while not time_step.is_last():
                action_step = self._agent.agent.policy.action(time_step)
                time_step = self._env.step(action_step.action)
                episode_reward += time_step.reward.numpy()

            training_logger.debug("Episode %d: Reward = %f", episode + 1, episode_reward)
            total_reward += episode_reward

        # Compute the average reward and convert it to a scalar
        average_reward = total_reward / num_episodes
        training_logger.debug("Policy evaluation complete. Average Reward = %.2f", float(average_reward))
        return float(average_reward)
    
    def _save_and_plot_losses(self, train_losses: List[float], checkpoint_dir: str):
        # Save losses to a CSV file
        losses_df = pd.DataFrame(train_losses, columns=["Loss"])
        losses_csv_path = os.path.join(checkpoint_dir, "training_losses.csv")
        losses_df.to_csv(losses_csv_path, index=False)
        training_logger.info("Training losses saved to %s", losses_csv_path)
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Iterations")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(checkpoint_dir, "training_loss_plot.png")
        plt.savefig(plot_path)
        training_logger.info("Training loss plot saved to %s", plot_path)
        plt.show()
    
    def train(self):
        """
        Train the agent.
        """
        # Init training parameters
        training_logger.debug("Initializing for training...")
        num_iterations = HYPER_PARAMS["training"]["iterations"]
        collect_steps_per_iteration = HYPER_PARAMS["training"]["collect_steps_per_iteration"]
        log_interval = CONFIG["training"]["log_interval"]
        eval_interval = CONFIG["training"]["eval_interval"]
        initial_dataset_size = HYPER_PARAMS["training"]["initial_dataset_size"]
        sample_batch_size = CONFIG["replay_buffer"]["sample_batch_size"]
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(_config_file_path)),
            CONFIG["training"]["checkpoint_dir"]
        )
        
        train_losses = []
        
        # Ensure the checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Setup checkpointing
        checkpoint = tf.train.Checkpoint(agent=self._agent.agent)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        # Collect initial data
        training_logger.debug("Collecting initial data...")
        for _ in range(initial_dataset_size):
            self._replay_buffer.collect_step(
                TablutCustomTFPolicy(
                    self._env.time_step_spec(), self._env.action_spec()
                )
            )
        training_logger.debug("Initial data collection complete.")

        # Create dataset
        training_logger.debug("Creating replay buffer dataset...")
        dataset = self._replay_buffer._buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=sample_batch_size,
            num_steps=2
        ).prefetch(3)     

        iterator = iter(dataset)
        training_logger.debug("Dataset created successfully.")

        # Train the agent
        training_logger.debug("Starting training...")
        for i in range(num_iterations):
            training_logger.debug("Iteration %d...", i)
            for _ in range(collect_steps_per_iteration):
                self._replay_buffer.collect_step(self._agent.agent.collect_policy)

            # Sample experience and log shapes
            experience, _ = next(iterator)

            # Ensure tensors are float32 before training
            train_loss = self._agent.agent.train(experience).loss
            step = self._agent.agent.train_step_counter.numpy()
            train_losses.append(train_loss)

            if step % log_interval == 0:
                training_logger.debug("Step %d: Loss = %f", step, train_loss)
                
            # Save checkpoints
            if step % eval_interval == 0:
                checkpoint_manager.save()
                training_logger.debug("Checkpoint saved at step %d.", step)
                
                # Evaluate the policy
                average_reward = self.evaluate()
                training_logger.info("Evaluation at step %d: Average Reward = %.2f", step, average_reward)
                
        training_logger.debug("Training completed.")
        self._save_and_plot_losses(train_losses, checkpoint_dir)
        
    def _extract_statistics(self, rewards: List[float]):
        rewards_wout_end = [r for r in rewards if r not in (WIN_REWARD, LOSS_REWARD, DRAW_REWARD, INVALID_ACTION_PUNISHMENT)]
            
        statistics = {
            "win": rewards.count(WIN_REWARD),
            "draw": rewards.count(DRAW_REWARD),
            "lose": rewards.count(LOSS_REWARD),
            "invalid_action": rewards.count(INVALID_ACTION_PUNISHMENT),
            "average": sum(rewards_wout_end) / len(rewards_wout_end)
        }
        logger.debug("Statistics: %s", statistics)
        return statistics
        
            
    def test(self, num_episodes = 100):
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
        rewards = []

        for _ in range(num_episodes):
            time_step = self._env.reset()
            # episode_reward = 0

            while not time_step.is_last():
                action_step = self._agent.agent.policy.action(time_step)
                time_step = self._env.step(action_step.action)
                episode_reward = time_step.reward.numpy()

                rewards.append(episode_reward[0])
                
        return self._extract_statistics(rewards)
