"""
import os

from shared.history import History
from shared.utils import strp_state, Board, Color
from shared.consts import INITIAL_STATE
from shared.random_player import RandomPlayer
from environment.tablut import Environment
import numpy as np
from tf_agents.environments.utils import validate_py_environment
if __name__ == '__main__':
    try:
        PLAYER_COLOR = os.environ['PLAYER_COLOR']
        TIMEOUT = os.environ['TIMEOUT']
        SERVER_IP = os.environ['SERVER_IP']
        WEBSOCKET_PORT = os.environ['WEBSOCKET_PORT']

        settings = {
            'current_state': INITIAL_STATE,
            'timeout': int(TIMEOUT),
            'server_ip': SERVER_IP,
            'port': int(WEBSOCKET_PORT)
        }
        player = RandomPlayer(color=strp_color(PLAYER_COLOR))
        client = Client(player=player, settings=settings)

        client.main()
    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == '__main__':

    current_state = strp_state(INITIAL_STATE)
    history = History(matches={})
    trainer = RandomPlayer(color=Color.WHITE)
    opponent = RandomPlayer(color=Color.BLACK)
    observation_spec_shape = (333, )
    action_spec_shape = (25, 16)
    discount_factor = 0.99

    env = Environment(
        current_state=current_state,
        history=history,
        trainer=trainer,
        observation_spec_shape=observation_spec_shape,
        action_spec_shape=action_spec_shape,
        discount_factor=discount_factor,
        opponent=opponent,
    )

    validate_py_environment(env, episodes=5)

"""
from tf_agents.environments import tf_py_environment, validate_py_environment

from environment import Environment
from model.utils.dqn_agent import DQNAgent
from shared.consts import INITIAL_STATE
from shared.history import History
from shared.random_player import RandomPlayer
from shared.utils import strp_state, Color


def test_agent(agent, env, num_episodes=100):
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

    for episode in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0

        while not time_step.is_last():
            action_step = agent.agent.policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_reward += time_step.reward.numpy()

        total_rewards.append(episode_reward)

    return total_rewards

current_state = strp_state(INITIAL_STATE)
history = History(matches={})
trainer = RandomPlayer(color=Color.WHITE)
opponent = RandomPlayer(color=Color.BLACK)
observation_spec_shape = (333, )
action_spec_shape = (400, )
discount_factor = 0.99

env = Environment(
    current_state=current_state,
    history=history,
    trainer=trainer,
    observation_spec_shape=observation_spec_shape,
    action_spec_shape=action_spec_shape,
    discount_factor=discount_factor,
    opponent=opponent,
)
tf_env = tf_py_environment.TFPyEnvironment(env)
agent = DQNAgent(tf_env)
rewards = test_agent(agent, tf_env)
print(f"Average reward over {len(rewards)} episodes: {sum(rewards) / len(rewards)}")