from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import trajectory
from tf_agents.agents.dqn.dqn_agent import DqnAgent
    
class ReplayMemory:
    
    def __init__(self, agent: DqnAgent, environment: PyEnvironment, memory_capacity: int, batch_size: int):
        self._agent = agent
        self._environment = environment
        self._memory_capacity = memory_capacity
        self._buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=batch_size,
            max_length=self._memory_capacity
        )
        
    def collect_step(self, policy):
        time_step = self._environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self._environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        self._buffer.add_batch(traj)