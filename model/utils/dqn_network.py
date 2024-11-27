"""
This module defines a custom Q-network for DQN (Deep Q-Network) in reinforcement learning. 
It extends the `QNetwork` class from the `tf-agents` library, allowing for greater 
flexibility in the design of the neural network. The `DqnNetwork` class is used to 
approximate the Q-value function in a DQN agent, which is an essential component 
in the reinforcement learning pipeline.

The class provides several parameters for customizing the network architecture, 
including the number of layers, the activation functions, and the preprocessing 
layers for input tensors.

Dependencies:
- TensorFlow (`tensorflow`)
- tf-agents (`tf_agents`)
"""

from typing import Tuple

import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.networks.q_network import QNetwork


def DQN(
    input_tensor_spec: array_spec,
    action_spec: array_spec,
    shape: Tuple[int, int],
    *,
    preprocessing_layers=None,
    preprocessing_combiner=None,
    conv_layer_params=None,
    dropout_layer_params=None, 
    activation_fn=tf.keras.activations.relu,
    kernel_initializer=None,
    batch_squash=True,
    dtype=tf.float32,
    q_layer_activation_fn: str = 'linear',
    name='DQNNetwork'
    ):
    """
    Initializes the DqnNetwork instance with the specified architecture 
    and parameters.

    Args are detailed in the class docstring.
    """
    return QNetwork(
        input_tensor_spec, 
        action_spec, 
        preprocessing_layers, 
        preprocessing_combiner, 
        conv_layer_params, 
        shape, 
        dropout_layer_params, 
        activation_fn, 
        kernel_initializer, 
        batch_squash, 
        dtype, 
        q_layer_activation_fn, 
        name
    )
