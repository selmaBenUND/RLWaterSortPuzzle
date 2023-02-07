import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tf_agents.specs import tensor_spec
from typing import Any, List, Sequence, Tuple
from tf_agents.environments import py_environment

import game_water_utils as cpu

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      env: py_environment.PyEnvironment, 
      all_layers_params: list,
      layer_params: list,
      layer_params2: list):
    """Initialize."""
    super().__init__()

    self.common1 = cpu.dense_layer(all_layers_params[0])
    self.common2 = cpu.dense_layer(all_layers_params[1])
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    layer_params[0] = num_actions
    self.actor = cpu.dense_layer(layer_params)
    self.critic = cpu.dense_layer(layer_params2)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    y = self.common1(inputs)
    x = self.common2(y)
    return self.actor(x), self.critic(x)