# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

from networkx import grid_graph
import numpy as np

from . import adversarial
from . import register

class DatasetEnv(adversarial.AdversarialEnv):
  def __init__(self,
               size=15,
               agent_view_size=5,
               max_steps=250,
               see_through_walls=True,
               seed=None,
               deterministic_start_state=False,
               agent_start_dir=0,
               levels=None,
               labels=None,
               next_level_on_reset='random'):
    """Initialize a multigrid environment designed to load levels
    from a dataset. Each episode will be a different level from
    the dataset, sampled randomly. Optionally, a list of labels
    can be provided to act as level identifiers.

    Args:
        levels (numpy.array): A numpy array of shape
          (n_levels, height, width, channels) containing the levels
          to load.
        labels (list): A list of labels to use for each level.
          If None, the levels will be assigned labels from 0
          to n_levels-1.
        next_level_on_reset (str): The behavior when the environment
            is reset. If 'random', a random level is chosen. If
            'sequential', the next level is chosen in a cyclic manner.
    """
    self.levels = levels
    if levels is not None:
      self.labels = labels if labels is not None else [i for i in range(len(levels))]
    else:
      self.labels = None
    self.previous_level_label = None
    self.current_level_label = None
    self.next_level_on_reset = next_level_on_reset
    super().__init__(
      size=size,
      agent_view_size=agent_view_size,
      max_steps=max_steps,
      see_through_walls=see_through_walls,
      seed=seed,
      deterministic_start_state=deterministic_start_state,
      agent_start_dir=agent_start_dir)

  def load_levels(self, levels, labels=None, next_level_on_reset=None):
    if levels is None:
      return
    self.levels = levels
    self.labels = labels if labels is not None else [i for i in range(len(levels))]
    self.previous_level_label = None
    self.current_level_label = None
    if next_level_on_reset is not None:
      self.next_level_on_reset = next_level_on_reset

  def reset(self):
    """Fully resets the environment to random level."""
    self.graph = grid_graph(dim=[self.width-2, self.height-2])
    self.wall_locs = []

    self.step_count = 0

    if not self.deterministic_start_state:
      self.agent_start_dir = self._rand_int(0, 4)

    # Current position and direction of the agent
    self.reset_agent_status()
    self.agent_start_pos = None
    self.goal_pos = None
    self.done = False

    if self.levels is not None:
      if self.next_level_on_reset == 'random':
        return self.reset_to_level(level=None)
      elif self.next_level_on_reset == 'sequential':
        if self.current_level_label is not None:
          current_idx = self.labels.index(self.current_level_label)
          next_idx = (current_idx + 1) % len(self.levels)
        else:
          next_idx = 0
        return self.reset_to_level(level=self.labels[next_idx])
    else: # Create empty grid if no levels have been loaded
      self._gen_grid(self.width, self.height)
      self.place_one_agent(0, rand_dir=False)
      self.agent_start_pos = self.agent_pos[0]
      return self.reset_agent()

  def reset_to_level(self, level=None):
    # Level is identified by the corresponding label in the dataset.
    # If no level is provided, a random level is chosen.
    assert level is None or isinstance(level, int)
    if level is None:
      level_id = np.random.randint(0, len(self.levels))
    else:
      level_id = self.labels.index(level)

    level = self.levels[level_id]
    self.previous_level_label = self.current_level_label
    self.current_level_label = self.labels[level_id]

    return self.reset_to_encoding(level)

  def mutate_level(self, num_edits=1):
    raise NotImplementedError

  def step_adversary(self, loc):
    raise NotImplementedError

  def generate_random_z(self):
    raise NotImplementedError

  def reset_random(self):
    raise NotImplementedError

class DatasetS45Env(DatasetEnv):
    def __init__(self, seed=None):
        super().__init__(size=45, agent_view_size=5, seed=seed, max_steps=2250)

class DatasetDeterministicEnv(DatasetEnv):
  def __init__(self, seed=None, agent_start_dir=0):
    super().__init__(seed=seed, max_steps=250,
                     deterministic_start_state=True, agent_start_dir=agent_start_dir)

class DatasetDeterministicS45Env(DatasetEnv):
  def __init__(self, seed=None, agent_start_dir=0):
    super().__init__(size=45, agent_view_size=5, seed=seed, max_steps=2250,
                     deterministic_start_state=True, agent_start_dir=agent_start_dir)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-DatasetEnv-v0',
    entry_point=module_path + ':DatasetEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-DatasetS45Env-v0',
    entry_point=module_path + ':DatasetS45Env',
    max_episode_steps=2250,
)

register.register(
    env_id='MultiGrid-DatasetDeterministicEnv-v0',
    entry_point=module_path + ':DatasetDeterministicEnv',
    max_episode_steps=250,
)

register.register(
    env_id='MultiGrid-DatasetDeterministicS45Env-v0',
    entry_point=module_path + ':DatasetDeterministicS45Env',
    max_episode_steps=2250,
)
