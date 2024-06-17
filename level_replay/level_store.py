# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# Following modifications by Samuel Garcin:
# - store latent representations, labels of level parameters
# - encode_bytes(), decode_bytes() methods
# - minor changes and fixes

from collections import namedtuple, defaultdict, deque

import numpy as np
import torch


INT32_MAX = 2147483647


class LevelStore(object):
	"""
	Manages a mapping between level index --> level, where the level
	may be represented by any arbitrary data structure. Typically, we can 
	represent any given level as a string.
	"""
	def __init__(self, max_size=None, data_info={}):
		self.max_size = max_size
		self.seed2level = defaultdict()
		self.level2seed = defaultdict()
		self.seed2parent = defaultdict()
		self.seed2label = defaultdict()
		self.label2seed = defaultdict(list)
		self.seed2datasetlabel = defaultdict()
		self.datasetlabel2seed = defaultdict()
		self.seed2latent = defaultdict()
		self.next_seed = 1
		self.levels = set()

		self.data_info = data_info

	def __len__(self):
		return len(self.levels)

	def _insert(self, level, latent=None, parent_seed=None, label=None, from_dataset=False):
		if level is None:
			return None

		if level not in self.levels:
			# FIFO if max size constraint
			if self.max_size is not None:
				while len(self.levels) >= self.max_size:
					first_idx = list(self.seed2level)[0]
					self._remove(first_idx)

			seed = self.next_seed
			self.seed2level[seed] = level
			if latent is not None:
				self.seed2latent[seed] = latent
			if parent_seed is not None:
				if isinstance(parent_seed, int):
					self.seed2parent[seed] = \
						self.seed2parent[parent_seed] + [self.seed2level[parent_seed]]
				elif hasattr(parent_seed, '__iter__'):
					self.seed2parent[seed] = [self.seed2level[ps] for ps in parent_seed]
			else:
				self.seed2parent[seed] = []
			if label is not None:
				self.seed2label[seed] = label
				self.label2seed[label].append(seed)
			if from_dataset:
				self.seed2datasetlabel[seed] = label
				if label is not None:
					self.datasetlabel2seed[label] = seed
			self.level2seed[level] = seed
			self.levels.add(level)
			self.next_seed += 1
			return seed
		else:
			return self.level2seed[level]

	def insert(self, level, latents=None, parent_seeds=None, labels=None, from_dataset=None):
		if hasattr(level, '__iter__'):
			idx = []
			for i, l in enumerate(level):
				z = None
				ps = None
				lbl = None
				if latents is not None:
					z = latents[i]
				if parent_seeds is not None:
					ps = parent_seeds[i]
				if labels is not None:
					lbl = labels[i]
					if isinstance(lbl, torch.Tensor):
						lbl = lbl.item()
				da = from_dataset[i] if from_dataset is not None else False
				idx.append(self._insert(l, latent=z, parent_seed=ps, label=lbl, from_dataset=da))
			return idx
		else:
			return self._insert(level, latent=latents, parent_seed=parent_seeds, label=labels, from_dataset=from_dataset)

	def _remove(self, level_seed):
		if level_seed is None or level_seed < 0:
			return

		level = self.seed2level[level_seed]
		label = self.seed2label.get(level_seed, None)
		self.levels.remove(level)
		del self.seed2level[level_seed]
		del self.level2seed[level]
		del self.seed2parent[level_seed]
		if level_seed in self.seed2latent:
			del self.seed2latent[level_seed]
		if level_seed in self.seed2datasetlabel:
			del self.seed2datasetlabel[level_seed]
			if label is not None:
				del self.datasetlabel2seed[label]
		if label is not None:
			del self.seed2label[level_seed]
			self.label2seed[label].remove(level_seed)
			if len(self.label2seed[label]) == 0:
				del self.label2seed[label]

	def remove(self, level_seed):
		if hasattr(level_seed, '__iter__'):
			for i in level_seed:
				self._remove(i)
		else:
			self._remove(level_seed)

	def reconcile_seeds(self, level_seeds):
		old_seeds = set(self.seed2level)
		new_seeds = set(level_seeds)

		# Don't update if empty seeds
		if len(new_seeds) == 1 and -1 in new_seeds:
			return

		ejected_seeds = old_seeds - new_seeds
		for seed in ejected_seeds:
			self._remove(seed)

	def get_level(self, level_seed):
		level = self.seed2level[level_seed]

		if self.data_info:
			if self.data_info.get('numpy', False):
				dtype = self.data_info['dtype']
				shape = self.data_info['shape']
				level = np.frombuffer(level, dtype=dtype).reshape(*shape)

		return level

	def get_label(self, level_seed):
		return self.seed2label[level_seed]

	def get_latent_distribution_parameter(self, level_seed):
		return self.seed2latent[level_seed]

	def get_latent_distribution_parameter_batch(self, level_seeds):
		mean, std = [], []
		for seed in level_seeds:
			m, s = self.get_latent_distribution_parameter(seed)
			mean.append(m)
			std.append(s)
		return torch.stack(mean), torch.stack(std)

	def encode_bytes(self, level, data_info=None):

		#This is to resolve an issue in which different agent start dirs are being stored as different levels,
		# whereas under the current minigrid logic the agent is systematically assigned a random starting direction.
		temp = level.copy()
		temp[..., -1] = 0

		if data_info is None:
			if self.data_info:
				data_info = self.data_info
			else:
				raise ValueError('No data_info provided to encode bytes')

		dtype = data_info['dtype']
		level_bytes = temp.astype(dtype).tobytes()

		return level_bytes

	def decode_bytes(self, level_bytes, data_info=None):

		assert isinstance(level_bytes, bytes), f"Input must be bytes. Got {type(level_bytes)}"

		if data_info is None:
			if self.data_info:
				data_info = self.data_info
			else:
				raise ValueError('No data_info provided to decode bytes')

		dtype = data_info['dtype']
		shape = data_info['shape']
		level = np.frombuffer(level_bytes, dtype=dtype).reshape(*shape)

		return level
