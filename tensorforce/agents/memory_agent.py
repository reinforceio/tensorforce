# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Default implementation for using a replay memory.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from tensorforce.config import create_config
from tensorforce.replay_memories import ReplayMemory
from tensorforce.agents import RLAgent

from tensorforce.default_configs import MemoryAgentConfig


class MemoryAgent(RLAgent):

    name = 'MemoryAgent'

    default_config = MemoryAgentConfig

    model = None

    def __init__(self, config, scope='memory_agent', network_builder=None):
        """
        Initialize a vanilla DQN agent as described in
        http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html.

        :param config: Configuration parameters for agent
        :param scope: TensorFlow scope
        """
        self.config = create_config(config, default=self.default_config)
        self.model = None

        self.memory = ReplayMemory(**self.config)
        self.step_count = 0
        self.update_repeat = self.config.update_repeat
        self.batch_size = self.config.batch_size
        self.update_steps = int(round(1 / self.config.update_rate))
        self.use_target_network = self.config.use_target_network

        if self.use_target_network:
            self.target_update_steps = int(round(1 / self.config.target_network_update_rate))

        self.min_replay_size = self.config.min_replay_size

        if self.__class__.model:
            self.model = self.__class__.model(self.config, scope, network_builder=network_builder)

    def setup(self):
        """
        Prepares the agent to run

        :return:
        """
        self.model.initialize()

    def update(self, batch):
        """
        Explicitly calls update using the provided batch of experiences.

        :param batch:
        :return:
        """
        self.model.update(batch)

    def get_action(self, *args, **kwargs):
        """
        Executes one reinforcement learning step.

        :return: Which action to take
        """
        action = self.model.get_action(*args, **kwargs)

        return action

    def add_observation(self, state, action, reward, terminal):
        """
        Adds an observation for training purposes. Implicitly computes updates
        according to the update frequency.

        :param state: State observed
        :param action: Action taken in state
        :param reward: Reward observed
        :param terminal: Indicates terminal state
        """
        self.memory.add_experience(state, action, reward, terminal)

        self.step_count += 1

        if self.step_count >= self.min_replay_size and self.step_count % self.update_steps == 0:
            for _ in xrange(self.update_repeat):
                batch = self.memory.sample_batch(self.batch_size)
                self.model.update(batch)

        if self.step_count >= self.min_replay_size and self.use_target_network \
                and self.step_count % self.target_update_steps == 0:
            self.model.update_target_network()

    def get_variables(self):
        return self.model.get_variables()

    def assign_variables(self, values):
        self.model.assign_variables(values)

    def get_gradients(self):
        return self.model.get_gradients()

    def apply_gradients(self, grads_and_vars):
        self.model.apply_gradients(grads_and_vars)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
