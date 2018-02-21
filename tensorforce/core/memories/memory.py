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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import util
import tensorforce.core.memories


class Memory(object):
    """
    Base class for memories.
    """

    def __init__(self, states, internals, actions, include_next_states, scope='memory', summary_labels=None):
        """
        Memory.

        Args:
            states: States specifiction.
            internals: Internal states specification.
            actions: Actions specification.
            include_next_states: Include subsequent state if true.
        """
        self.states_spec = states
        self.internals_spec = internals
        self.actions_spec = actions
        self.include_next_states = include_next_states
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.summaries = list()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                assert not kwargs.get('trainable', False)
                self.variables[name] = variable
            return variable

        self.initialize = tf.make_template(
            name_=(scope + '/initialize'),
            func_=self.tf_initialize,
            custom_getter_=custom_getter
        )
        self.store = tf.make_template(
            name_=(scope + '/store'),
            func_=self.tf_store,
            custom_getter_=custom_getter
        )
        self.retrieve_timesteps = tf.make_template(
            name_=(scope + '/retrieve_timesteps'),
            func_=self.tf_retrieve_timesteps,
            custom_getter_=custom_getter
        )
        self.retrieve_episodes = tf.make_template(
            name_=(scope + '/retrieve_episodes'),
            func_=self.tf_retrieve_episodes,
            custom_getter_=custom_getter
        )
        self.retrieve_sequences = tf.make_template(
            name_=(scope + '/retrieve_sequences'),
            func_=self.tf_retrieve_sequences,
            custom_getter_=custom_getter
        )
        self.update_batch = tf.make_template(
            name_=(scope + '/update_batch'),
            func_=self.tf_update_batch,
            custom_getter_=custom_getter
        )

    def tf_initialize(self):
        """
        Initializes memory.
        """
        raise NotImplementedError

    def tf_store(self, states, internals, actions, terminal, reward):
        """"
        Stores experiences, i.e. a batch of timesteps.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
        """
        raise NotImplementedError

    def tf_retrieve_timesteps(self, n):
        """
        Retrieves a given number of timesteps from the stored experiences.

        Args:
            n: Number of timesteps to retrieve.

        Returns:
            Dicts containing the retrieved experiences.
        """
        raise NotImplementedError

    def tf_retrieve_episodes(self, n):
        """
        Retrieves a given number of episodes from the stored experiences.

        Args:
            n: Number of episodes to retrieve.

        Returns:
            Dicts containing the retrieved experiences.
        """
        raise NotImplementedError

    def tf_retrieve_sequences(self, n, sequence_length):
        """
        Retrieves a given number of temporally consistent timestep sequences from the stored
        experiences.

        Args:
            n: Number of sequences to retrieve.
            sequence_length: Length of timestep sequences.

        Returns:
            Dicts containing the retrieved experiences.
        """
        raise NotImplementedError

    def tf_update_batch(self, loss_per_instance):
        """
        Updates the internal information of the latest batch instances based on their loss.

        Args:
            loss_per_instance: Loss per instance tensor.
        """
        pass

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the memory.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the memory.

        Returns:
            List of summaries.
        """
        return self.summaries

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a memory from a specification dict.
        """
        memory = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.memories.memories,
            kwargs=kwargs
        )
        assert isinstance(memory, Memory)
        return memory
