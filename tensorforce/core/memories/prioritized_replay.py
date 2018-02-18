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
from tensorforce.core.memories import Memory


class PrioritizedReplay(Memory):
    """
    Naive prioritized replay in TensorFlow. Works by first inserting all new experiences
    into a buffer. When sampling, we first use the buffer of new experiences, then experiences
    from the actual memory for which we have computed priorities.
    """
    def __init__(
        self,
        states,
        internals,
        actions,
        include_next_states,
        capacity,
        prioritization_weight=1.0,
        buffer_size=100,
        scope='queue',
        summary_labels=None
    ):
        super(PrioritizedReplay, self).__init__(
            states=states,
            internals=internals,
            actions=actions,
            include_next_states=include_next_states,
            scope=scope,
            summary_labels=summary_labels
        )
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.prioritization_weight = prioritization_weight

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                assert not kwargs.get('trainable', False)
                self.variables[name] = variable
            return variable

        self.retrieve_indices = tf.make_template(
            name_=(scope + '/retrieve_indices'),
            func_=self.tf_retrieve_indices,
            custom_getter_=custom_getter
        )

        self.update_batch = tf.make_template(
            name_=(scope + '/update_batch'),
            func_=self.tf_update_batch,
            custom_getter_=custom_getter
        )

    def tf_initialize(self):
        # States
        self.states_memory = dict()
        for name, state in self.states_spec.items():
            self.states_memory[name] = tf.get_variable(
                name=('state-' + name),
                shape=(self.capacity,) + tuple(state['shape']),
                dtype=util.tf_dtype(state['type']),
                trainable=False
            )

        # Internals
        self.internals_memory = dict()
        for name, internal in self.internals_spec.items():
            self.internals_memory[name] = tf.get_variable(
                name=('internal-' + name),
                shape=(self.capacity,) + tuple(internal['shape']),
                dtype=util.tf_dtype(internal['type']),
                trainable=False
            )

        # Actions
        self.actions_memory = dict()
        for name, action in self.actions_spec.items():
            self.actions_memory[name] = tf.get_variable(
                name=('action-' + name),
                shape=(self.capacity,) + tuple(action['shape']),
                dtype=util.tf_dtype(action['type']),
                trainable=False
            )

        # Terminal
        self.terminal_memory = tf.get_variable(
            name='terminal',
            shape=(self.capacity,),
            dtype=util.tf_dtype('bool'),
            initializer=tf.constant_initializer(
                value=tuple(n == self.capacity - 1 for n in range(self.capacity)),
                dtype=util.tf_dtype('bool')
            ),
            trainable=False
        )

        # Reward
        self.reward_memory = tf.get_variable(
            name='reward',
            shape=(self.capacity,),
            dtype=util.tf_dtype('float'),
            trainable=False
        )

        # Memory index
        self.memory_index = tf.get_variable(
            name='memory-index',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        # Priorities
        self.priorities = tf.get_variable(
            name='priorities',
            shape=(self.capacity,),
            dtype=util.tf_dtype('float'),
            trainable=False
        )

        # Buffer variables. The buffer is used to insert data for which we
        # do not have priorities yet.
        self.buffer_index = tf.get_variable(
            name='buffer-index',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        self.states_buffer = dict()
        for name, state in self.states_spec.items():
            self.states_buffer[name] = tf.get_variable(
                name=('state-buffer-' + name),
                shape=(self.buffer_size,) + tuple(state['shape']),
                dtype=util.tf_dtype(state['type']),
                trainable=False
            )

        # Internals
        self.internals_buffer = dict()
        for name, internal in self.internals_spec.items():
            self.internals_buffer[name] = tf.get_variable(
                name=('internal-buffer-' + name),
                shape=(self.capacity,) + tuple(internal['shape']),
                dtype=util.tf_dtype(internal['type']),
                trainable=False
            )

        # Actions
        self.actions_buffer = dict()
        for name, action in self.actions_spec.items():
            self.actions_buffer[name] = tf.get_variable(
                name=('action-buffer-' + name),
                shape=(self.buffer_size,) + tuple(action['shape']),
                dtype=util.tf_dtype(action['type']),
                trainable=False
            )

        # Terminal
        self.terminal_buffer = tf.get_variable(
            name='terminal-buffer',
            shape=(self.capacity,),
            dtype=util.tf_dtype('bool'),
            initializer=tf.constant_initializer(
                value=tuple(n == self.buffer_size - 1 for n in range(self.capacity)),
                dtype=util.tf_dtype('bool')
            ),
            trainable=False
        )

        # Reward
        self.reward_buffer = tf.get_variable(
            name='reward-buffer',
            shape=(self.buffer_size,),
            dtype=util.tf_dtype('float'),
            trainable=False
        )

        # Indices of batch experiences in main memory.
        self.batch_indices = tf.get_variable(
            name='batch-indices',
            dtype=util.tf_dtype('int'),
            shape=(self.capacity,),
            trainable=False
        )

        # Indices of batch experiences in buffer..
        self.last_batch_buffer_elems = tf.get_variable(
            name='last-batch-buffer-elems',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

    def tf_store(self, states, internals, actions, terminal, reward):
        # We first store new experiences into a buffer that is separate from main memory.
        # We insert these into the main memory once we have computed priorities on a given batch.
        num_instances = tf.shape(input=terminal)[0]
        start_index = self.buffer_index
        end_index = self.buffer_index + num_instances

        # Assign new observations.
        assignments = list()
        for name, state in states.items():
            assignments.append(tf.assign(ref=self.states_buffer[name][start_index:end_index], value=state))
        for name, internal in internals.items():
            assignments.append(tf.assign(
                ref=self.internals_buffer[name][start_index:end_index],
                value=internal
            ))
        for name, action in actions.items():
            assignments.append(tf.assign(ref=self.actions_buffer[name][start_index:end_index], value=action))

        assignments.append(tf.assign(ref=self.terminal_buffer[start_index:end_index], value=terminal))
        assignments.append(tf.assign(ref=self.reward_buffer[start_index:end_index], value=reward))

        # Increment memory index.
        with tf.control_dependencies(control_inputs=assignments):
            assignment = tf.assign(ref=self.buffer_index, value=(self.buffer_index + num_instances))

        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.no_op()

    def tf_retrieve_timesteps(self, n):
        num_buffer_elems = tf.minimum(x=self.buffer_index, y=n)
        num_priority_elements = n - num_buffer_elems

        def sampling_fn():
            # Vectorized sampling.
            sum_priorities = tf.reduce_sum(input_tensor=self.priorities, axis=0)
            sample = tf.random_uniform(shape=(num_priority_elements,), dtype=tf.float32)
            indices = tf.zeros(shape=(num_priority_elements,), dtype=tf.int32)

            def cond(loop_index, sample):
                return tf.reduce_all(input_tensor=(sample <= 0.0))

            def sampling_body(loop_index, sample):
                priority = tf.gather(params=self.priorities, indices=loop_index)
                sample -= priority / sum_priorities
                loop_index += tf.cast(
                    x=(sample > 0.0),
                    dtype=tf.int32,
                )

                return loop_index, sample

            priority_indices = tf.while_loop(
                cond=cond,
                body=sampling_body,
                loop_vars=(indices, sample)
            )[0]
            return priority_indices

        priority_indices = tf.cond(
            pred=num_priority_elements > 0,
            true_fn=sampling_fn,
            false_fn=lambda: tf.zeros(shape=(num_priority_elements,), dtype=tf.int32)
        )
        priority_terminal = tf.gather(params=self.terminal_memory, indices=priority_indices)
        priority_indices = tf.boolean_mask(tensor=priority_indices, mask=tf.logical_not(x=priority_terminal))

        # Store how many elements we retrieved from the buffer for updating priorities.
        # Note that this is just the count, as we can reconstruct the indices from that.
        assignments = list()
        assignments.append(tf.assign(ref=self.last_batch_buffer_elems, value=num_buffer_elems))

        # Store indices used from priority memory. Note that these are the full indices
        # as they were not taken in order.
        assignments.append(tf.scatter_update(
            ref=self.batch_indices,
            indices=priority_indices,
            updates=tf.ones(shape=tf.shape(input=priority_indices), dtype=tf.int32))
        )
        # Fetch results.
        with tf.control_dependencies(control_inputs=assignments):
            return self.retrieve_indices(buffer_elements=num_buffer_elems, priority_indices=priority_indices)

    def tf_retrieve_indices(self, buffer_elements, priority_indices):
        """
        Fetches experiences for given indices by combining entries from buffer
        which have no priorities, and entries from priority memory.

        Args:
            buffer_elements: Number of buffer elements to retrieve
            priority_indices: Index tensor for priority memory

        Returns: Batch of experiences
        """
        states = dict()

        buffer_start = (self.buffer_index - buffer_elements)
        buffer_start = tf.Print(buffer_start, [buffer_start], 'buffer start=', summarize=100)
        buffer_end = (self.buffer_index)
        buffer_end = tf.Print(buffer_end, [buffer_end], 'buffer_end=', summarize=100)
        # Fetch entries from respective memories, concat.
        for name, state_memory in self.states_memory.items():
            buffer_state_memory = self.states_buffer[name]
            buffer_states = buffer_state_memory[buffer_start:buffer_end]
            memory_states = tf.gather(params=state_memory, indices=priority_indices)
            # buffer_states = tf.Print(buffer_states, [buffer_states], "buffer states=", summarize=100)
            # memory_states = tf.Print(memory_states, [memory_states], "memory states=", summarize=100)
            states[name] = tf.concat(values=(buffer_states, memory_states), axis=0)

        internals = dict()
        for name, internal_memory in self.internals_memory.items():
            internal_buffer_memory = self.internals_buffer[name]
            buffer_internals = internal_buffer_memory[buffer_start:buffer_end]
            memory_internals = tf.gather(params=internal_memory, indices=priority_indices)
            internals[name] = tf.concat(values=(buffer_internals, memory_internals), axis=0)

        actions = dict()
        for name, action_memory in self.actions_memory.items():
            action_buffer_memory = self.actions_buffer[name]
            buffer_action = action_buffer_memory[buffer_start:buffer_end]
            memory_action = tf.gather(params=action_memory, indices=priority_indices)
            actions[name] = tf.concat(values=(buffer_action, memory_action), axis=0)

        buffer_terminal = self.terminal_buffer[buffer_start:buffer_end]
        priority_terminal = tf.gather(params=self.terminal_memory, indices=priority_indices)
        terminal = tf.concat(values=(buffer_terminal, priority_terminal), axis=0)

        buffer_reward = self.reward_buffer[buffer_start:buffer_end]
        priority_reward = tf.gather(params=self.reward_memory, indices=priority_indices)
        reward = tf.concat(values=(buffer_reward, priority_reward), axis=0)

        if self.include_next_states:
            assert util.rank(priority_indices) == 1
            next_priority_indices = (priority_indices + 1) % self.capacity
            next_buffer_start = (buffer_start + 1) % self.buffer_size
            next_buffer_end = (buffer_end + 1) % self.buffer_size
            # else:
            #     next_indices = (indices[:, -1] + 1) % self.capacity

            next_states = dict()
            for name, state_memory in self.states_memory.items():
                buffer_state_memory = self.states_buffer[name]
                buffer_next_states = buffer_state_memory[next_buffer_start:next_buffer_end]
                memory_next_states = tf.gather(params=state_memory, indices=next_priority_indices)
                next_states[name] = tf.concat(values=(buffer_next_states, memory_next_states), axis=0)

            next_internals = dict()
            for name, internal_memory in self.internals_memory.items():
                buffer_internal_memory = self.internals_buffer[name]
                buffer_next_internals = buffer_internal_memory[next_buffer_start:next_buffer_end]
                memory_next_internals = tf.gather(params=internal_memory, indices=next_priority_indices)
                next_internals[name] = tf.concat(values=(buffer_next_internals, memory_next_internals), axis=0)

            return dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                next_states=next_states,
                next_internals=next_internals
            )
        else:
            return dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )

    def tf_update_batch(self, loss_per_instance):
        """
        Updates priorities and moves respective elements from buffer to priority memory.

        :param loss_per_instance: Losses for last recent batch
        """
        # 1. We reconstruct the batch from the buffer and the priority memory.
        mask = tf.not_equal(
            x=self.batch_indices,
            y=tf.zeros(shape=tf.shape(input=self.batch_indices), dtype=tf.int32)
        )
        priority_indices = tf.where(condition=mask)
        priority_indices = tf.Print(priority_indices, [priority_indices], message="Priority indices")
        sampled_batch = self.tf_retrieve_indices(
            buffer_elements=self.last_batch_buffer_elems,
            priority_indices=priority_indices
        )
        sampled_batch = tf.Print(sampled_batch, [sampled_batch], message="sampled batch: ")
        states = sampled_batch['states']
        internals = sampled_batch['internals']
        actions = sampled_batch['actions']
        terminal = sampled_batch['terminal']
        reward = sampled_batch['reward']

        # TODO this is incorrect
        start_index = 0
        end_index = self.last_batch_buffer_elems
        priorities = loss_per_instance ** self.prioritization_weight
        # How do we map batch indices to memory indices and insert?

        # For testing retrieval loop, no priority inserts yet.
        assignments = list()
        for name, state in states.items():
            assignments.append(tf.assign(ref=self.states_memory[name][start_index:end_index], value=state))
        for name, internal in internals.items():
            assignments.append(tf.assign(
                ref=self.internals_buffer[name][start_index:end_index],
                value=internal
            ))
        assignments.append(tf.assign(ref=self.terminal_memory[start_index:end_index], value=terminal))
        assignments.append(tf.assign(ref=self.reward_memory[start_index:end_index], value=reward))
        assignments.append(tf.assign(ref=self.priorities[start_index:end_index], value=priorities))

        for name, action in actions.items():
            assignments.append(tf.assign(ref=self.actions_memory[name][start_index:end_index], value=action))

        # 2. We delete entries from the priority memory. There is no need
        # to delete entries from the buffer because we just move the idnex.

        # Start index for inserting
        # buffer_end_insert = tf.constant(value=)

        # Reset buffer index.
        with tf.control_dependencies(control_inputs=assignments):
            assignment = tf.assign_sub(ref=self.buffer_index, value=self.last_batch_buffer_elems)
        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.no_op()

    def tf_retrieve_episodes(self, n):
        pass

    def tf_retrieve_sequences(self, n, sequence_length):
        pass
