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

from tensorforce.agents import Agent


class BatchAgent(Agent):
    """
    The `BatchAgent` class implements a batch memory, which is cleared after every update.

    Each agent requires the following ``Configuration`` parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `BatchAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.
    * `keep_last_timestep`: bool optionally keep the last observation for use in the next batch

    """

    def __init__(self, states_spec, actions_spec, config):
        assert isinstance(config.batch_size, int) and config.batch_size > 0
        self.batch_size = config.batch_size

        assert isinstance(config.keep_last_timestep, bool)
        self.keep_last_timestep = config.keep_last_timestep

        super(BatchAgent, self).__init__(states_spec, actions_spec, config)

        self.batch = None
        self.reset_batch()

    def observe(self, terminal, reward):
        """
        Adds an observation and performs an update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        Args:
            reward: float of a scalar reward
            terminal: boolean whether episode is terminated or not

        Returns: void
        """
        super(BatchAgent, self).observe(terminal=terminal, reward=reward)

        for name, batch_state in self.batch['states'].items():
            batch_state.append(self.current_states[name])
        for batch_internal, internal in zip(self.batch['internals'], self.current_internals):
            batch_internal.append(internal)
        for name, batch_action in self.batch['actions'].items():
            batch_action.append(self.current_actions[name])
        self.batch['terminal'].append(self.current_terminal)
        self.batch['reward'].append(self.current_reward)

        self.batch_count += 1
        if self.batch_count == self.batch_size:
            self.model.update(batch=self.batch)
            self.reset_batch()

    def reset_batch(self):
        if self.batch is None or not self.keep_last_timestep:
            self.batch = dict(
                states={name: [] for name in self.states_spec},
                internals=[[] for _ in range(len(self.current_internals))],
                actions={name: [] for name in self.actions_spec},
                terminal=[],
                reward=[]
            )
            self.batch_count = 0

        else:
            self.batch = dict(
                states={name: [self.batch['states'][name][-1]] for name in self.states_spec},
                internals=[[self.batch['internals'][i][-1]] for i in range(len(self.current_internals))],
                actions={name: [self.batch['actions'][name][-1]] for name in self.actions_spec},
                terminal=[self.batch['terminal'][-1]],
                reward=[self.batch['reward'][-1]]
            )
            self.batch_count = 1
