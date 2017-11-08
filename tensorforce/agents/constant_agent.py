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
Random agent that always returns a random action. Useful to be able to get random
agents with specific shapes.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import Agent
from tensorforce.models.constant_model import ConstantModel


class ConstantAgent(Agent):
    """
    Constant action agent for sanity checks.
    """

    # TODO: Document action_values parameter
    default_config = dict(
        # Agent
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        batched_observe=1000,
        # General
        log_level='info',
        device=None,
        scope='constant',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None
    )

    def __init__(self, states_spec, actions_spec, config):
        config = config.copy()
        config.default(self.__class__.default_config)
        config.obligatory(
            optimizer=None,
            discount=1.0,
            normalize_rewards=False,
            variable_noise=None
        )
        super(ConstantAgent, self).__init__(states_spec, actions_spec, config)

    def initialize_model(self, states_spec, actions_spec, config):
        return ConstantModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            config=config
        )
