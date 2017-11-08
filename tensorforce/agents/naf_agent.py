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

from tensorforce.agents import MemoryAgent
from tensorforce.models import QNAFModel


class NAFAgent(MemoryAgent):
    """
    NAF: https://arxiv.org/abs/1603.00748

    ### Configuration options

    #### General:

    * `scope`: TensorFlow variable scope name (default: 'vpg')

    #### Hyperparameters:

    * `batch_size`: Positive integer (**mandatory**)
    * `learning_rate`: positive float (default: 1e-3)
    * `discount`: Positive float, at most 1.0 (default: 0.99)
    * `normalize_rewards`: Boolean (default: false)
    * `entropy_regularization`: None or positive float (default: none)

    #### Optimizer:

    * `optimizer`: Specification dict (default: Adam with learning rate 1e-3)

    #### Pre-/post-processing:

    * `state_preprocessing`: None or dict with (default: none)
    * `exploration`: None or dict with (default: none)
    * `reward_preprocessing`: None or dict with (default: none)

    #### Logging:

    * `log_level`: Logging level, one of the following values (default: 'info')
        + 'info', 'debug', 'critical', 'warning', 'fatal'

    #### TensorFlow Summaries:
    * `summary_logdir`: None or summary directory string (default: none)
    * `summary_labels`: List of summary labels to be reported, some possible values below (default: 'total-loss')
        + 'total-loss'
        + 'losses'
        + 'variables'
        + 'activations'
        + 'relu'
    * `summary_frequency`: Positive integer (default: 1)
    """

    default_config = dict(
        # Agent
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        batched_observe=1000,
        # MemoryAgent
        # batch_size !!!
        memory=dict(  # not documented!!!
            type='replay',
            capacity=100000
        ),
        first_update=10000,  # not documented!!!
        update_frequency=4,  # not documented!!!
        repeat_update=1,  # not documented!!!
        # Model
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,  # not documented!!!
        # DistributionModel
        distributions_spec=None,  # not documented!!!
        entropy_regularization=None,
        # QModel
        target_sync_frequency=10000,  # not documented!!!
        target_update_weight=1.0,  # not documented!!!
        double_q_model=False,  # not documented!!!
        huber_loss=None,  # not documented!!!
        # General
        log_level='info',
        device=None,
        scope='naf',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None
    )

    def __init__(self, states_spec, actions_spec, network_spec, config):
        self.network_spec = network_spec
        config = config.copy()
        config.default(self.__class__.default_config)
        super(NAFAgent, self).__init__(states_spec, actions_spec, config)

    def initialize_model(self, states_spec, actions_spec, config):
        return QNAFModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            config=config
        )
