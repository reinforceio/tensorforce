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

from tensorforce import TensorForceError
from tensorforce.agents import MemoryAgent
from tensorforce.models import QModel


class DQNAgent(MemoryAgent):
    """
    Deep-Q-Network agent (DQN). The piece de resistance of deep reinforcement learning as described by
    [Minh et al. (2015)](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). Includes
    an option for double-DQN (DDQN; [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461))

    DQN chooses from one of a number of discrete actions by taking the maximum Q-value
    from the value function with one output neuron per available action. DQN uses a replay memory for experience
    playback.

    Configuration:

    Each agent requires the following configuration parameters:

    * `states`: dict containing one or more state definitions.
    * `actions`: dict containing one or more action definitions.
    * `preprocessing`: dict or list containing state preprocessing configuration.
    * `exploration`: dict containing action exploration configuration.

    The `MemoryAgent` class additionally requires the following parameters:

    * `batch_size`: integer of the batch size.
    * `memory_capacity`: integer of maximum experiences to store.
    * `memory`: string indicating memory type ('replay' or 'prioritized_replay').
    * `update_frequency`: integer indicating the number of steps between model updates.
    * `first_update`: integer indicating the number of steps to pass before the first update.
    * `repeat_update`: integer indicating how often to repeat the model update.

    Each model requires the following configuration parameters:

    * `discount`: float of discount factor (gamma).
    * `learning_rate`: float of learning rate (alpha).
    * `optimizer`: string of optimizer to use (e.g. 'adam').
    * `device`: string of tensorflow device name.
    * `tf_summary`: string directory to write tensorflow summaries. Default None
    * `tf_summary_level`: int indicating which tensorflow summaries to create.
    * `tf_summary_interval`: int number of calls to get_action until writing tensorflow summaries on update.
    * `log_level`: string containing logleve (e.g. 'info').
    * `distributed`: boolean indicating whether to use distributed tensorflow.
    * `global_model`: global model.
    * `session`: session to use.

    The DQN agent expects the following additional configuration parameters:

    * `target_update_frequency`: int of states between updates of the target network.
    * `update_target_weight`: float of update target weight (tau parameter).
    * `double_q_model`: boolean indicating whether to use a double q-model.
    * `clip_loss`: float if not 0, uses the huber loss with clip_loss as the linear bound

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

    def __init__(
        self,
        states_spec,
        actions_spec,
        network_spec,
        device=None,
        session_config=None,
        scope='dqn',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None,
        optimizer=None,
        discount=0.99,
        normalize_rewards=False,
        variable_noise=None,
        distributions_spec=None,
        entropy_regularization=None,
        target_sync_frequency=10000,
        target_update_weight=1.0,
        double_q_model=False,
        huber_loss=None,
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        batched_observe=1000,
        batch_size=32,
        memory=None,
        first_update=10000,
        update_frequency=4,
        repeat_update=1
    ):
        """
        Creates a Deep-Q agent.

        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            network_spec: List of layers specifying a neural network via layer types, sizes and optional arguments
                such as activation or regularisation. Full examples are in the examples/configs folder.
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver_spec: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            summary_spec:
            distributed_spec:
            optimizer:
            discount:
            normalize_rewards:
            variable_noise:
            distributions_spec:
            entropy_regularization:
            target_sync_frequency:
            target_update_weight:
            double_q_model:
            huber_loss:
            preprocessing: Optional list of preprocessors (e.g. `image_resize`, `grayscale`) to apply to state. Each
                preprocessor is a dict containing a type and optional necessary arguments.
            exploration: Optional dict specifying exploration type (epsilon greedy strategies or Gaussian noise)
                and arguments.
            reward_preprocessing: Optional dict specifying reward preprocessor using same syntax as state preprocessing.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
            batch_size: Int specifying batch size used to sample from memory. Should be smaller than memory size.
            memory: Dict describing memory via `type` (e.g. `replay`) and `capacity`.
            first_update: Int describing at which time step the first update is performed. Should be larger
                than batch size.
            update_frequency: Int specifying number of observe steps to perform until an update is executed.
            repeat_update: Int specifying how many update steps are performed per update, where each update step implies
                sampling a batch from the memory and passing it to the model.
        """

        if network_spec is None:
            raise TensorForceError("No network_spec provided.")

        if optimizer is None:
            self.optimizer = dict(
                type='adam',
                learning_rate=1e-3
            )
        else:
            self.optimizer = optimizer

        if memory is None:
            memory = dict(
                type='replay',
                capacity=100000
            )
        else:
            self.memory = memory

        self.network_spec = network_spec
        self.device = device
        self.session_config = session_config
        self.scope = scope
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.distributed_spec = distributed_spec
        self.discount = discount
        self.normalize_rewards = normalize_rewards
        self.variable_noise = variable_noise
        self.distributions_spec = distributions_spec
        self.entropy_regularization = entropy_regularization
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight
        self.double_q_model = double_q_model
        self.huber_loss = huber_loss

        super(DQNAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            preprocessing=preprocessing,
            exploration=exploration,
            reward_preprocessing=reward_preprocessing,
            batched_observe=batched_observe,
            batch_size=batch_size,
            memory=memory,
            first_update=first_update,
            update_frequency=update_frequency,
            repeat_update=repeat_update
        )

    def initialize_model(self, states_spec, actions_spec):
        return QModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            device=self.device,
            session_config=self.session_config,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            optimizer=self.optimizer,
            discount=self.discount,
            normalize_rewards=self.normalize_rewards,
            variable_noise=self.variable_noise,
            distributions_spec=self.distributions_spec,
            entropy_regularization=self.entropy_regularization,
            target_sync_frequency=self.target_sync_frequency,
            target_update_weight=self.target_update_weight,
            double_q_model=self.double_q_model,
            huber_loss=self.huber_loss,
            # TEMP: Random sampling fix
            random_sampling_fix=True
        )
