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
from tensorforce.agents import BatchAgent
from tensorforce.models import PGProbRatioModel


class PPOAgent(BatchAgent):
    """
    Proximal Policy Optimization agent ([Schulman et al., 2017]
    (https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf).

    ### Configuration options

    #### General:

    * `scope`: TensorFlow variable scope name (default: 'ppo')

    #### Hyperparameters:

    * `batch_size`: Positive integer (**mandatory**)
    * `learning_rate`: positive float (default: 1e-4)
    * `discount`: Positive float, at most 1.0 (default: 0.99)
    * `entropy_regularization`: None or positive float (default: 0.01)
    * `gae_lambda`: None or float between 0.0 and 1.0 (default: none)
    * `normalize_rewards`: Boolean (default: false)
    * `likelihood_ratio_clipping`: None or positive float (default: 0.2)

    #### Multi-step optimizer:

    * `step_optimizer`: Specification dict (default: Adam with learning rate 1e-4)
    * `optimization_steps`: positive integer (default: 10)

    #### Baseline:

    * `baseline_mode`: None, or one of 'states' or 'network' specifying the baseline input (default: none)
    * `baseline`: None or specification dict, or per-state specification for aggregated baseline (default: none)
    * `baseline_optimizer`: None or specification dict (default: none)

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

    def __init__(
            self,
            states_spec,
            actions_spec,
            network_spec,
            device=None,
            session_config=None,
            scope='ppo',
            saver_spec=None,
            summary_spec=None,
            distributed_spec=None,
            discount=0.99,
            normalize_rewards=False,
            variable_noise=None,
            distributions_spec=None,
            entropy_regularization=1e-2,
            baseline_mode=None,
            baseline=None,
            baseline_optimizer=None,
            gae_lambda=None,
            preprocessing=None,
            exploration=None,
            reward_preprocessing=None,
            batched_observe=1000,
            batch_size=1000,
            keep_last_timestep=True,
            likelihood_ratio_clipping=None,
            step_optimizer=None,
            optimization_steps=10
    ):

        # random_sampling=True  # Sampling strategy for replay memory

        """
        Creates a proximal policy optimization agent (PPO), ([Schulman et al., 2017]
        (https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf).

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
            discount:
            normalize_rewards:
            variable_noise:
            distributions_spec:
            entropy_regularization:
            baseline_mode:
            baseline:
            baseline_optimizer:
            gae_lambda:
            preprocessing: Optional list of preprocessors (e.g. `image_resize`, `grayscale`) to apply to state. Each
                preprocessor is a dict containing a type and optional necessary arguments.
            exploration: Optional dict specifying exploration type (epsilon greedy strategies or Gaussian noise)
                and arguments.
            reward_preprocessing: Optional dict specifying reward preprocessor using same syntax as state preprocessing.
            batched_observe: Optional int specifying how many observe calls are batched into one session run.
                Without batching, throughput will be lower because every `observe` triggers a session invocation to
                update rewards in the graph.
            batch_size: Int specifying number of samples collected via `observe` before an update is executed.
            keep_last_timestep: Boolean flag specifying whether last sample is kept, default True.
            likelihood_ratio_clipping:
            step_optimizer:
            optimization_steps:
        """
        if network_spec is None:
            raise TensorForceError("No network_spec provided.")

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
        self.baseline_mode = baseline_mode
        self.baseline = baseline
        self.baseline_optimizer = baseline_optimizer
        self.gae_lambda = gae_lambda
        self.likelihood_ratio_clipping = likelihood_ratio_clipping

        if step_optimizer is None:
            step_optimizer = dict(
                type='adam',
                learning_rate=1e-4
            )

        self.optimizer = dict(
            type='multi_step',
            optimizer=step_optimizer,
            num_steps=optimization_steps
        )

        super(PPOAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            preprocessing=preprocessing,
            exploration=exploration,
            reward_preprocessing=reward_preprocessing,
            batched_observe=batched_observe,
            batch_size=batch_size,
            keep_last_timestep=keep_last_timestep
        )

    def initialize_model(self, states_spec, actions_spec):
        return PGProbRatioModel(
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
            baseline_mode=self.baseline_mode,
            baseline=self.baseline,
            baseline_optimizer=self.baseline_optimizer,
            gae_lambda=self.gae_lambda,
            likelihood_ratio_clipping=self.likelihood_ratio_clipping
        )
