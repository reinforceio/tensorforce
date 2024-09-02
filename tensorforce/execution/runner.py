# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import time

import numpy as np
from tqdm.auto import tqdm

from tensorforce import Agent, Environment, TensorforceError, util
from tensorforce.environments import RemoteEnvironment


class Runner(object):
    """
    Tensorforce runner utility.

    Args:
        agent (specification | Agent object | Agent.load kwargs): Agent specification or object
            (<b>note</b>: if passed as object, `agent.close()` is not (!) automatically triggered
            as part of `runner.close()`), or keyword arguments to `Agent.load()` in particular
            containing `directory`, in all cases argument `environment` is implicitly specified
            as the following argument, and argument `parallel_interactions` is either implicitly
            specified as `num_parallel` or expected to be at least `num_parallel`
            (<span style="color:#C00000"><b>required</b></span>).
        environment (specification | Environment object): Environment specification or object
            (<b>note</b>: if passed as object, `environment.close()` is not (!) automatically
            triggered as part of `runner.close()`), where argument `max_episode_timesteps` is
            implicitly specified as the following argument
            (<span style="color:#C00000"><b>required</b></span>, or alternatively `environments`,
            invalid for "socket-client" remote mode).
        max_episode_timesteps (int > 0): Maximum number of timesteps per episode, overwrites the
            environment default if defined
            (<span style="color:#00C000"><b>default</b></span>: environment default, invalid for
            "socket-client" remote mode).
        num_parallel (int >= 2): Number of environment instances to execute in parallel, usually
            requires argument `remote` to be specified for proper parallel execution unless
            vectorizable environment
            (<span style="color:#00C000"><b>default</b></span>: no parallel execution, implicitly
            specified by `environments`).
        environments (list[specification | Environment object]): Environment specifications or
            objects to execute in parallel, the latter are not closed automatically as part of
            `runner.close()`
            (<span style="color:#00C000"><b>default</b></span>: no parallel execution,
            alternatively specified via `environment` and `num_parallel`, invalid for
            "socket-client" remote mode).
        evaluation (bool): Whether to run the last of multiple parallel environments in evaluation
            mode, only valid with `num_parallel` or `environments`
            (<span style="color:#00C000"><b>default</b></span>: no evaluation).
        remote ("multiprocessing" | "socket-client"): Communication mode for remote environment
            execution of parallelized environment execution, not compatible with environment(s)
            given as Environment objects, "socket-client" mode requires a corresponding
            "socket-server" running
            (<span style="color:#00C000"><b>default</b></span>: local execution).
        blocking (bool): Whether remote environment calls should be blocking, only valid if remote
            mode given
            (<span style="color:#00C000"><b>default</b></span>: not blocking, invalid unless
            "multiprocessing" or "socket-client" remote mode).
        host (str, iter[str]): Socket server hostname(s) or IP address(es)
            (<span style="color:#C00000"><b>required</b></span> only for "socket-client" remote
            mode).
        port (int, iter[int]): Socket server port(s), increasing sequence if single host and port
            given
            (<span style="color:#C00000"><b>required</b></span> only for "socket-client" remote
            mode).
    """

    def __init__(
        self, agent, environment=None, max_episode_timesteps=None, num_parallel=None,
        environments=None, evaluation=False, remote=None, blocking=False, host=None, port=None,
        **kwargs
    ):
        if environment is None and environments is None:
            if remote != 'socket-client':
                raise TensorforceError.required(
                    name='Runner', argument='environment or environments'
                )
            if num_parallel is None:
                raise TensorforceError.required(
                    name='Runner', argument='num_parallel', condition='socket-client remote mode'
                )
            environments = [None for _ in range(num_parallel)]

        elif environment is None:
            if environments is None:
                raise TensorforceError.required(
                    name='Runner', argument='environment or environments'
                )
            if not util.is_iterable(x=environments):
                raise TensorforceError.type(
                    name='Runner', argument='environments', value=environments
                )
            if len(environments) == 0:
                raise TensorforceError.value(
                    name='Runner', argument='len(environments)', value=len(environments)
                )
            if num_parallel is not None and num_parallel != len(environments):
                raise TensorforceError.value(
                    name='Runner', argument='num_parallel', value=num_parallel,
                    hint='!= len(environments)'
                )
            num_parallel = len(environments)
            environments = list(environments)

        elif num_parallel is None:
            if environments is not None:
                raise TensorforceError.invalid(
                    name='Runner', argument='environments', condition='environment is specified'
                )
            if evaluation:
                raise TensorforceError.invalid(
                    name='Runner', argument='evaluation', condition='single environment'
                )
            num_parallel = 1
            environments = [environment]

        else:
            if not isinstance(num_parallel, int):
                raise TensorforceError.value(
                    name='Runner', argument='num_parallel', dtype=type(num_parallel)
                )
            elif num_parallel < 2:
                raise TensorforceError.value(
                    name='Runner', argument='num_parallel', value=num_parallel, hint='< 2'
                )
            if environments is not None:
                raise TensorforceError.invalid(
                    name='Runner', argument='environments', condition='environment is specified'
                )
            if isinstance(environment, Environment):
                raise TensorforceError.value(
                    name='Runner', argument='environment', value=environment,
                    condition='num_parallel',
                    hint='is Environment instance, but specification dict is required'
                )
            environments = [environment for _ in range(num_parallel)]

        if port is None or isinstance(port, int):
            if isinstance(host, str):
                port = [port + n for n in range(num_parallel)]
            else:
                port = [port for _ in range(num_parallel)]
        else:
            if len(port) != num_parallel:
                raise TensorforceError.value(
                    name='Runner', argument='len(port)', value=len(port), hint='!= num_parallel'
                )
        if host is None or isinstance(host, str):
            host = [host for _ in range(num_parallel)]
        else:
            if len(host) != num_parallel:
                raise TensorforceError.value(
                    name='Runner', argument='len(host)', value=len(host), hint='!= num_parallel'
                )

        self.environments = list()
        self.is_environment_external = isinstance(environments[0], Environment)
        environment = Environment.create(
            environment=environments[0], max_episode_timesteps=max_episode_timesteps,
            remote=remote, blocking=blocking, host=host[0], port=port[0], **kwargs
        )
        self.is_environment_remote = isinstance(environment, RemoteEnvironment)
        states = environment.states()
        actions = environment.actions()
        self.environments.append(environment)
        if remote is None and num_parallel > 1 and environment.is_vectorizable():
            self.num_vectorized = num_parallel
            environments = environments[:1]
            if evaluation:
                raise TensorforceError.invalid(
                    name='Runner', argument='evaluation', condition='vectorized environment'
                )
        elif environment.num_actors() > 1:
            assert num_parallel == 1
            num_parallel = environment.num_actors()
            self.num_vectorized = environment.num_actors()
        else:
            self.num_vectorized = None

        for n, environment in enumerate(environments[1:], start=1):
            assert isinstance(environment, Environment) == self.is_environment_external
            environment = Environment.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps,
                remote=remote, blocking=blocking, host=host[n], port=port[n], **kwargs
            )
            assert isinstance(environment, RemoteEnvironment) == self.is_environment_remote
            assert util.is_equal(x=environment.states(), y=states)
            assert util.is_equal(x=environment.actions(), y=actions)
            self.environments.append(environment)

        self.evaluation = evaluation

        self.is_agent_external = isinstance(agent, Agent)
        if not self.is_agent_external and 'directory' in agent:
            self.agent = Agent.load(
                **agent, environment=environment,
                parallel_interactions=(num_parallel - int(self.evaluation))
            )
        elif num_parallel - int(self.evaluation) > 1:
            self.agent = Agent.create(
                agent=agent, environment=environment,
                parallel_interactions=(num_parallel - int(self.evaluation))
            )
        else:
            self.agent = Agent.create(agent=agent, environment=environment)

    def close(self):
        if hasattr(self, 'tqdm'):
            self.tqdm.close()
        if not self.is_agent_external:
            self.agent.close()
        if not self.is_environment_external:
            for environment in self.environments:
                environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(
        self,
        # General
        num_episodes=None, num_timesteps=None, num_updates=None,
        # Parallel
        batch_agent_calls=False, sync_timesteps=False, sync_episodes=False, num_sleep_secs=0.001,
        # Callback
        callback=None, callback_episode_frequency=None, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, mean_horizon=1,
        # Evaluation
        evaluation=False, save_best_agent=None, evaluation_callback=None
    ):
        """
        Run experiment.

        Args:
            num_episodes (int > 0): Number of episodes to run experiment, sum of episodes across all
                parallel/vectorized environment(s) / actors in a multi-actor environment
                (<span style="color:#00C000"><b>default</b></span>: no episode limit).
            num_timesteps (int > 0): Number of timesteps to run experiment, sum of timesteps across
                all parallel/vectorized environment(s) / actors in a multi-actor environment
                (<span style="color:#00C000"><b>default</b></span>: no timestep limit).
            num_updates (int > 0): Number of agent updates to run experiment
                (<span style="color:#00C000"><b>default</b></span>: no update limit).
            batch_agent_calls (bool): Whether to batch agent calls for parallel environment
                execution
                (<span style="color:#00C000"><b>default</b></span>: false, separate call per
                environment).
            sync_timesteps (bool): Whether to synchronize parallel environment execution on
                timestep-level, implied by batch_agent_calls
                (<span style="color:#00C000"><b>default</b></span>: false, unless
                batch_agent_calls is true).
            sync_episodes (bool): Whether to synchronize parallel environment execution on
                episode-level
                (<span style="color:#00C000"><b>default</b></span>: false).
            num_sleep_secs (float): Sleep duration if no environment is ready
                (<span style="color:#00C000"><b>default</b></span>: one milliseconds).
            callback (callable[(Runner, parallel) -> bool]): Callback function taking the runner
                instance plus parallel index and returning a boolean value indicating whether
                execution should continue
                (<span style="color:#00C000"><b>default</b></span>: callback always true).
            callback_episode_frequency (int): Episode interval between callbacks
                (<span style="color:#00C000"><b>default</b></span>: every episode).
            callback_timestep_frequency (int): Timestep interval between callbacks
                (<span style="color:#00C000"><b>default</b></span>: not specified).
            use_tqdm (bool): Whether to display a tqdm progress bar for the experiment run
                (<span style="color:#00C000"><b>default</b></span>: true), with the following
                additional information (averaged over number of episodes given via mean_horizon):
                <ul>
                <li>return &ndash; cumulative episode return</li>
                <li>std &ndash; standard deviation of cumulative episode return</li>
                <li>ts/ep &ndash; timesteps per episode</li>
                <li>sec/ep &ndash; seconds per episode</li>
                <li>ms/ts &ndash; milliseconds per timestep</li>
                <li>agent &ndash; percentage of time spent on agent computation</li>
                <li>comm &ndash; if remote environment execution, percentage of time spent on
                communication</li>
                </ul>
            mean_horizon (int): Number of episodes progress bar values and evaluation score are
                averaged over (<span style="color:#00C000"><b>default</b></span>: not averaged).
            evaluation (bool): Whether to run in evaluation mode, only valid if single environment
                (<span style="color:#00C000"><b>default</b></span>: no evaluation).
            save_best_agent (string): Directory to save the best version of the agent according to
                the evaluation score
                (<span style="color:#00C000"><b>default</b></span>: best agent is not saved).
            evaluation_callback (int | callable[Runner -> float]): Callback function taking the
                runner instance and returning an evaluation score
                (<span style="color:#00C000"><b>default</b></span>: cumulative evaluation return
                averaged over mean_horizon episodes).
        """
        # General
        if num_episodes is None:
            self.num_episodes = float('inf')
        else:
            self.num_episodes = num_episodes
        if num_timesteps is None:
            self.num_timesteps = float('inf')
        else:
            self.num_timesteps = num_timesteps
        if num_updates is None:
            self.num_updates = float('inf')
        else:
            self.num_updates = num_updates

        # Parallel
        if len(self.environments) == 1:
            condition = 'single environment'
        elif self.num_vectorized is not None:
            condition = 'vectorized environment'
        else:
            condition = None
        if condition is None:
            pass
        elif batch_agent_calls:
            raise TensorforceError.invalid(
                name='Runner.run', argument='batch_agent_calls', condition=condition
            )
        elif sync_timesteps:
            raise TensorforceError.invalid(
                name='Runner.run', argument='sync_timesteps', condition=condition
            )
        elif sync_episodes:
            raise TensorforceError.invalid(
                name='Runner.run', argument='sync_episodes', condition=condition
            )
        self.batch_agent_calls = batch_agent_calls or (self.num_vectorized is not None)
        self.sync_timesteps = sync_timesteps or self.batch_agent_calls
        self.sync_episodes = sync_episodes or (self.num_vectorized is not None)
        self.num_sleep_secs = num_sleep_secs
        if self.num_vectorized is None:
            self.num_environments = len(self.environments)
        else:
            self.num_environments = self.num_vectorized

        # Callback
        assert callback_episode_frequency is None or callback_timestep_frequency is None
        if callback_episode_frequency is None and callback_timestep_frequency is None:
            callback_episode_frequency = 1
        if callback_episode_frequency is None:
            self.callback_episode_frequency = float('inf')
        else:
            self.callback_episode_frequency = callback_episode_frequency
        if callback_timestep_frequency is None:
            self.callback_timestep_frequency = float('inf')
        else:
            self.callback_timestep_frequency = callback_timestep_frequency
        if callback is None:
            self.callback = (lambda r, p: True)
        elif util.is_iterable(x=callback):
            def sequential_callback(runner, parallel):
                result = True
                for fn in callback:
                    x = fn(runner, parallel)
                    if isinstance(result, bool):
                        result = result and x
                return result
            self.callback = sequential_callback
        else:
            def boolean_callback(runner, parallel):
                result = callback(runner, parallel)
                if isinstance(result, bool):
                    return result
                else:
                    return True
            self.callback = boolean_callback

        # Experiment statistics
        self.episode_returns = list()
        self.episode_timesteps = list()
        self.episode_seconds = list()
        self.episode_agent_seconds = list()
        if self.is_environment_remote:
            self.episode_env_seconds = list()
        if self.evaluation or evaluation:
            self.evaluation_returns = list()
            self.evaluation_timesteps = list()
            self.evaluation_seconds = list()
            self.evaluation_agent_seconds = list()
            if self.is_environment_remote:
                self.evaluation_env_seconds = list()
            if self.num_environments == 1:
                # for tqdm
                self.episode_returns = self.evaluation_returns
                self.episode_timesteps = self.evaluation_timesteps
                self.episode_seconds = self.evaluation_seconds
                self.episode_agent_seconds = self.evaluation_agent_seconds
                if self.is_environment_remote:
                    self.episode_env_seconds = self.evaluation_env_seconds
        else:
            # for tqdm
            self.evaluation_returns = self.episode_returns
            self.evaluation_timesteps = self.episode_timesteps
            self.evaluation_seconds = self.episode_seconds
            self.evaluation_agent_seconds = self.episode_agent_seconds
            if self.is_environment_remote:
                self.evaluation_env_seconds = self.episode_env_seconds

        # Timestep/episode/update counter
        self.timesteps = 0
        self.episodes = 0
        self.updates = 0

        # Tqdm
        if use_tqdm:
            if hasattr(self, 'tqdm'):
                self.tqdm.close()

            assert self.num_episodes != float('inf') or self.num_timesteps != float('inf')
            inner_callback = self.callback

            if self.num_episodes != float('inf'):
                # Episode-based tqdm (default option if both num_episodes and num_timesteps set)
                assert self.num_episodes != float('inf')
                bar_format = (
                    '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, return={postfix[0]:.2f}, stddev={postfix[1]:.2f}, ts/ep='
                    '{postfix[2]}, sec/ep={postfix[3]:.2f}, ms/ts={postfix[4]:.1f}, agent='
                    '{postfix[5]:.1f}%]'
                )
                postfix = [0.0, 0.0, 0, 0.0, 0.0, 0.0]
                if self.is_environment_remote:
                    bar_format = bar_format[:-1] + ', comm={postfix[6]:.1f}%]'
                    postfix.append(0.0)

                self.tqdm = tqdm(
                    desc='Episodes', total=self.num_episodes, bar_format=bar_format,
                    initial=self.episodes, postfix=postfix
                )
                self.tqdm_last_update = self.episodes

                def tqdm_callback(runner, parallel):
                    if len(runner.evaluation_returns) > 0:
                        mean_return = float(np.mean(runner.evaluation_returns[-mean_horizon:]))
                        std_return = float(np.std(runner.evaluation_returns[-mean_horizon:]))
                        runner.tqdm.postfix[0] = mean_return
                        runner.tqdm.postfix[1] = std_return
                    if len(runner.episode_timesteps) > 0:
                        mean_ts_per_ep = int(np.mean(runner.episode_timesteps[-mean_horizon:]))
                        mean_sec_per_ep = float(np.mean(runner.episode_seconds[-mean_horizon:]))
                        mean_agent_sec = float(
                            np.mean(runner.episode_agent_seconds[-mean_horizon:])
                        )
                        try:
                            mean_ms_per_ts = mean_sec_per_ep * 1000.0 / mean_ts_per_ep
                        except ZeroDivisionError:
                            mean_ms_per_ts = 0.0
                        try:
                            mean_rel_agent = mean_agent_sec * 100.0 / mean_sec_per_ep
                        except ZeroDivisionError:
                            mean_rel_agent = 0.0
                        runner.tqdm.postfix[2] = mean_ts_per_ep
                        runner.tqdm.postfix[3] = mean_sec_per_ep
                        runner.tqdm.postfix[4] = mean_ms_per_ts
                        runner.tqdm.postfix[5] = mean_rel_agent
                    if runner.is_environment_remote and len(runner.episode_env_seconds) > 0:
                        mean_env_sec = float(np.mean(runner.episode_env_seconds[-mean_horizon:]))
                        mean_rel_comm = (mean_agent_sec + mean_env_sec) * 100.0 / mean_sec_per_ep
                        mean_rel_comm = 100.0 - mean_rel_comm
                        runner.tqdm.postfix[6] = mean_rel_comm
                    runner.tqdm.update(n=(runner.episodes - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.episodes
                    return inner_callback(runner, parallel)

            else:
                # Timestep-based tqdm
                self.tqdm = tqdm(
                    desc='Timesteps', total=self.num_timesteps, initial=self.timesteps,
                    postfix=dict(mean_return='n/a')
                )
                self.tqdm_last_update = self.timesteps

                def tqdm_callback(runner, parallel):
                    # sum_timesteps_return = sum(runner.timestep_returns[num_mean_return:])
                    # num_timesteps = min(num_mean_return, runner.evaluation_timestep)
                    # mean_return = sum_timesteps_return / num_episodes
                    runner.tqdm.set_postfix(mean_return='n/a')
                    runner.tqdm.update(n=(runner.timesteps - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.timesteps
                    return inner_callback(runner, parallel)

            self.callback = tqdm_callback

        # Evaluation
        if evaluation and self.num_environments > 1:
            raise TensorforceError.invalid(
                name='Runner.run', argument='evaluation', condition='parallel environments'
            )
        self.evaluation_run = self.evaluation or evaluation
        self.save_best_agent = save_best_agent
        if evaluation_callback is None:
            self.evaluation_callback = (lambda r: None)
        else:
            self.evaluation_callback = evaluation_callback
        if self.save_best_agent is not None:
            inner_evaluation_callback = self.evaluation_callback

            def mean_return_callback(runner):
                result = inner_evaluation_callback(runner)
                if result is None:
                    return float(np.mean(runner.evaluation_returns[-mean_horizon:]))
                else:
                    return result

            self.evaluation_callback = mean_return_callback
            self.best_evaluation_score = None

        # Episode statistics
        self.episode_return = [0.0 for _ in range(self.num_environments)]
        self.episode_timestep = [0 for _ in range(self.num_environments)]
        # if self.batch_agent_calls:
        #     self.episode_agent_second = 0.0
        #     self.episode_start = time.time()
        if self.evaluation_run:
            self.episode_agent_second = [0.0 for _ in range(self.num_environments - 1)]
            self.episode_start = [time.time() for _ in range(self.num_environments - 1)]
        else:
            self.episode_agent_second = [0.0 for _ in range(self.num_environments)]
            self.episode_start = [time.time() for _ in range(self.num_environments)]
        self.evaluation_agent_second = 0.0
        self.evaluation_start = time.time()

        # Values
        self.terminate = 0
        self.prev_terminals = [-1 for _ in range(self.num_environments)]
        self.states = [None for _ in range(self.num_environments)]
        self.terminals = [None for _ in range(self.num_environments)]
        self.rewards = [None for _ in range(self.num_environments)]
        if self.evaluation_run:
            self.evaluation_internals = self.agent.initial_internals()

        # Required if agent was previously stopped mid-episode
        self.agent.reset()

        # Reset environments
        if self.num_vectorized is None:
            for environment in self.environments:
                environment.start_reset()
        else:
            if self.environments[0].is_vectorizable():
                parallel, states = self.environments[0].reset(num_parallel=self.num_vectorized)
            else:
                parallel, states = self.environments[0].reset()
            for i, n in enumerate(parallel):
                self.states[n] = states[i]
                self.prev_terminals[n] = -2

        # Runner loop
        while any(terminal <= 0 for terminal in self.prev_terminals):
            self.terminals = [None for _ in self.terminals]

            if self.batch_agent_calls:

                if self.num_vectorized is None:
                    # Retrieve observations (only if not already terminated)
                    while any(terminal is None for terminal in self.terminals):
                        for n in range(self.num_environments):
                            if self.terminals[n] is not None:
                                # Already received
                                continue
                            elif self.prev_terminals[n] <= 0:
                                # Receive if not terminal
                                observation = self.environments[n].receive_execute()
                                if observation is None:
                                    continue
                                self.states[n], self.terminals[n], self.rewards[n] = observation
                            else:
                                # Terminal
                                self.states[n] = None
                                self.terminals[n] = self.prev_terminals[n]
                                self.rewards[n] = None

                else:
                    # Vectorized environment execute
                    if all(terminal >= -1 for terminal in self.prev_terminals):
                        parallel, states, terminals, rewards = self.environments[0].execute(
                            actions=np.asarray(self.actions)
                        )
                        i = 0
                        for n, terminal in enumerate(self.prev_terminals):
                            if terminal <= 0:
                                self.terminals[n] = terminals[i]
                                self.rewards[n] = rewards[i]
                                if terminals[i] > 0:
                                    self.states[n] = None
                                i += 1
                            else:
                                self.states[n] = None
                                self.terminals[n] = self.prev_terminals[n]
                                self.rewards[n] = None
                        for i, n in enumerate(parallel):
                            assert self.terminals[n] <= 0 or self.terminals[n] == 2
                            self.states[n] = states[i]
                    else:
                        for n, terminal in enumerate(self.prev_terminals):
                            if terminal < -1:
                                self.terminals[n] = -1
                            else:
                                self.terminals[n] = self.prev_terminals[n]

                self.handle_observe_joint()
                self.handle_act_joint()

            # Parallel environments loop
            no_environment_ready = True
            for n in range(self.num_environments):

                if self.prev_terminals[n] > 0:
                    # Continue if episode terminated (either sync_episodes or finished)
                    self.terminals[n] = self.prev_terminals[n]
                    continue

                elif self.batch_agent_calls:
                    # Handled before parallel environments loop
                    pass

                elif self.sync_timesteps:
                    # Wait until environment is ready
                    while True:
                        observation = self.environments[n].receive_execute()
                        if observation is not None:
                            break

                else:
                    # Check whether environment is ready, otherwise continue
                    observation = self.environments[n].receive_execute()
                    if observation is None:
                        self.terminals[n] = self.prev_terminals[n]
                        continue

                no_environment_ready = False
                if not self.batch_agent_calls:
                    self.states[n], self.terminals[n], self.rewards[n] = observation

                # Check whether evaluation environment
                if self.evaluation_run and n == self.num_environments - 1:
                    if self.terminals[n] == -1:
                        # Initial act
                        self.handle_act_evaluation()
                    else:
                        # Observe
                        self.handle_observe_evaluation()
                        if self.terminals[n] == 0:
                            # Act
                            self.handle_act_evaluation()
                        else:
                            # Terminal
                            self.handle_terminal_evaluation()

                else:
                    if self.terminals[n] == -1:
                        # Initial act
                        self.handle_act(parallel=n)
                    else:
                        # Observe
                        self.handle_observe(parallel=n)
                        if self.terminals[n] == 0:
                            # Act
                            self.handle_act(parallel=n)
                        else:
                            # Terminal
                            self.handle_terminal(parallel=n)

            self.prev_terminals = list(self.terminals)

            # Sync_episodes: Reset if all episodes terminated
            if self.sync_episodes and all(terminal > 0 for terminal in self.terminals):
                num_episodes_left = self.num_episodes - self.episodes
                if self.num_vectorized is None:
                    num_noneval_environments = self.num_environments - int(self.evaluation_run)
                    for n in range(min(num_noneval_environments, num_episodes_left)):
                        self.prev_terminals[n] = -1
                        self.environments[n].start_reset()
                    if self.evaluation_run and num_episodes_left > 0:
                        self.prev_terminals[-1] = -1
                        self.environments[-1].start_reset()
                elif num_episodes_left > 0:
                    if self.environments[0].is_vectorizable():
                        parallel, states = self.environments[0].reset(
                            num_parallel=min(num_episodes_left, self.num_vectorized)
                        )
                    else:
                        parallel, states = self.environments[0].reset()
                    for i, n in enumerate(parallel):
                        self.states[n] = states[i]
                        self.prev_terminals[n] = -2
                else:
                    self.prev_terminals = list()

            # Sleep if no environment was ready
            if no_environment_ready:
                time.sleep(self.num_sleep_secs)

    def handle_act(self, parallel):
        if self.batch_agent_calls:
            if self.num_vectorized is None:
                self.environments[parallel].start_execute(actions=self.actions[parallel])

        else:
            agent_start = time.time()
            actions = self.agent.act(states=self.states[parallel], parallel=parallel)
            self.episode_agent_second[parallel] += time.time() - agent_start

            self.environments[parallel].start_execute(actions=actions)

        # Update episode statistics
        self.episode_timestep[parallel] += 1

        # Maximum number of timesteps or timestep callback (after counter increment!)
        self.timesteps += 1
        if ((
            self.episode_timestep[parallel] % self.callback_timestep_frequency == 0 and
            not self.callback(self, parallel)
        ) or self.timesteps >= self.num_timesteps):
            self.terminate = 2

    def handle_act_joint(self):
        parallel = [
            n for n in range(self.num_environments - int(self.evaluation_run))
            if self.terminals[n] <= 0
        ]
        if len(parallel) > 0:
            agent_start = time.time()
            self.actions = self.agent.act(
                states=[self.states[p] for p in parallel], parallel=parallel
            )
            agent_second = (time.time() - agent_start) / len(parallel)
            for p in parallel:
                self.episode_agent_second[p] += agent_second
            if self.num_vectorized is None:
                self.actions = [
                    self.actions[parallel.index(n)] if n in parallel else None
                    for n in range(self.num_environments)
                ]

        if self.evaluation_run and self.terminals[-1] <= 0:
            assert self.num_vectorized is None
            agent_start = time.time()
            self.actions[-1], self.evaluation_internals = self.agent.act(
                states=self.states[-1], internals=self.evaluation_internals, independent=True,
                deterministic=True
            )
            self.episode_agent_second[-1] += time.time() - agent_start

    def handle_act_evaluation(self):
        if self.batch_agent_calls:
            assert self.num_vectorized is None
            actions = self.actions[-1]

        else:
            agent_start = time.time()
            actions, self.evaluation_internals = self.agent.act(
                states=self.states[-1], internals=self.evaluation_internals, independent=True,
                deterministic=True
            )
            self.evaluation_agent_second += time.time() - agent_start

        self.environments[-1].start_execute(actions=actions)

        # Update episode statistics
        self.episode_timestep[-1] += 1

        # Maximum number of timesteps or timestep callback (after counter increment!)
        if self.evaluation_run and self.num_environments == 1:
            self.timesteps += 1
            if ((
                self.episode_timestep[-1] % self.callback_timestep_frequency == 0 and
                not self.callback(self, -1)
            ) or self.timesteps >= self.num_timesteps):
                self.terminate = 2

    def handle_observe(self, parallel):
        # Update episode statistics
        self.episode_return[parallel] += self.rewards[parallel]

        # Not terminal but finished
        if self.terminals[parallel] == 0 and self.terminate == 2:
            self.terminals[parallel] = 2

        # Observe unless batch_agent_calls
        if not self.batch_agent_calls:
            agent_start = time.time()
            updated = self.agent.observe(
                terminal=self.terminals[parallel], reward=self.rewards[parallel], parallel=parallel
            )
            self.episode_agent_second[parallel] += time.time() - agent_start
            self.updates += int(updated)

        # Maximum number of updates (after counter increment!)
        if self.updates >= self.num_updates:
            self.terminate = 2

    def handle_observe_joint(self):
        parallel = [
            n for n in range(self.num_environments - int(self.evaluation_run))
            if self.prev_terminals[n] <= 0 and self.terminals[n] >= 0
        ]
        if len(parallel) > 0:
            agent_start = time.time()
            updated = self.agent.observe(
                terminal=[self.terminals[p] for p in parallel],
                reward=[self.rewards[p] for p in parallel], parallel=parallel
            )
            agent_second = (time.time() - agent_start) / len(parallel)
            for p in parallel:
                self.episode_agent_second[p] += agent_second
            self.updates += updated

    def handle_observe_evaluation(self):
        # Update episode statistics
        self.episode_return[-1] += self.rewards[-1]

        # Reset agent if terminal
        if self.terminals[-1] > 0 or self.terminate == 2:
            agent_start = time.time()
            self.evaluation_agent_second += time.time() - agent_start

    def handle_terminal(self, parallel):
        # Update experiment statistics
        if self.num_vectorized is None:
            actual_episode_return = self.environments[parallel].episode_return()
        else:
            actual_episode_return = self.environments[0].episode_return(parallel=parallel)
        if actual_episode_return is None:
            self.episode_returns.append(self.episode_return[parallel])
        else:
            self.episode_returns.append(actual_episode_return)
        self.episode_timesteps.append(self.episode_timestep[parallel])
        self.episode_seconds.append(time.time() - self.episode_start[parallel])
        self.episode_agent_seconds.append(self.episode_agent_second[parallel])
        if self.is_environment_remote:
            self.episode_env_seconds.append(self.environments[parallel]._episode_seconds)

        # Maximum number of episodes or episode callback (after counter increment!)
        self.episodes += 1
        if self.terminate == 0 and ((
            self.episodes % self.callback_episode_frequency == 0 and
            not self.callback(self, parallel)
        ) or self.episodes >= self.num_episodes):
            self.terminate = 1

        # Reset episode statistics
        self.episode_return[parallel] = 0.0
        self.episode_timestep[parallel] = 0
        self.episode_agent_second[parallel] = 0.0
        self.episode_start[parallel] = time.time()

        # Reset environment
        if self.terminate == 0 and not self.sync_episodes:
            self.terminals[parallel] = -1
            self.environments[parallel].start_reset()

    def handle_terminal_evaluation(self):
        # Update experiment statistics
        if self.num_vectorized is None:
            actual_episode_return = self.environments[-1].episode_return()
        else:
            actual_episode_return = self.environments[0].episode_return(parallel=-1)
        if actual_episode_return is None:
            self.evaluation_returns.append(self.episode_return[-1])
        else:
            self.evaluation_returns.append(actual_episode_return)
        self.evaluation_timesteps.append(self.episode_timestep[-1])
        self.evaluation_seconds.append(time.time() - self.evaluation_start)
        self.evaluation_agent_seconds.append(self.evaluation_agent_second)
        if self.is_environment_remote:
            self.evaluation_env_seconds.append(self.environments[-1]._episode_seconds)

        # Evaluation callback
        if self.save_best_agent is not None:
            evaluation_score = self.evaluation_callback(self)
            assert isinstance(evaluation_score, float)
            if self.best_evaluation_score is None:
                self.best_evaluation_score = evaluation_score
            elif evaluation_score > self.best_evaluation_score:
                self.best_evaluation_score = evaluation_score
                self.agent.save(
                    directory=self.save_best_agent, filename='best-model', append=None
                )
        else:
            self.evaluation_callback(self)

        # Maximum number of episodes or episode callback (after counter increment!)
        if self.evaluation_run and self.num_environments == 1:
            self.episodes += 1
            if self.terminate == 0 and ((
                self.episodes % self.callback_episode_frequency == 0 and
                not self.callback(self, -1)
            ) or self.episodes >= self.num_episodes):
                self.terminate = 1

        # Reset episode statistics
        self.episode_return[-1] = 0.0
        self.episode_timestep[-1] = 0
        self.evaluation_agent_second = 0.0
        self.evaluation_start = time.time()

        # Reset environment
        if self.terminate == 0 and not self.sync_episodes:
            self.terminals[-1] = 0
            self.environments[-1].start_reset()
            self.evaluation_internals = self.agent.initial_internals()
