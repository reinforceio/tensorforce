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

from tensorforce import TensorforceError
from tensorforce.core import tf_function, tf_util
from tensorforce.core.parameters import Parameter


class Constant(Parameter):
    """
    Constant hyperparameter  (specification key: `constant`).

    Args:
        value (float | int | bool): Constant hyperparameter value
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    # Argument 'value' first for default specification
    def __init__(self, value, *, name=None, dtype=None, min_value=None, max_value=None):
        if isinstance(value, bool):
            if dtype != 'bool':
                raise TensorforceError.type(name='Constant', argument='value', dtype=type(value))
        elif isinstance(value, int):
            if dtype != 'int':
                raise TensorforceError.type(name='Constant', argument='value', dtype=type(value))
        elif isinstance(value, float):
            if dtype != 'float':
                raise TensorforceError.type(name='Constant', argument='value', dtype=type(value))
        else:
            raise TensorforceError.unexpected()
        if min_value is not None and value < min_value:
            raise TensorforceError.value(
                name='Constant', argument='value', value=value,
                hint='< {} lower bound'.format(min_value)
            )
        if max_value is not None and value > max_value:
            raise TensorforceError.value(
                name='Constant', argument='value', value=value,
                hint='> {} upper bound'.format(max_value)
            )

        self.constant_value = value

        super().__init__(name=name, dtype=dtype, min_value=min_value, max_value=max_value)

    def min_value(self):
        return self.constant_value

    def max_value(self):
        return self.constant_value

    def final_value(self):
        return self.constant_value

    def initialize(self):
        super(Parameter, self).initialize()

    @tf_function(num_args=0)
    def value(self):
        return tf_util.constant(value=self.constant_value, dtype=self.spec.type)
