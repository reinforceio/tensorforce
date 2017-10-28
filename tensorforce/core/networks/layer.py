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
Collection of custom layer implementations. We prefer not to use contrib-layers to retain full control over shapes and internal states.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import sqrt

import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util
import tensorforce.core.networks


class Layer(object):
    """
    Base class for network layers.
    """

    def __init__(self, num_internals=0, scope='layer', summary_labels=None):
        self.num_internals = num_internals
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.all_variables = dict()
        self.summaries = list()

        with tf.name_scope(name=scope):
            def custom_getter(getter, name, registered=False, **kwargs):
                variable = getter(name=name, registered=True, **kwargs)
                if not registered:
                    self.all_variables[name] = variable
                    if kwargs.get('trainable', True):
                        self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        summary = tf.summary.histogram(name=name, values=variable)
                        self.summaries.append(summary)
                return variable

            self.apply = tf.make_template(
                name_='apply',
                func_=self.tf_apply,
                custom_getter_=custom_getter
            )
            self.regularization_loss = tf.make_template(
                name_='regularization-loss',
                func_=self.tf_regularization_loss,
                custom_getter_=custom_getter
            )

    def tf_apply(self, x):
        """
        Creates the TensorFlow operations for applying the layer to the given input.

        Args:
            x: Layer input tensor.

        Returns:
            Layer output tensor.
        """
        raise NotImplementedError

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the layer regularization loss.

        Returns:
            Regularization loss tensor.
        """
        return None

    def internal_inputs(self):
        """
        Returns the TensorFlow placeholders for internal state inputs.

        Returns:
            List of internal state input placeholders.
        """
        return list()

    def internal_inits(self):
        """
        Returns the TensorFlow tensors for internal state initializations.

        Returns:
            List of internal state initialization tensors.
        """
        return list()

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the layer.

        Returns:
            List of variables.
        """
        if include_non_trainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the layer.

        Returns:
            List of summaries.
        """
        return self.summaries

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a layer from a specification dict.
        """
        layer = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.networks.layers,
            kwargs=kwargs
        )
        assert isinstance(layer, Layer)
        return layer


class Flatten(Layer):
    """
    Flatten layer reshaping the input.
    """

    def __init__(self, scope='flatten', summary_labels=()):
        super(Flatten, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        return tf.reshape(tensor=x, shape=(-1, util.prod(util.shape(x)[1:])))


class Nonlinearity(Layer):
    """
    Non-linearity layer applying a non-linear transformation.
    """

    def __init__(self, name='relu', scope='nonlinearity', summary_labels=()):
        """
        Non-linearity layer.

        Args:
            name: Non-linearity name, one of 'elu', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'tanh' or 'none'.
        """
        self.name = name
        super(Nonlinearity, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        if self.name == 'elu':
            x = tf.nn.elu(features=x)

        elif self.name == 'relu':
            x = tf.nn.relu(features=x)
            if 'relu' in self.summary_labels:
                non_zero = tf.cast(x=tf.count_nonzero(input_tensor=x), dtype=tf.float32)
                size = tf.cast(x=tf.reduce_prod(input_tensor=tf.shape(input=x)), dtype=tf.float32)
                summary = tf.summary.scalar(name='relu', tensor=(non_zero / size))
                self.summaries.append(summary)

        elif self.name == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            negative = alpha * tf.nn.elu(features=x)
            x = scale * tf.where(condition=(x >= 0.0), x=x, y=negative)

        elif self.name == 'sigmoid':
            x = tf.sigmoid(x=x)

        elif self.name == 'softmax':
            x = tf.nn.softmax(logits=x)

        elif self.name == 'softplus':
            x = tf.nn.softplus(features=x)

        elif self.name == 'tanh':
            x = tf.nn.tanh(x=x)

        elif self.name == 'none':
            x = tf.identity(input=x)            

        else:
            raise TensorForceError('Invalid non-linearity: {}'.format(self.name))

        return x


class Linear(Layer):
    """
    Linear fully-connected layer.
    """

    def __init__(self, size, weights=None, bias=True, l2_regularization=0.0, l1_regularization=0.0, scope='linear', summary_labels=()):
        """
        Linear layer.

        Args:
            size: Layer size.
            weights: Weight initialization, random if None.
            bias: Bias initialization, random if True, no bias added if False.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        self.size = size
        self.weights_init = weights
        self.bias_init = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        super(Linear, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        if util.rank(x) != 2:
            raise TensorForceError('Invalid input rank for linear layer: {},'
                                   ' must be 2.'.format(util.rank(x)))

        weights_shape = (x.shape[1].value, self.size)

        if self.weights_init is None:
            stddev = min(0.1, sqrt(2.0 / (x.shape[1].value + self.size)))
            self.weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)

        elif isinstance(self.weights_init, float):
            if self.weights == 0.0:
                self.weights_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.weights_init = tf.constant_initializer(value=self.weights, dtype=tf.float32)

        elif isinstance(self.weights_init, list):
            self.weights_init = np.asarray(self.weights_init, dtype=np.float32)
            if self.weights.shape != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights.shape, weights_shape)
                )
            self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, np.ndarray):
            if self.weights.shape != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights.shape, weights_shape)
                )
            self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, tf.Tensor):
            if util.shape(self.weights_init) != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights.shape, weights_shape)
                )

        bias_shape = (self.size,)

        if isinstance(self.bias_init, bool):
            if self.bias_init:
                self.bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.bias_init = None

        elif isinstance(self.bias_init, float):
            if self.bias_init == 0.0:
                self.bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias, list):
            self.bias_init = np.asarray(self.bias_init, dtype=np.float32)
            if self.bias_init.shape != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias.shape, bias_shape)
                )
            self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias, np.ndarray):
            if self.bias_init.shape != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias.shape, bias_shape)
                )
            self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, tf.Tensor):
            if util.shape(self.bias_init) != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias.shape, bias_shape)
                )

        if isinstance(self.weights_init, tf.Tensor):
            self.weights = self.weights_init
        else:
            self.weights = tf.get_variable(name='W', shape=weights_shape, dtype=tf.float32, initializer=self.weights_init)
        x = tf.matmul(a=x, b=self.weights)

        if self.bias_init is None:
            self.bias = None

        else:
            if isinstance(self.bias_init, tf.Tensor):
                self.bias = self.bias_init
            else:
                self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=self.bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        return x

    def tf_regularization_losses(self):
        if super(Linear, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(Linear, self).tf_regularization_loss()]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.weights))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.weights)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None


class Dense(Layer):
    """
    Dense layer, i.e. linear fully connected layer with subsequent non-linearity.
    """

    def __init__(
        self,
        size,
        bias=True,
        activation='tanh',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='dense',
        summary_labels=()
    ):
        """
        Dense layer.

        Args:
            size: Layer size.
            bias: If true, bias is added.
            activation: Type of nonlinearity.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        self.linear = Linear(size=size, bias=bias, l2_regularization=l2_regularization, l1_regularization=l1_regularization, summary_labels=summary_labels)
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Dense, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        x = self.linear.apply(x=x)
        x = self.nonlinearity.apply(x=x)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        if super(Dense, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(Dense, self).tf_regularization_loss()]

        if self.linear.regularization_loss() is not None:
            losses.append(self.linear.regularization_loss())

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Dense, self).get_variables(include_non_trainable=include_non_trainable)

        linear_variables = self.linear.get_variables(include_non_trainable=include_non_trainable)

        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + linear_variables + nonlinearity_variables


class Dueling(Layer):
    """
    Dueling layer, i.e. Duel pipelines for Exp & Adv to help with stability
    """

    def __init__(
        self,
        size,
        bias=False,
        activation='none',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='dueling',
        summary_labels=()
    ):
        """
        Dueling layer.

        [Dueling Networks] (https://arxiv.org/pdf/1511.06581.pdf)
        Implement Y = Expectation[x] + (Advantage[x] - Mean(Advantage[x]))

        Args:
            size: Layer size.
            bias: If true, bias is added.
            activation: Type of nonlinearity.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        # Expectation is broadcast back over advantage values so output is of size 1 
        self.linear_exp = Linear(size=1, bias=bias, l2_regularization=l2_regularization, l1_regularization=l1_regularization, summary_labels=summary_labels)
        self.linear_adv = Linear(size=size, bias=bias, l2_regularization=l2_regularization, l1_regularization=l1_regularization, summary_labels=summary_labels)
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Dueling, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        expectation = self.linear_exp.apply(x=x)
        advantage   = self.linear_adv.apply(x=x)

        x = expectation + advantage - tf.reduce_mean(advantage,axis=1,keep_dims=True)

        x = self.nonlinearity.apply(x=x)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        if super(Dueling, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(Dueling, self).tf_regularization_loss()]

        if self.linear_exp.regularization_loss() is not None:
            losses.append(self.linear_exp.regularization_loss())

        if self.linear_adv.regularization_loss() is not None:
            losses.append(self.linear_adv.regularization_loss())            

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Dueling, self).get_variables(include_non_trainable=include_non_trainable)

        linear_variables_exp = self.linear_exp.get_variables(include_non_trainable=include_non_trainable)
        linear_variables_adv = self.linear_adv.get_variables(include_non_trainable=include_non_trainable)

        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + linear_variables_exp + linear_variables_adv + nonlinearity_variables

class Conv1d(Layer):
    """
    1-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='conv1d',
        summary_labels=()
    ):
        """
        1D convolutional layer.

        Args:
            size: Number of filters
            window: Convolution window size
            stride: Convolution stride
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        self.window = window
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Conv1d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        if util.rank(x) != 3:
            raise TensorForceError('Invalid input rank for conv1d layer: {}, must be 3'.format(util.rank(x)))

        filters_shape = (self.window, x.shape[2].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        x = tf.nn.conv1d(input=x, filter=self.filters, strides=self.stride, padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        if super(Conv1d, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(Conv1d, self).tf_regularization_loss()]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Conv1d, self).get_variables(include_non_trainable=include_non_trainable)

        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + nonlinearity_variables


class Conv2d(Layer):
    """
    2-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='conv2d',
        summary_labels=()
    ):
        """
        2D convolutional layer.

        Args:
            size: Number of filters
            window: Convolution window size, either an integer or pair of integers.
            stride: Convolution stride, either an integer or pair of integers.
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        if isinstance(window, int):
            self.window = (window, window)
        elif len(window) != 2:
            raise TensorForceError('Invalid window {} for conv2d layer, must be of size 2'.format(window))
        else:
            self.window = tuple(window)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Conv2d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x):
        if util.rank(x) != 4:
            raise TensorForceError('Invalid input rank for conv2d layer: {}, must be 4'.format(util.rank(x)))

        filters_shape = self.window + (x.shape[3].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        stride_h, stride_w = self.stride if type(self.stride) is tuple else (self.stride, self.stride)
        x = tf.nn.conv2d(input=x, filter=self.filters, strides=(1, stride_h, stride_w, 1), padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        if super(Conv2d, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [super(Conv2d, self).tf_regularization_loss()]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Conv2d, self).get_variables(include_non_trainable=include_non_trainable)

        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + nonlinearity_variables


class Lstm(Layer):
    """
    Long short-term memory layer.
    """

    def __init__(self, size, dropout=None, scope='lstm', summary_labels=()):
        """
        LSTM layer.

        Args:
            size: LSTM size.
            dropout: Dropout rate.
        """
        self.size = size
        self.dropout = dropout
        super(Lstm, self).__init__(num_internals=1, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, state):
        if util.rank(x) != 2:
            raise TensorForceError('Invalid input rank for lstm layer: {}, must be 2.'.format(util.rank(x)))

        c = state[:, 0, :]
        h = state[:, 1, :]
        state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)

        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.size)
        if self.dropout is not None:
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=(1.0 - self.dropout))

        x, state = self.lstm_cell(inputs=x, state=state)

        internal_output = tf.stack(values=(state.c, state.h), axis=1)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x, (internal_output,)

    def internal_inputs(self):
        return super(Lstm, self).internal_inputs() + [tf.placeholder(dtype=tf.float32, shape=(None, 2, self.size))]

    def internal_inits(self):
        return super(Lstm, self).internal_inits() + [np.zeros(shape=(2, self.size))]
