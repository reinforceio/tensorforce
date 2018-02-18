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

from tensorforce import util
from tensorforce.core.preprocessing import Preprocessor
import tensorforce.core.preprocessing


class PreprocessorStack(object):
    """
    A class to handle many Preprocessor objects applied in a sequence to some state. For example: An image
    tensor as state signal could be re-sized first, then grayscaled, then normalized.
    """
    def __init__(self):
        self.preprocessors = list()

    def reset(self):
        """
        Calls `reset` on all our Preprocessor objects.

        Returns:
            A list of tensors to be fetched.
        """
        fetches = []
        for processor in self.preprocessors:
            fetches.extend(processor.reset() or [])
        return fetches

    def process(self, tensor):
        """
        Process state.

        Args:
            tensor: tensor to process

        Returns: processed state

        """
        for processor in self.preprocessors:
            tensor = processor.process(tensor=tensor)
        return tensor

    def processed_shape(self, shape):
        """
        Shape of preprocessed state given original shape.

        Args:
            shape: original state shape

        Returns: processed state shape
        """
        for processor in self.preprocessors:
            shape = processor.processed_shape(shape=shape)
        return shape

    def get_variables(self):
        return [variable for preprocessor in self.preprocessors for variable in preprocessor.get_variables()]

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a preprocessing stack from a specification dict.
        """
        if isinstance(spec, dict):
            spec = [spec]

        stack = PreprocessorStack()
        for spec in spec:
            preprocessor = util.get_object(
                obj=spec,
                predefined_objects=tensorforce.core.preprocessing.preprocessors,
                kwargs=kwargs
            )
            assert isinstance(preprocessor, Preprocessor)
            stack.preprocessors.append(preprocessor)

        return stack
