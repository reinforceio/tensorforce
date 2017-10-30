from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce import Configuration
from tensorforce.agents import NAFAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestNAFAgent(BaseAgentTest, unittest.TestCase):

    agent = NAFAgent
    deterministic = True

    config = Configuration(
        memory=dict(
            type='replay',
            capacity=1000
        ),
        batch_size=8,
        first_update=10,
        target_sync_frequency=10
    )

    exclude_bool = True
    exclude_int = True
    exclude_bounded = True
    exclude_multi = True
    exclude_lstm = True
