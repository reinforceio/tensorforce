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

from socket import SHUT_RDWR, socket as Socket
import time

import msgpack
import msgpack_numpy

from tensorforce import TensorforceError
from tensorforce.environments import RemoteEnvironment


msgpack_numpy.patch()


class SocketEnvironment(RemoteEnvironment):
    """
    An earlier version of this code (#626) was originally developed as part of the following work:

    Rabault, J., Kuhnle, A (2019). Accelerating Deep Reinforcement Leaning strategies of Flow
    Control through a multi-environment approach. Physics of Fluids.
    """

    MAX_BYTES = 4096

    @classmethod
    def remote(cls, port, environment, max_episode_timesteps=None, reward_shaping=None, **kwargs):
        socket = Socket()
        socket.bind(('', port))
        socket.listen(1)
        connection, address = socket.accept()
        socket.close()
        super().remote(
            connection=connection, environment=environment,
            max_episode_timesteps=max_episode_timesteps, reward_shaping=reward_shaping, **kwargs
        )

    @classmethod
    def proxy_send(cls, connection, function, kwargs):
        str_function = function.encode()
        num_bytes = len(str_function)
        str_num_bytes = '{:08d}'.format(num_bytes).encode()
        bytes_sent = connection.send(str_num_bytes + str_function)
        if bytes_sent != num_bytes + 8:
            raise TensorforceError.unexpected()

        str_kwargs = msgpack.packb(o=kwargs)
        num_bytes = len(str_kwargs)
        str_num_bytes = '{:08d}'.format(num_bytes).encode()
        bytes_sent = connection.send(str_num_bytes + str_kwargs)
        if bytes_sent != num_bytes + 8:
            raise TensorforceError.unexpected()

    @classmethod
    def proxy_receive(cls, connection):
        str_success = connection.recv(1)
        if len(str_success) != 1 or (str_success != b'0' and str_success != b'1'):
            raise TensorforceError.unexpected()
        success = (str_success == b'1')

        str_num_bytes = connection.recv(8)
        if len(str_num_bytes) != 8:
            raise TensorforceError.unexpected()
        num_bytes = int(str_num_bytes.decode())
        str_result = b''
        for n in range(num_bytes // cls.MAX_BYTES):
            str_result += connection.recv(cls.MAX_BYTES)
            if len(str_result) != (n + 1) * cls.MAX_BYTES:
                raise TensorforceError.unexpected()
        str_result += connection.recv(num_bytes % cls.MAX_BYTES)
        if len(str_result) != num_bytes:
            raise TensorforceError.unexpected()
        result = msgpack.unpackb(packed=str_result)

        return success, result

    @classmethod
    def proxy_close(cls, connection):
        connection.shutdown(SHUT_RDWR)
        connection.close()

    @classmethod
    def remote_send(cls, connection, success, result):
        str_success = b'1' if success else b'0'
        bytes_sent = connection.send(str_success)
        if bytes_sent != 1:
            raise TensorforceError.unexpected()

        str_result = msgpack.packb(o=result)
        num_bytes = len(str_result)
        str_num_bytes = '{:08d}'.format(num_bytes).encode()
        bytes_sent = connection.send(str_num_bytes + str_result)
        if bytes_sent != num_bytes + 8:
            raise TensorforceError.unexpected()

    @classmethod
    def remote_receive(cls, connection):
        str_num_bytes = connection.recv(8)
        if len(str_num_bytes) != 8:
            raise TensorforceError.unexpected()
        num_bytes = int(str_num_bytes.decode())
        str_function = b''
        for n in range(num_bytes // cls.MAX_BYTES):
            str_function += connection.recv(cls.MAX_BYTES)
            if len(str_function) != (n + 1) * cls.MAX_BYTES:
                raise TensorforceError.unexpected()
        str_function += connection.recv(num_bytes % cls.MAX_BYTES)
        if len(str_function) != num_bytes:
            raise TensorforceError.unexpected()
        function = str_function.decode()

        str_num_bytes = connection.recv(8)
        if len(str_num_bytes) != 8:
            raise TensorforceError.unexpected()
        num_bytes = int(str_num_bytes.decode())
        str_kwargs = b''
        for n in range(num_bytes // cls.MAX_BYTES):
            str_kwargs += connection.recv(cls.MAX_BYTES)
            if len(str_kwargs) != n * cls.MAX_BYTES:
                raise TensorforceError.unexpected()
        str_kwargs += connection.recv(num_bytes % cls.MAX_BYTES)
        if len(str_kwargs) != num_bytes:
            raise TensorforceError.unexpected()
        kwargs = msgpack.unpackb(packed=str_kwargs)

        return function, kwargs

    @classmethod
    def remote_close(cls, connection):
        connection.shutdown(SHUT_RDWR)
        connection.close()

    def __init__(self, host, port, blocking=False):
        socket = Socket()
        for _ in range(100):  # TODO: 10sec timeout, not configurable
            try:
                socket.connect((host, port))
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
        else:
            raise TensorforceError("Remote socket connection could not be established.")
        super().__init__(connection=socket, blocking=blocking)
