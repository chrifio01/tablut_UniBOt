"""
The `Client` class represents a client that connects to the server to play the game.
It includes methods for sending the player's name, move, and receiving the current game state.
Attributes:
    player (AbstractPlayer): The player that connects to the server.
    timeout (int): The time limit for the connection.
    server_ip (str): The IP address of the server.
    port (int): The port number of the server.
    current_state (State): The current game state visible to the player.
    socket (socket.socket): The socket connection to the server.
Methods:
    connect(player, server_ip, port) -> socket.socket:
            Connects to the server using the player's socket.
    send_name(): Sends the player's name to the server.
    send_move(action): Sends the player's move to the server.
    compute_move() -> dict: Computes the player's move.
    read_state(): Reads the current game state from the server.
"""

import socket
import struct
import json
from typing import Dict

from shared.loggers import logger

from shared.utils import AbstractPlayer, strp_state, state_decoder, Turn, Action

class Client:
    """
    The `Client` class represents a client that connects to the server to play the game.
    It includes methods for sending the player's name, move, and receiving the current game state.

    Attributes:
        player (AbstractPlayer): The player that connects to the server.
        timeout (int): The time limit for the connection.
        server_ip (str): The IP address of the server.
        port (int): The port number of the server.
        current_state (State): The current game state visible to the player.
        socket (socket.socket): The socket connection to the server.

    Methods:
        connect(): Connects to the server using the player's socket.
        send_name(): Sends the player's name to the server.
        send_move(action): Sends the player's move to the server.
        compute_move() -> Action: Computes the player's move.
        read_state(): Reads the current game state from the server.
    """

    def __init__(self, *, player: AbstractPlayer, settings: Dict[str, any]):
        """
        Initializes a Client instance.

        Args:
            player (AbstractPlayer): The player instance that connects to the server.
            settings (Dict[str, any]): A dictionary containing the configuration settings.
                The dictionary must include:
                - 'current_state' (str): The current game state visible to the player.
                - 'timeout' (int): The time limit for the connection in seconds.
                - 'server_ip' (str): The IP address of the server.
                - 'port' (int): The port number of the server.
        Notes:
            All arguments are keyword-only. The `settings` dictionary is required
            and must contain the keys 'server_ip', 'port', 'current_state' and 'timeout'.
            These parameters are necessary for the Client to function properly.
        """
        missing_keys = [key for key in ['server_ip', 'port', 'current_state', 'timeout']
                        if key not in settings]
        if missing_keys:
            raise RuntimeError(f"Missing required settings: {missing_keys}")
        self.player = player
        self.server_ip = settings['server_ip']
        self.port = settings['port']
        self.current_state = strp_state(settings['current_state']) \
            if 'current_state' in settings else None
        self.timeout = settings['timeout']
        self._connect()

    def __del__(self):
        """
        Closes the socket connection when the Client instance is deleted.
        """
        if self.socket:
            self.socket.close()

    def _connect(self):
        """
        Establishes a connection to the server.
        """
        try:
            logger.debug("Connecting to %s:%d as %s...", self.server_ip, self.port, self.player.name)
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_ip, self.port))
            logger.debug("Connection established!")
        except socket.timeout as exc:
            logger.error("Connection to %s:%s timed out.", self.server_ip, self.port)
            raise RuntimeError(f"Connection to {self.server_ip}:{self.port} timed out.") from exc
        except socket.gaierror as exc:
            logger.error("Address-related error connecting to %s:%s.", self.server_ip, self.port)
            raise RuntimeError(f"Address-related error connecting to {self.server_ip}:{self.port}.") from exc
        except ConnectionRefusedError as exc:
            logger.error("Connection refused by the server at %s:%s.", self.server_ip, self.port)
            raise RuntimeError(f"Connection refused by the server at {self.server_ip}:{self.port}.") from exc
        except socket.error as exc:
            logger.error("Failed to connect to %s:%s due to: %s", self.server_ip, self.port, exc)
            raise RuntimeError(f"Failed to connect to {self.server_ip}:{self.port} due to: {exc}") from exc

    def _send_name(self):
        """Sends the player's name to the server."""
        try:
            name_bytes = self.player.name.encode()
            self.socket.send(struct.pack('>i', len(name_bytes)))
            self.socket.send(name_bytes)
            logger.debug("Declared name '%s' to server.", self.player.name)
        except socket.error as exc:
            logger.error("Failed to send name to the server: %s", exc)
            raise RuntimeError(f"Failed to send name to the server: {exc}") from exc

    def _send_move(self, action):
        """
        Sends the player's move to the server.

        Args:
            action (Action): The player's move as an Action.
        """
        try:
            action_str = str(action)
            self.socket.send(struct.pack('>i', len(action_str)))
            self.socket.send(action_str.encode())
        except socket.error as exc:
            logger.error("Failed to send move to the server: %s", exc)
            raise RuntimeError(f"Failed to send move to the server: {exc}") from exc

    def _compute_move(self) -> Action:
        """Computes the player's move and returns it."""
        return self.player.fit(self.current_state)

    def _read_state(self):
        """
        Reads the current game state from the server.

        Returns:
            State: The current game state visible to the player.
        """
        try:
            len_bytes = struct.unpack('>i', self._recvall(4))[0]
            state_data = self._recvall(len_bytes)
            if state_data is None:
                raise RuntimeError("Failed to receive game state data.")
            self.current_state = json.loads(state_data, object_hook=state_decoder)
        except (socket.error, json.JSONDecodeError) as exc:
            logger.error("Failed to read or decode the server response: %s", exc)
            raise RuntimeError("Failed to decode server response.") from exc

    def _recvall(self, n: int) -> bytes:
        """
        Helper function to receive `n` bytes or return None if EOF is hit.

        Args:
            n (int): The number of bytes to receive.

        Returns:
            bytes: The received bytes data.
        """
        data = b''
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def main(self):
        """
        Main loop for the client to handle game state updates and send moves.
        """
        self._send_name()
        while True:
            logger.debug("Reading state...")
            self._read_state()
            logger.debug(self.current_state)

            if self.current_state.turn in (Turn.DRAW, Turn.BLACK_WIN, Turn.WHITE_WIN):
                logger.debug("Game ended...\nResult: %s", self.current_state.turn.value)
                return

            if self.current_state.turn.value == self.player.color.value:
                logger.debug("Calculating move...")
                action = self._compute_move()
                logger.debug("Sending move:\n%s", action)
                self._send_move(action)
                logger.debug("Action sent")
            else:
                logger.debug("Waiting for opponent's move...")
