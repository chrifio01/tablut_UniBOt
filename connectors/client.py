"""
The `Client` class represents a client that connects to the server to play the game.
It includes methods for sending the player's name, move, and receiving the current game state.
Attributes:
    player (AbstractPlayer): The player that connects to the server.
    timeout (int): The time limit for the connection.
    server_ip (str): The IP address of the server.
    port (int): The port number of the server.
    current_state (State): The current game state visible to the player.
    player_socket (socket.socket): The socket connection to the server.
Methods:
    connect(player, server_ip, port) -> socket.socket: Connects to the server using the player's socket.
    send_name(): Sends the player's name to the server.
    send_move(action): Sends the player's move to the server.
    compute_move() -> dict: Computes the player's move.
    read_state(): Reads the current game state from the server.
"""

import socket
import struct
import json

from shared.loggers import logger

from shared.utils import AbstractPlayer, strp_state, state_decoder, Turn


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
        player_socket (socket.socket): The socket connection to the server.

    Methods:
        connect() -> socket.socket: Connects to the server using the player's socket.
        send_name(): Sends the player's name to the server.
        send_move(action): Sends the player's move to the server.
        compute_move() -> dict: Computes the player's move.
        read_state(): Reads the current game state from the server.
    """

    def __init__(self, player: AbstractPlayer, server_ip: str, port: int, current_state=None, timeout: int = 60):
        """
        Initializes a Client instance.

        Args:
            player (AbstractPlayer): The player instance that connects to the server.
            server_ip (str): The IP address of the server.
            port (int): The port number of the server.
            current_state (optional): The current game state visible to the player.
            timeout (int, optional): The time limit for the connection. Defaults to 60 seconds.
        """
        self.player = player
        self.timeout = timeout
        self.server_ip = server_ip
        self.port = port
        self.current_state = strp_state(current_state) if current_state else None
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

        Returns:
            socket.socket: The socket connection object.
        """
        try:
            logger.debug(f"Connecting to {self.server_ip}:{self.port} as {self.player.name}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_ip, self.port))
            logger.debug("Connection established!")
        except socket.timeout:
            logger.debug(f"Connection to {self.server_ip}:{self.port} timed out.")
        except socket.gaierror:
            logger.debug(f"Address-related error connecting to {self.server_ip}:{self.port}.")
        except ConnectionRefusedError:
            logger.debug(f"Connection refused by the server at {self.server_ip}:{self.port}.")
        except socket.error as e:
            logger.debug(f"Failed to connect to {self.server_ip}:{self.port} due to: {e}")

    def _send_name(self):
        """
        Sends the player's name to the server.
        """
        try:
            self.socket.send(struct.pack('>i', len(self.player.name)))
            self.socket.send(self.player.name.encode())
            logger.debug(f"Declared name '{self.player.name}' to server.")
        except socket.error as e:
            logger.debug(f"Failed to send name to the server: {e}")

    def _send_move(self, action):
        """
        Sends the player's move to the server.

        Args:
            action (dict): The player's move as a dictionary.
        """
        try:
            action_str = str(action)
            self.socket.send(struct.pack('>i', len(action_str)))
            self.socket.send(action_str.encode())
        except socket.error as e:
            logger.debug(f"Failed to send move to the server: {e}")

    def _compute_move(self) -> dict:
        """
        Computes the player's move.

        Returns:
            dict: The player's computed move.
        """
        return self.player.fit(self.current_state)

    def _read_state(self):
        """
        Reads the current game state from the server.

        Returns:
            State: The current game state visible to the player.
        """
        try:
            len_bytes = struct.unpack('>i', self._recvall(4))[0]
            current_state_server_bytes = self.socket.recv(len_bytes)
            self.current_state = json.loads(current_state_server_bytes, object_hook=state_decoder)
        except (socket.error, json.JSONDecodeError) as e:
            logger.debug(f"Failed to read or decode the server response: {e}")
            raise RuntimeError('Failed to decode server response')

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
                logger.debug(f"Game ended...\nResult: {self.current_state.turn.value}")
                return

            if self.current_state.turn.value == self.player.color.value:
                logger.debug("Calculating move...")
                action = self._compute_move()
                logger.debug(f"Sending move:\n{action}")
                self._send_move(action)
                logger.debug("Action sent")
            else:
                logger.debug("Waiting for opponent's move...")
