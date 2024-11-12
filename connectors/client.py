import socket
import json

from shared.random_player import RandomPlayer


class Client:
    def __init__(self, player: RandomPlayer, server_ip, port, current_state=None, timeout=60):
        self.player = player
        self.timeout = timeout
        self.server_ip = server_ip
        self.port = port
        self.current_state = current_state
        self.player_socket = self.connect(player, server_ip, port)

    def connect(self, player, server_ip, port) -> socket.socket:
        try:
            player_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            player_socket.settimeout(60)

            print(f"Connecting to {server_ip}:{port} as {player}...")
            player_socket.connect((server_ip, port))
            print("Connection established!")

            return player_socket

        except socket.timeout:
            print(f"Connection to {server_ip}:{port} timed out.")
        except socket.gaierror:
            print(f"Address-related error connecting to {server_ip}:{port}.")
        except ConnectionRefusedError:
            print(f"Connection refused by the server at {server_ip}:{port}.")
        except socket.error as e:
            print(f"Failed to connect to {server_ip}:{port} due to: {e}")

    def send_name(self):
        try:
            name_bytes = (self.player.name + '\n').encode('utf-8')

            self.player_socket.sendall(name_bytes)

            print(f"Sent name '{self.player.name}' to server.")
        except socket.error as e:
            print(f"Failed to send name: {e}")

    def send_move(self, action):
        try:
            action_json = json.dumps(action)
            action_bytes = action_json.encode('utf-8') + b'\n'

            self.player_socket.sendall(action_bytes)

            print(f"Sent move '{action_json}' to server.")
        except (socket.error, json.JSONDecodeError) as e:
            print(f"Failed to send move: {e}")

    def compute_move(self):
        return {"move": "some_move"}  # Provisory mocked action

    def read_state(self):
        try:
            state_data = self.player_socket.recv(1024)
            if not state_data:
                print("Received empty response from server.")
                return None

            state_json = state_data.decode('utf-8')
            self.current_state = json.loads(state_json)
            print(f"Received state '{self.current_state}' from server.")
        except (socket.error, json.JSONDecodeError) as e:
            print(f"Failed to read state: {e}")
            return None

