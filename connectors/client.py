import socket
import json
import os


class Client:
    def __init__(self, player, name, timeout=60):
        self.player = player
        self.name = name  # W or B
        self.timeout = timeout
        self.server_ip = os.getenv("SERVER_IP", "localhost")
        self.port = 5800 if player.lower() == 'w' else 5801  #
        self.current_state = None

        if self.player.lower() not in ('w', 'b'):
            raise ValueError("Player role must be B (BLACK) or W (WHITE)")

        self.player_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.player_socket.settimeout(self.timeout)
        print(f"Connecting to {self.server_ip}:{self.port} as {self.player}")
        self.player_socket.connect((self.server_ip, self.port))
        print("Connection established!")
        self.in_stream = self.player_socket.makefile('rb')
        self.out_stream = self.player_socket.makefile('wb')

    # Send the move to the server
    def send_move(self, action):
        try:
            action_json = json.dumps(action)
            self.out_stream.write(action_json.encode('utf-8') + b'\n')
            self.out_stream.flush()
            print(f"Sent move '{action_json}' to server.")
        except (socket.error, json.JSONDecodeError) as e:
            print(f"Failed to send move: {e}")

    def compute_move(self):
        next_move = "move"  # Provisory mocked action

    def declare_name(self):
        try:
            self.out_stream.write((self.name + '\n').encode('utf-8'))
            self.out_stream.flush()
            print(f"Sent name '{self.name}' to server.")
        except (socket.error, json.JSONDecodeError) as e:
            print(f"Failed to send name: {e}")

    def read_state(self):
        state_json = self.in_stream.readline().decode('utf-8').strip()
        if not state_json:
            print("Received empty response from server.")
            return None

        try:
            self.current_state = json.loads(state_json)
            print(f"Received state '{self.current_state}' from server.")
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {state_json}")
            return None
