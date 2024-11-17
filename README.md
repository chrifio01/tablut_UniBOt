# tablut_UniBOt

This Tablut DQN model was developed by @chrifio01 @alessandrocapialbi @mpreda01 @raphsa for the Tablut challenge at UniBO, hosted by @AndreaGalassi.

## Prerequisites

1. **Docker**: Make sure Docker is installed on your machine. You can download it from [Docker's official site](https://www.docker.com/get-started).
2. **Docker Compose**: Docker Compose usually comes with Docker Desktop. If not, install it separately as needed.

## Setup

Clone the repository and navigate to the project root directory.

```sh
git clone git@github.com:chrifio01/tablut_UniBOt.git
cd tablut_UniBOt
```

# Build and Run in Development

Use Docker Compose to set up and run the bot in development mode. The bot requires environment variables to be passed when starting the Docker container, including:

```plaintext
    PLAYER_COLOR and OPPONENT_COLOR: Set to WHITE or BLACK (case-insensitive).
    TIMEOUT: Set to a positive integer (e.g., 60).
    SERVER_IP: Set to the IP address of the server.
```

# Build the Docker Image

To build the Docker image for development, use the following command:

```sh
docker compose -f docker/docker-compose.dev.yaml build
```

# Run the Model in Development

Once the image is built, you can start the bot using Docker Compose and passing in the required environment variables:

```sh
PLAYER_COLOR=WHITE OPPONENT_COLOR=BLACK TIMEOUT=60 docker compose -f docker/docker-compose.dev.yaml up
```

## Explanation of Environment Variables

    PLAYER_COLOR: Specifies the bot’s role as either the "WHITE" or "BLACK" for our player.
    OPPONENT_COLOR: Specifies the bot’s role as either the "WHITE" or "BLACK" for the opponent.
    TIMEOUT: Specifies the timeout (in seconds) for the bot’s operations.
    SERVER_IP: Sets the server IP address for communication.
