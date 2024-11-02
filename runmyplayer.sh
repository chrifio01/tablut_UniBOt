#!/bin/sh

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <white|black> <timeout> <ip>"
  exit 1
fi

# Convert the player color to uppercase for consistent handling
PLAYER_COLOR=$(echo "$1" | tr '[:lower:]' '[:upper:]')
if [ "$PLAYER_COLOR" != "WHITE" ] && [ "$PLAYER_COLOR" != "BLACK" ]; then
  echo "Error: First parameter must be 'WHITE' or 'BLACK' (case insensitive)"
  exit 1
fi

# Validate timeout is an integer greater than 0
TIMEOUT=$2
if ! echo "$TIMEOUT" | grep -qE '^[0-9]+$' || [ "$TIMEOUT" -le 0 ]; then
  echo "Error: Second parameter must be an integer greater than 0"
  exit 1
fi

# Explicitly convert TIMEOUT to an integer
TIMEOUT=$((TIMEOUT))

# Validate IP address format (basic regex using grep)
IP_REGEX="^([0-9]{1,3}\.){,3}([0-9]{1,3})+$"
if ! echo "$3" | grep -qE "$IP_REGEX"; then
  echo "Error: Third parameter must be a valid IP address"
  exit 1
fi
SERVER_IP=$3

# Export environment variables to make them available to main.py
export PLAYER_COLOR=$PLAYER_COLOR
export TIMEOUT=$TIMEOUT
export SERVER_IP=$SERVER_IP

# Set the websocket port based on the player color
if [ "$PLAYER_COLOR" = "WHITE" ]; then
  export WEBSOCKET_PORT=5800
else
  export WEBSOCKET_PORT=5801
fi

# Run the main.py script with the configured environment variables
python main.py