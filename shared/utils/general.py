"""
General utilities for parsing YAML files.

This module provides functionality to read and parse YAML files into Python
dictionary objects for easy manipulation and access to structured data.

Dependencies:
    - yaml (requires PyYAML library)

Functions:
    - parse_yaml: Parses a YAML file and returns its contents as a dictionary.
"""

import yaml

def parse_yaml(filename: str) -> dict:
    """
    Parse a YAML file and return its contents as a dictionary.

    Args:
        filename (str): The path to the YAML file to be parsed.

    Returns:
        dict: A dictionary representation of the YAML file's contents.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(filename, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)
