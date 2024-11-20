"""
    Our model module.
"""

import os

from shared.utils import parse_yaml

_config_file_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.yaml"
)

CONFIG = parse_yaml(_config_file_path)
