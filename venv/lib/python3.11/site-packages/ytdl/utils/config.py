#!/usr/bin/env python3

import os
import getpass
from typing import Any

from bella import write_json_to_file, read_json_from_file

ENV = os.getenv('ENV', 'prod')

USER_DIR = f'/home/{getpass.getuser()}' if ENV == 'prod' else './ptester'
YTDL_CONF_DIR = f'{USER_DIR}/.config/ytdl'
CONFIG_FILE = f'{YTDL_CONF_DIR}/config.json'
YTDL_DEFAULT_STORE_DIR = f'{USER_DIR}/ytdl_files'


def load_config():
    if not os.path.exists(YTDL_CONF_DIR):
        os.system(f'mkdir -p {YTDL_CONF_DIR}')
    if not os.path.exists(CONFIG_FILE):
        if not os.path.exists(YTDL_DEFAULT_STORE_DIR):
            os.system(f'mkdir -p {YTDL_DEFAULT_STORE_DIR}')
        write_json_to_file(CONFIG_FILE, {
            'api_key': '',
            'store_dir': YTDL_DEFAULT_STORE_DIR
        })

    return read_json_from_file(CONFIG_FILE)


def get_config(key: str = None):
    c = load_config()
    if not key:
        print(c)
    elif key not in c:
        print(f'`{key}` is a invalid key')
    else:
        print(f'{key} --> {c[key] if key in c else "None"}')


def set_config(key: str, val: Any):
    c = load_config()
    c[key] = val
    write_json_to_file(CONFIG_FILE, c)
    print(f'`config.{key}` has been set to `{val}`')
