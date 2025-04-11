#!/usr/bin/env python3

import os

import fire
from pyyoutube import Api

from .utils.config import load_config, get_config, set_config
from .utils.playlist import get_playlist_id, get_playlist_index
from .utils.video import get_videos_from_indexed_list, \
    get_video_id, get_video_by_id
from .utils.audio import get_audios_from_indexed_list, get_audio_by_id


conf = load_config()

store_dir = os.path.normpath(
    os.path.join(os.getcwd(), conf['store_dir'])
) if not os.path.isabs(conf['store_dir']) else conf['store_dir']


def show_info():
    print('*' * 72)
    print(f'YouTube API key: `{conf["api_key"]}`')
    print(f'Actual store_dir: `{store_dir}`')
    print('*' * 72)


def configure(key=None, val='---'):
    if val == '---':
        return get_config(key) if key else show_info()
    else:
        return set_config(key, val)


def get_playlist(url=None, filename=None):
    pid = get_playlist_id(url)
    if not pid:
        return 'Please provide playlist ID or full URL'
    api = Api(api_key=conf['api_key'])
    return get_playlist_index(api, store_dir, pid, filename)


def get_video(
    url=None,
    file_name=None,
    index_file=None,
    since=None,
    limit=None,
    prefix_name=None,
    prefix_num=None
):
    api = Api(api_key=conf['api_key'])
    if index_file and os.path.exists(index_file):
        return get_videos_from_indexed_list(
            api, store_dir,
            index_file, since, limit, prefix_name, prefix_num
        )
    vid = get_video_id(url)
    if not vid:
        return 'Please provide video ID or full URL'
    return get_video_by_id(api, store_dir, vid, file_name)


def get_audio(
    url=None,
    file_name=None,
    index_file=None,
    since=None,
    limit=None,
    prefix_name=None,
    prefix_num=None
):
    api = Api(api_key=conf['api_key'])
    if index_file and os.path.exists(index_file):
        return get_audios_from_indexed_list(
            api, store_dir,
            index_file, since, limit, prefix_name, prefix_num
        )
    vid = get_video_id(url)
    if not vid:
        return 'Please provide video ID or full URL'
    return get_audio_by_id(api, store_dir, vid, file_name)


def init():
    if 'api_key' not in conf:
        print('Please set `api_key` first. Command help:')
        print('*' * 72)
        print('  ytdl config api_key YOUR_API_KEY')
        print('*' * 72)
        return False

    return fire.Fire({
        'config': configure,
        'c': configure,
        'playlist': get_playlist,
        'p': get_playlist,
        'video': get_video,
        'v': get_video,
        'audio': get_audio,
        'a': get_audio,
        'info': show_info
    })


if __name__ == '__main__':
    init()
