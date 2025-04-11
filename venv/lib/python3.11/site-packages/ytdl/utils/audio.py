#!/usr/bin/env python3

import os

from pytube import YouTube
from bella import read_json_from_file

from .helper import byte_to_text, find_video_index, get_title


def get_audio_by_id(api, store_dir: str, vid: str, fname: str = None):
    audio_dir = f'{store_dir}/audio'
    dl_dir = f'{store_dir}/downloads'
    if not os.path.exists(audio_dir):
        os.system(f'mkdir -p {audio_dir}')
    if not os.path.exists(dl_dir):
        os.system(f'mkdir -p {dl_dir}')

    print('Get video info', vid)
    ytUrl = f'https://youtu.be/{vid}'
    yt = YouTube(ytUrl)

    title = yt.title
    print('Loaded video file', title)
    file_name = fname if fname else yt.title

    print('Available stream:')
    for s in yt.streams:
        print(s)

    print('/' * 70)

    stream = yt.streams.filter(only_audio=True).order_by('abr').last()
    print('Selected stream:')
    print(stream)
    print(f'File size: {byte_to_text(stream.filesize)}')

    def on_complete(fpath):
        print('Convert to mp3', fpath)
        mp3_file = f'{audio_dir}/{file_name}.mp3'
        cmd = f'ffmpeg -i "{fpath}" -y "{mp3_file}"'
        print(f'Run command: {cmd}')
        os.system(cmd)
        print(f'Finish downloading {file_name}, mp3 saved at {mp3_file}')

    print(f'Downloading {ytUrl}')
    stream.on_complete = on_complete
    stream.download(
        dl_dir,
        file_name,
    )


def get_audios_from_indexed_list(
        api, store_dir, indexed_file, since, limit, prefix_name, prefix_num):
    print(indexed_file, since, limit, prefix_name)
    videos = read_json_from_file(indexed_file)
    total = len(videos)
    if total == 0:
        return 'List is empty!'
    begin_at = 0 if not since else find_video_index(videos, since)
    end_at = total if not limit else min(total, limit + begin_at)
    extracted = videos[begin_at:end_at]
    k = prefix_num if prefix_num else begin_at + 1
    for item in extracted:
        print('*' * 80)
        title = item['title']
        item['title'] = get_title(title, k, prefix_name, total)
        print(f'Start downloading {item["title"]}')
        get_audio_by_id(api, store_dir, item['video_id'], item['title'])
        k += 1
