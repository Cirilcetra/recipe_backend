#!/usr/bin/env python3

import os
from urllib.parse import urlparse, parse_qs

from pytube import YouTube
from bella import read_json_from_file

from .helper import byte_to_text, find_video_index, get_title


def get_video_id(param):
    video_id = None if param.startswith('https://') else param
    if not video_id:
        query = urlparse(param).query
        queryDict = parse_qs(query)
        if 'v' in queryDict and len(queryDict['v']) > 0:
            video_id = queryDict['v'][0]
    return video_id


def get_video_by_id(api, store_dir: str, vid: str, fname: str = None):
    video_dir = f'{store_dir}/video'
    if not os.path.exists(video_dir):
        os.system(f'mkdir -p {video_dir}')

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

    stream = yt.streams.get_highest_resolution()
    print('Selected stream:')
    print(stream)
    print(f'File size: {byte_to_text(stream.filesize)}')

    def on_complete(fpath):
        print(f'Finished downloading: {fpath}')

    print(f'Downloading {ytUrl}')
    stream.on_complete = on_complete
    stream.download(
        video_dir,
        file_name,
    )


def get_videos_from_indexed_list(
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
        get_video_by_id(api, store_dir, item['video_id'], item['title'])
        k += 1
