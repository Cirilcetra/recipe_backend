#!/usr/bin/env python3

from urllib.parse import urlparse, parse_qs

from bella import write_json_to_file


def get_playlist_title(api, pid: str):
    print(f'get playlist info {pid}')
    pl = api.get_playlist_by_id(playlist_id=pid)
    if len(pl.items) == 0:
        return False
    return pl.items[0].snippet.title


def get_video_from_playlist(api, pid: str):
    print(f'get items from playlist {pid}')
    result = api.get_playlist_items(
        playlist_id=pid,
        count=None
    )
    videos = []

    for item in result.items:
        video_id = item.snippet.resourceId.videoId
        print(f'loading video {video_id}')
        video = api.get_video_by_id(video_id=video_id, return_json=True)
        snippet = video['items'][0]['snippet']
        videos.append(dict(
            video_id=video_id,
            title=snippet['title'],
            published_at=snippet['publishedAt']
        ))
        print(f'loaded {len(videos)} items')

    return videos


def get_playlist_id(param):
    playlist_id = None if param.startswith('https://') else param
    if not playlist_id:
        query = urlparse(param).query
        queryDict = parse_qs(query)
        if 'list' in queryDict and len(queryDict['list']) > 0:
            playlist_id = queryDict['list'][0]
    return playlist_id


def get_playlist_index(api, store_dir: str, pid: str, filename: str):
    pl_title = get_playlist_title(api, pid)
    if not pl_title:
        return f'Playlist {pid} is empty or unavailable to load'
    videos = get_video_from_playlist(api, pid)
    if videos and len(videos) > 0:
        fname = filename if filename else pl_title
        fpath = f'{store_dir}/{fname}.json'
        write_json_to_file(fpath, videos)
        return f'Indexed {len(videos)} videos from playlist {pid}'
    else:
        return f'There is no video from playlist {pid}'
