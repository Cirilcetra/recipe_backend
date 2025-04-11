#!/usr/bin/env python3

def find_video_index(videos: list, vid: str):
    for k, video in enumerate(videos):
        if video['video_id'] == vid:
            return k
    return 0


def get_title(vtitle: str, curr_index: int, prefix_name: str, total: int):
    if not prefix_name:
        return vtitle
    txtnum = str(curr_index).rjust(len(str(total)), '0')
    return f'{prefix_name} - {txtnum}'


def byte_to_text(bytesize: int, precision: int = 2):
    abbrevs = (
        (1 << 50, 'PB'),
        (1 << 40, 'TB'),
        (1 << 30, 'G'),
        (1 << 20, 'M'),
        (1 << 10, 'K'),
        (1, 'bytes')
    )
    if bytesize == 1:
        return '1 byte'
    for factor, suffix in abbrevs:
        if bytesize >= factor:
            break
    if factor == 1:
        precision = 0
    result = bytesize / float(factor)
    if result <= 0:
        return 0
    return '%.*f %s' % (precision, result, suffix)
