import json
import os
import sys

import numpy as np

images_per_second = 10

data_folder = '/arc/airplanes'
bridge_labels = os.path.join(data_folder, 'bridge')
plane_labels = os.path.join(data_folder, 'plane')

for f in [bridge_labels, plane_labels]:
    if not os.path.exists(f):
        os.mkdir(f)

file = 'chunk_14.mp4.json'
if len(sys.argv) > 1:
    file = sys.argv[1]


def main(file):
    d = json.load(open(file))

    events = d['videoEvents']

    for name, v in events.items():
        print name
        process_one(name, v)
        # break


def process_one(name, v):
    start_meta = 'aircraft_stationary'
    end_meta = 'pushback_started'

    if not name.startswith('chunk_'):
        print "WTF name is " + name
        return

    starts = filter_events(v, start_meta)
    ends = filter_events(v, end_meta)

    if not starts or not ends:
        print "No starts or ends events"
        return

    index = get_video_index(name)
    count = images_count(index)

    res = get_res(count, ends, starts)

    np.save(os.path.join(plane_labels, str(index)), res)


def get_res(count, ends, starts):
    res = np.zeros(count * images_per_second)
    last = 0
    i, j = 0, 0
    while i < len(starts) or j < len(ends):
        start = starts[i] if i < len(starts) else 10 ** 9
        end = ends[j] if j < len(ends) else 10 ** 9

        is_end = start > end
        next = int(min(start, end) * images_per_second)
        if is_end:
            res[last: next] = 1
            j += 1
            print last, next
        else:
            if j >= len(ends):
                res[next:] = 1
                print next
            i += 1

        last = next

    return res


def get_video_index(name):
    try:
        return int(name[6:8])
    except:
        return int(name[6])


def images_count(video_index):
    return 10000
    # return len(os.listdir(os.path.join(data_folder, str(video_index))))


def filter_events(v, metaId):
    return list(sorted([t["timestamp"] for t in v if t['metaId'] == metaId]))


main(file)
