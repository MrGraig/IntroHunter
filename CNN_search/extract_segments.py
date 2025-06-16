import json
import numpy as np

SEGMENT_LEN = 6  # seconds
STEP = 5


def time_to_seconds(time_str):
    parts = [int(p) for p in time_str.strip().split(':')]
    h, m, s = parts
    return h * 3600 + m * 60 + s


def get_segments(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = []
    for entry in meta.values():
        video_name = entry['url'].split('-')[-1]
        video_path = f"../data/data_train_short/data_train_short/-{video_name}/-{video_name}.mp4"
        intro_start = time_to_seconds(entry['start'])
        intro_end = time_to_seconds(entry['end'])
        duration = intro_end
        intros = [[intro_start, intro_end]]
        # нарезаем на сегменты длиной SEGMENT_LEN с шагом STEP
        times = np.arange(0, duration - SEGMENT_LEN, STEP)
        for t0 in times:
            t1 = t0 + SEGMENT_LEN

            label = 0
            for intro_start, intro_end in intros:
                inter = max(0, min(t1, intro_end) - max(t0, intro_start))
                # если есть хотя бы 50% перекрытия с интро
                if inter >= 0.5 * SEGMENT_LEN:
                    label = 1
                    break
            samples.append({'video': video_path, 't_start': t0, 't_end': t1, 'label': label})
    return samples


