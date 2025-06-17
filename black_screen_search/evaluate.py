import os
import json
import glob
from datetime import datetime, timedelta
from search_between_black_stream import find_short_intro_between_black


def parse_time_str(time_str):
    t = datetime.strptime(time_str, '%H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second


"""
Было замечено, что в сете есть примеры, где start > end. 
Эмпирически было выявлено, что start в таком случае нужно уменьшить на 1 мин.
"""
def correct_times(start, end):
    if start > end:
        start -= 60
    return start, end


def interval_iou(interval_a, interval_b):
    a_start, a_end = interval_a
    b_start, b_end = interval_b
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union == 0:
        return 0.0
    return intersection / union


def main(data_dir, json_path, iou_thr=0.5):
    with open(json_path, encoding='utf-8') as f:
        gt = json.load(f)

    TP, FP, FN = 0, 0, 0
    total = 0

    for video_key, video_info in gt.items():
        video_folder = os.path.join(data_dir, video_key)
        video_files = glob.glob(os.path.join(video_folder, '*.mp4'))
        if not video_files:
            print(f"Видео для {video_key} не найдено")
            continue
        video_path = video_files[0]

        start_time = parse_time_str(video_info["start"])
        end_time = parse_time_str(video_info["end"])
        start_time, end_time = correct_times(start_time, end_time)
        gt_interval = (start_time, end_time)

        pred_intervals = find_short_intro_between_black(video_path)
        pred_used = [False] * len(pred_intervals)
        matched = False

        # Перебор всех предсказаний — ищем совпадения
        for idx, pred in enumerate(pred_intervals):
            iou = interval_iou(gt_interval, pred)
            if iou >= iou_thr:
                TP += 1
                pred_used[idx] = True
                matched = True
                break

        FP += pred_used.count(False)

        if not matched:
            FN += 1

        total += 1
        print(f"{video_info['name']} | GT: {gt_interval} | Pred: {pred_intervals} | TP: {TP} | FP: {FP} | FN: {FN}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"\nВсего: {total}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")


if __name__ == "__main__":
    main(data_dir='../data/data_test_short/data_test_short',
         json_path="../data/data_test_short/data_test_short/labels.json")
