import cv2
import numpy as np
import pytesseract


# Функция поиска промежутков между черными экранами
def find_black_screens(video_path, black_thresh=5, min_count_black_frames=5, max_duration_minutes=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frame = int(max_duration_minutes * 60 * fps)  # Рассматриваем только первые 5 минут
    idx = 0
    black_frames = []
    intro_timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= max_frame:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < black_thresh:
            black_frames.append(idx)
        else:
            if len(black_frames) >= min_count_black_frames:
                end_idx = black_frames[-1]
                end_time = end_idx / fps
                intro_timestamps.append(end_time)
            black_frames = []
        idx += 1

    if len(black_frames) >= min_count_black_frames:
        end_idx = black_frames[-1]
        end_time = end_idx / fps
        intro_timestamps.append(end_time)

    cap.release()
    return intro_timestamps


# Функция поиска текста в промежутке
def has_text_in_interval(video_path, start, end, fps, step=5, min_ocr_chars=3):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n_frames = int((end - start) * fps)
    ocr_found = False
    for i in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps) + i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if len(text.strip()) >= min_ocr_chars:
            ocr_found = True
            break
    cap.release()
    return ocr_found


# Функция сопоставления временным рамкам
def find_short_intro_between_black(video_path, min_intro=5, max_intro=30, min_ocr_chars=3):
    black_timestamps = find_black_screens(video_path)
    result_intros = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    for i in range(len(black_timestamps) - 1):
        start = black_timestamps[i]
        end = black_timestamps[i + 1]
        duration = end - start
        if min_intro <= duration <= max_intro:
            ocr = has_text_in_interval(video_path, start, end, fps, min_ocr_chars=min_ocr_chars)
            print(f"Interval: ({start:.2f}, {end:.2f}) sec, duration: {duration:.2f} sec, OCR: {ocr}")
            if ocr:
                result_intros.append((start, end))
    return result_intros

