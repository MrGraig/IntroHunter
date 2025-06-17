import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from extract_segments import get_segments
from extract_video_features import extract_video_features


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def infer(model, device, samples, threshold=0.5, fps=1):
    preds = []
    for sample in tqdm(samples, desc="Inference"):
        video_path = sample['video']
        t_start = sample['t_start']
        t_end = sample['t_end']
        features = extract_video_features(model, device, video_path, t_start, t_end, fps=fps)
        features_tensor = torch.tensor(features).float().to(device).unsqueeze(0)

        with torch.no_grad():
            pred_score = model(features_tensor)

        label = int(pred_score > threshold)
        preds.append({
            "video": video_path,
            "t_start": t_start,
            "t_end": t_end,
            "score": pred_score,
            "label": label
        })
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к файлу с обученной моделью')
    parser.add_argument('--meta', type=str, required=True, help='Путь к json c описанием видео и разметкой')
    parser.add_argument('--device', type=str, default='cuda', help='cuda или cpu')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог вероятности для интро')
    parser.add_argument('--out', type=str, default='predictions.json', help='Куда сохранить результат')
    parser.add_argument('--fps', type=int, default=1, help='Частота кадров для извлечения эмбеддингов')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.model, device)
    samples = get_segments(args.meta)

    preds = infer(model, device, samples, threshold=args.threshold, fps=args.fps)

    intro_preds = {}
    for pred in preds:
        if pred["label"]:
            vid = os.path.basename(pred["video"])
            if vid not in intro_preds:
                intro_preds[vid] = []
            intro_preds[vid].append({
                "t_start": pred["t_start"],
                "t_end": pred["t_end"],
                "score": pred["score"]
            })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(intro_preds, f, ensure_ascii=False, indent=2)
    print(f"Готово! Сохранено в {args.out}")


if __name__ == "__main__":
    main()
