import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_video_features(model, device, video_path, t_start, t_end, fps=1):
    cap = cv2.VideoCapture(video_path)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_indices = [int((t_start + i) * current_fps) for i in np.arange(0, t_end - t_start,
                                                                         1 / fps)]  # Список индексов кадров, которые соответствуют времени внутри сегмента
    features = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()
        if not ret:
            print('Не удалось считать кадр')
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(x).flatten().cpu().numpy()

        features.append(feat)

    cap.release()
    if features:
        return np.mean(features, axis=0)  # Итоговый эмбеддинг[512]
    else:
        return np.zeros(512)
