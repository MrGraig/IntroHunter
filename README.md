# IntroHunter

## Методы для автоматического обнаружения заставок сериалов/фильмов:

1. **Классический метод** на основе детектирования черных экранов и OCR
2. **Метод на основе CNN-эмбеддингов** и бинарного классификатора

---

## Структура проекта

```
IntroHunter/
├── extract_segments.py            # Генерация обучающих сегментов из JSON-разметки
├── search_between_black_stream.py # Классический метод: детектирование черных экранов + OCR
├── evaluate.py                    # Скрипт оценки качества классического метода
├── model.ipynb                    # Jupyter Notebook: обучение и оценка CNN-эмбеддингового классификатора
├── requirements.txt               # Зависимости проекта
└── .gitignore                     # Исключения для Git
```

---

## Установка

1. Клонировать репозиторий:

   ```bash
   git clone https://github.com/MrGraig/IntroHunter.git
   cd IntroHunter
   ```
2. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```

---

## Генерация обучающих сегментов

Скрипт `extract_segments.py` преобразует JSON-разметку интро в набор сегментов:

```python
from extract_segments import get_segments

segments = get_segments('path/to/labels.json')
# segments — список словарей:
# {'video': 'путь_к_видео', 't_start': t0, 't_end': t1, 'label': 0 или 1}
```

Длительность сегмента и шаг задаются внутри скрипта (6 cегунд и 5 секунд по умолчанию).

---

## Классический метод: поиск интро

Функция `find_short_intro_between_black(video_path)` из `search_between_black_stream.py` выполняет:

1. Детектирование длинных фрагментов черного экрана (порог яркости < 5)
2. Поиск промежутков между черными фрагментами длительностью от 5 до 30 секунд
3. Проверка наличия текста в этих промежутках с помощью OCR (pytesseract)
4. Возврат списка кортежей `(start_sec, end_sec)` с найденными интро

Пример использования:

```python
from search_between_black_stream import find_short_intro_between_black

intros = find_short_intro_between_black('video.mp4')
print(intros)
```

---

## Оценка классического метода

Скрипт `evaluate.py` сравнивает предсказанные интервалы с разметкой JSON и вычисляет TP, FP, FN, Precision и Recall:

```bash
python evaluate.py
```

По умолчанию используются директории:

* `data_dir='../data/data_test_short/data_test_short'`
* `json_path='../data/data_test_short/data_test_short/labels.json'`

---

## Обучение CNN-эмбеддингового классификатора

1. Открыть ноутбук:

   ```bash
   jupyter notebook model.ipynb
   ```
2. Указать пути к видео и JSON-разметке сегментов.
3. В ноутбуке выполняются этапы:

   * Генерация сегментов функцией `get_segments`
   * Извлечение эмбеддингов кадров с помощью ResNet18
   * Усреднение эмбеддингов по каждому сегменту
   * Обучение бинарного классификатора на PyTorch
   * Оценка качества на отложенной выборке

---

## Возможные улучшения

* **Классический метод**:

  * Настройка порогов яркости и длительности черных фреймов
  * Замена или доработка OCR-модулей для повышенной точности

* **CNN-метод**:

  * Добавить учет временной информации (LSTM, Transformers)
  * Анализ аудио-дорожки для комплексного подхода
  * Fine-tuning предобученной CNN на специфичных видеоматериалах

---

## Автор

**MrGraig**
GitHub: [https://github.com/MrGraig/Int](https://github.com/MrGraig/Int)
