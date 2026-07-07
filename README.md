# cv_squats

![Preview](demo.gif)

Видео-демонстрационное приложение для оценки позы человека и анализа техники приседаний.

Проект выполнен в рамках практики МТУСИ.

Проект использует дообученную модель RTMPose-S с backend MMPose. Приложение определяет 2D-ключевые точки тела человека на видео с приседаниями, отрисовывает скелет, рассчитывает показатели техники, сохраняет артефакты запусков и при необходимости экспортирует Excel-отчёт.

## Возможности

* Загрузка видео с приседаниями через интерфейс Streamlit.
* Запуск RTMPose-S pose estimation на выбранных кадрах видео.
* Отрисовка скелета тела на выбранных кадрах и аннотированном видео.
* Расчёт показателей, связанных с техникой приседания:

  * угол левого и правого колена;
  * выбранный коленный угол;
  * угол наклона корпуса;
  * фаза приседания: `top`, `middle`, `bottom`, `unknown`;
  * статус глубины: `sufficient`, `insufficient`, `unknown`;
  * confidence и флаги надёжности;
  * сводка предупреждений.
* Сохранение артефактов в JSON, CSV, аннотированные кадры, аннотированное видео, JSONL-историю и Excel-отчёт.

## Структура проекта

```text
cv_squats/
├── configs
│   ├── app.yaml
│   └── mmpose
│       └── rtmpose_s_5k_30e.py
├── docs/
│   └── report/
├── outputs/
│   ├── metrics/
│   ├── models/
│   │   └── rtmpose/
│   │       └── best_coco_AP_epoch_30.pth
│   └── figures/
├── src/
│   ├── analysis/
│   ├── app/
│   ├── inference/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── run_app.py
├── requirements.txt
└── README.md
```

Папки, создаваемые во время работы приложения, например `outputs/predictions/`, `outputs/uploads/`, `outputs/reports/` и `outputs/history/`, намеренно игнорируются Git.

## Модель

Итоговая выбранная модель:

* RTMPose-S
* Backend: MMPose
* Checkpoint: `outputs/models/rtmpose/best_coco_AP_epoch_30.pth`

Checkpoint включён в репозиторий для воспроизводимости демонстрационного приложения.

## Окружение

Рекомендуемая версия Python: 3.10.

Demo-приложение использует RTMPose-S через MMPose, поэтому запускать его нужно только из окружения `.venv-mmpose`.

Проверенный стек:

* Python 3.10
* PyTorch 2.1.0 + CUDA 12.1
* TorchVision 0.16.0 + CUDA 12.1
* MMEngine 0.10.7
* MMCV 2.1.0
* MMDetection 3.2.0
* MMPose 1.3.2
* MMPreTrain 1.2.0

### Установка окружения

Из корня репозитория:

```bash
python3.10 -m venv .venv-mmpose

.venv-mmpose/bin/python -m pip install --upgrade pip setuptools wheel

# Базовые зависимости demo-приложения, тестов и Excel-отчёта
.venv-mmpose/bin/python -m pip install -r requirements.txt

# PyTorch CUDA 12.1 и базовые зависимости OpenMMLab
.venv-mmpose/bin/python -m pip install -r requirements-mmpose-cu121.txt

# OpenMMLab stack
.venv-mmpose/bin/mim install "mmcv==2.1.0"
.venv-mmpose/bin/mim install "mmdet==3.2.0"
.venv-mmpose/bin/mim install "mmpose==1.3.2"
.venv-mmpose/bin/mim install "mmpretrain==1.2.0"

## Запуск Streamlit Demo

```bash
.venv-mmpose/bin/python -m streamlit run run_app.py
```

После запуска откройте локальный URL Streamlit в браузере и загрузите MP4-видео с приседаниями.

## Запуск анализа видео через CLI

```bash
.venv-mmpose/bin/python -m src.inference.video_pipeline \
  --input path/to/squat_video.mp4 \
  --config configs/app.yaml \
  --frame-stride 5 \
  --save-annotated-video \
  --save-history \
  --export-excel
```

Быстрый smoke-запуск:

```bash
.venv-mmpose/bin/python -m src.inference.video_pipeline \
  --input path/to/squat_video.mp4 \
  --config configs/app.yaml \
  --frame-stride 5 \
  --max-frames 200 \
  --save-annotated-video \
  --save-history \
  --export-excel
```

## Выходные артефакты

Каждый запуск анализа видео создаёт директорию:

```text
outputs/predictions/<run_id>/
```

Основные файлы:

```text
video_summary.json
video_frames_metrics.csv
video_keypoints.json
sampled_frames/*.jpg
annotated_video.mp4
annotated_video_h264.mp4
```

Дополнительные результаты:

```text
outputs/history/runs.jsonl
outputs/reports/video_report_<run_id>.xlsx
```

## Основные показатели

* `processed_frame_count`: количество обработанных кадров видео.
* `valid_keypoint_frame_ratio`: доля кадров с валидно найденными ключевыми точками.
* `usable_knee_frame_ratio`: доля кадров, на которых можно рассчитать угол колена.
* `reliable_frame_ratio`: доля кадров, признанных надёжными для анализа приседания.
* `min_selected_knee_angle`: минимальный выбранный угол колена за весь запуск.
* `max_selected_knee_angle`: максимальный выбранный угол колена за весь запуск.
* `mean_torso_lean_angle`: средний угол наклона корпуса.
* `depth_status`: результат оценки глубины приседания: `sufficient`, `insufficient` или `unknown`.
* `phase_counts`: количество кадров по фазам приседания.
* `dominant_warnings`: агрегированные типы предупреждений.

## Известные ограничения

* Анализ техники является MVP-решением на основе правил поверх 2D-ключевых точек.
* Приложение не оценивает 3D-позу тела и нагрузки на суставы.
* Demo использует инференс по ограничивающей рамке всего кадра; качество может снижаться, если человек виден частично, находится слишком близко или слишком далеко, перекрыт другими объектами либо расположен у границы кадра.
* Качество результата зависит от угла съёмки, освещения, одежды, перекрытий и видимости всего тела.

## Тесты

Запуск тестов, связанных с demo:

```bash
.venv-mmpose/bin/python -m pytest
```

Запуск только demo/runtime-подмножества:

```bash
.venv-mmpose/bin/python -m pytest \
  tests/test_angles.py \
  tests/test_history.py \
  tests/test_report_excel.py \
  tests/test_rtmpose_video_runner.py \
  tests/test_streamlit_app.py \
  tests/test_video_export.py \
  tests/test_video_pipeline.py
```

## Материалы для отчёта

Материалы для отчёта хранятся в директории:

```text
docs/report/
```

Экспериментальные и сравнительные метрики хранятся в директории:

```text
outputs/metrics/
```

---

Video-only demo application for human pose estimation and squat technique analysis.

The project uses a fine-tuned RTMPose-S model with the MMPose backend. The application estimates 2D human keypoints on squat videos, draws a skeleton, calculates squat indicators, stores run artifacts, and optionally exports an Excel report.

## Features

* Upload a squat video through a Streamlit interface.
* Run RTMPose-S pose estimation on sampled video frames.
* Draw a body skeleton on sampled frames and annotated video.
* Calculate squat-specific indicators:

  * left and right knee angles;
  * selected knee angle;
  * torso lean angle;
  * squat phase: `top`, `middle`, `bottom`, `unknown`;
  * depth status: `sufficient`, `insufficient`, `unknown`;
  * confidence and reliability flags;
  * warning summary.
* Save artifacts as JSON, CSV, annotated frames, annotated video, JSONL history, and Excel report.

## Project Structure

```text
cv_squats/
├── configs/
│   └── app.yaml
├── docs/
│   └── report/
├── outputs/
│   ├── metrics/
│   ├── models/
│   │   └── rtmpose/
│   │       └── best_coco_AP_epoch_30.pth
│   └── figures/
├── src/
│   ├── analysis/
│   ├── app/
│   ├── inference/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── run_app.py
├── requirements.txt
└── README.md
```

Runtime-generated folders such as `outputs/predictions/`, `outputs/uploads/`, `outputs/reports/`, and `outputs/history/` are intentionally ignored by Git.

## Model

Final selected model:

* RTMPose-S
* Backend: MMPose
* Checkpoint: `outputs/models/rtmpose/best_coco_AP_epoch_30.pth`

The checkpoint is included in this repository for demo reproducibility.

## Environment

Recommended Python version: 3.10.

The project was tested in a separate MMPose environment named `.venv-mmpose`.

Create and activate the environment:

```bash
python3.10 -m venv .venv-mmpose
source .venv-mmpose/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MMPose/MMCV installation can be sensitive to CUDA, PyTorch, and Python versions. The tested stack was based on:

* Python 3.10
* PyTorch 2.1.0 + CUDA 12.1
* MMEngine 0.10.7
* MMCV 2.1.0
* MMPose 1.3.2

## Run Streamlit Demo

```bash
.venv-mmpose/bin/python -m streamlit run run_app.py
```

Then open the local Streamlit URL in a browser and upload an MP4 squat video.

## Run CLI Video Analysis

```bash
.venv-mmpose/bin/python -m src.inference.video_pipeline \
  --input path/to/squat_video.mp4 \
  --config configs/app.yaml \
  --frame-stride 5 \
  --save-annotated-video \
  --save-history \
  --export-excel
```

Optional fast smoke run:

```bash
.venv-mmpose/bin/python -m src.inference.video_pipeline \
  --input path/to/squat_video.mp4 \
  --config configs/app.yaml \
  --frame-stride 5 \
  --max-frames 200 \
  --save-annotated-video \
  --save-history \
  --export-excel
```

## Output Artifacts

Each video run creates a directory:

```text
outputs/predictions/<run_id>/
```

Main files:

```text
video_summary.json
video_frames_metrics.csv
video_keypoints.json
sampled_frames/*.jpg
annotated_video.mp4
annotated_video_h264.mp4
```

Additional outputs:

```text
outputs/history/runs.jsonl
outputs/reports/video_report_<run_id>.xlsx
```

## Main Indicators

* `processed_frame_count`: number of processed video frames.
* `valid_keypoint_frame_ratio`: share of frames with valid detected keypoints.
* `usable_knee_frame_ratio`: share of frames where knee angle can be calculated.
* `reliable_frame_ratio`: share of frames considered reliable for squat analysis.
* `min_selected_knee_angle`: minimum selected knee angle across the run.
* `max_selected_knee_angle`: maximum selected knee angle across the run.
* `mean_torso_lean_angle`: average torso lean angle.
* `depth_status`: squat depth result: `sufficient`, `insufficient`, or `unknown`.
* `phase_counts`: frame counts by squat phase.
* `dominant_warnings`: aggregated warning types.

## Known Limitations

* The technique analysis is an MVP rule-based system over 2D keypoints.
* It is not medical or professional coaching advice.
* The application does not estimate 3D body pose or joint loads.
* The demo uses whole-frame bbox inference; quality may degrade when the person is partially visible, too close, too small, occluded, or near frame boundaries.
* No video benchmark was performed, so this project does not claim real-time video performance.
* Result quality depends on camera angle, lighting, clothing, occlusions, and full-body visibility.

## Tests

Run demo-related tests:

```bash
.venv-mmpose/bin/python -m pytest
```

Or run only the demo/runtime subset:

```bash
.venv-mmpose/bin/python -m pytest \
  tests/test_angles.py \
  tests/test_history.py \
  tests/test_report_excel.py \
  tests/test_rtmpose_video_runner.py \
  tests/test_streamlit_app.py \
  tests/test_video_export.py \
  tests/test_video_pipeline.py
```

## Report Materials

Report-facing notes are stored in:

```text
docs/report/
```

Experiment and comparison metrics are stored in:

```text
outputs/metrics/
```
