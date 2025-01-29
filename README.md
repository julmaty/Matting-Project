# SAM 2: Matting

![SAM 2 architecture](assets/model_diagram.png?raw=true)

Данная работа сделана к качестве итогового проекта курса по компьютерному зрению от Deep Learning School. Необходимо было переучить нейросеть Sam 2 от Meta (Признана экстримистской организацией и запрещена в РФ) с задачи сегментации на задачу матирования.

Основная часть работы находится в ноутбуках Train Pipeline.ipynb, Metrics.py. Посмотреть на работу сети можно в ноутбуке ./sam2/notebooks/video_predictor_matting_example.ipynb

## Подготовка

При по [Sam](https://arxiv.org/abs/2304.02643), [Sam2](https://arxiv.org/abs/2408.00714), [Matting Anything](https://arxiv.org/abs/2306.05399), RVB (https://peterl1n.github.io/RobustVideoMatting/#/).
В основе ледит проект [Sam2](https://github.com/facebookresearch/sam2). Он был изучен, базовая модель запущена - в ноутбуках, а также в виде демо.


## Обучение

Ноутбук дообучения Lora-адаптера модели можно найти в файле Train Pipeline.ipynb. Обучение проводилось на датасете [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets), также использовались бекграунды из него. В конце модель дообучалась на изображениях датасета [RefMatte-RW100](https://github.com/jizhiziLi/rim).

Lora-адаптер был внедрен в линейные слои Mask Decoder, а также в линейные слои финального Attantion слоя. В качестве функции потерь использовалась L1. Итоговые маски переводились в диапазон от 0 до 1 из логитов с помощью сигмоиды. Для обучения использовалась встроенная функциональность Sam2Train. Для финального дообучения также использовалась функциональность SAM2VideoPredictor.

Итоговые чекпойнты сохранены в './checkpoints/sam2.1_hiera_large.pt. Для обучения использовалась конфигурация, которую можно найти по пути '/configs/sam2.1_hiera_l_training.yaml'

## Доработка кода

В папке sam2 вы можете найти все стандартные файлы проекта sam2. Однако в целях удобства имплементации модели матирования также были были добавлены некоторые кастомные файлы.
В частности, в папке 'sam2/sam2' можно найти lora.py включающий в себя класс лора-адаптера и его имплементации в слои MLP и Attantion.
Также был создан build_sam_matting.py для удобства создания матирующей модели при инференсе и класс SAM2VideoPredictorTrain для удобства тренировки.

## Метрики

Было проведено оценивание дообученной модели с базовой по метрикам SAD, MAD и MSE.
Ноутбук оценки метрик можно найти в файле Metrics.py.

С точки зрения метрик, мы получили впечатляющее улучшение в задаче матирования по всем метрикам. Измерения проводились на видео 200-299 датасета VideoMatte240K. 
Следует, однако, понимать, что не все ошибки связаны непосредственно с улучшением матирования. Иногда оригинальная сеть просто предпочитает сосредоточится на части изображения вместо всей фигуры.
Например, выбирает галстук вместо мужчины и т.п.

Дообученная матирующая сеть:
SAD: 5.639358961760998
MAD: 17.11857883655466
MSE: 15.173368273564847

Оригинальная Sam2:
SAD_Sam2: 31.777083465397354
MAD_Sam2: 96.14875594817568
MSE_Sam2: 92.7790659313323

## Запуск

В отличие от исходной модели, нашу реализацию можно запустить в Colab без предварительной установки. Достаточно только загрузить на Гугл диск необходимые файлы модели. После этого можно будет пользоваться ноутбуками.

Чекпойнты дообученной модели необходимо скачать с [Google диска по ссылке](https://drive.google.com/file/d/1jilheGaE0vztm3Xp7uk-Eq8_CptnPo02/view?usp=drive_link) и положить в папку checkpoints.
Если планируете использовать какие-либо фоновые изображения, их необходимо закачать в папку Backgrounds. Можно, например, использовать фоновые изображения от VideoMatte240K.

Для локального запуска, необходимо установить зависимости

```bash
pip install -r requirements.txt
```

В качестве альтернативы можно перейти в папку sam2 и запустить базовую установку sam2:

```bash
pip install -e .
pip install -e ".[notebooks]"
```
## Демо

Переработанный для матирования демо размещено в './sam2/demo/backend_matting. В ответ на запросы к graphQl он теперь отдает rle строку, в которой закодированы маски со значениями прозрачности от 0 до 255. Для этого была добавлена новая функция кодирования. Ранее отдавались бинарные маски.

Фронтенд часть требует дальнейших переработок.

Новую бэкенд часть можно запустить через 

```bash
conda create --name sam2-demo python=3.10 --yes
conda activate sam2-demo
conda install -c conda-forge ffmpeg
pip install -e '.[interactive-demo]'

cd demo/backend_matting/server/

PYTORCH_ENABLE_MPS_FALLBACK=1 \
APP_ROOT="$(pwd)/../../../" \
API_URL=http://localhost:7263 \
DATA_PATH="$(pwd)/../../../data" \
DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4 \
gunicorn \
    --worker-class gthread app:app \
    --workers 1 \
    --threads 2 \
    --bind 0.0.0.0:7263 \
    --timeout 60
```

Или при запуске через Windows:

```bash
set FLASK_RUN_PORT=7263
set PYTORCH_ENABLE_MPS_FALLBACK=1
set APP_ROOT=%cd%\../../../../
set API_URL=http://localhost:7263
set DATA_PATH=%cd%\../../data
set DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4

flask run
```

## Возможные доработки

Для корректной работы всего демо необходима доработка фронтенда.

Кроме того, из-за того, что в датасете VideoMatte240K был только один объект на переднем плане, после переобучения сеть стала хуже в трекинге нескольких движущихся объектов на переднем плане.
Это можно исправить путем обучения нейросети на более большой датасете с несколькими размеченными масками (например, RefMatte). Но на это у автора работы нет ресурсов.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```


```bibtex
@article{VideoMatte240K,
  title={Real-Time High-Resolution Background Matting},
  author={Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian Curless, Steve Seitz, Ira Kemelmacher-Shlizerman},
  journal={arXiv preprint arXiv:2012.07810 },
  url={https://arxiv.org/abs/2012.07810},
  year={2020}
}
```

```bibtex
@inproceedings{rim,
  title={Referring Image Matting},
  author={Li, Jizhizi and Zhang, Jing and Tao, Dacheng},
  booktitle={Proceedings of the IEEE Computer Vision and Pattern Recognition},
  year={2023}
}
```
