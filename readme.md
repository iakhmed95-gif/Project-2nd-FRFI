

## 1) Необходимо подготовить файл с данными
1. Создаем папку `data` рядом с файлами скриптов.
2. Кладем туда свой файл с данными, например `dataset.csv`.
3. Необходимо убедиться, что в файле есть **две колонки**:
   - колонка с текстом (сообщения)
   - колонка с меткой (тональность/класс)

## 2) Далее необходимо установить Python 3.11 (обязательно)
Вариант через Homebrew:
```
brew install python@3.11
```

## 3) Устанавливаем библиотеки
Вводим:

```
python -m pip install pandas numpy scikit-learn nltk gensim scipy matplotlib seaborn pymorphy3
```

## 4) Настраиваем три строки в `data_preprocessing.py`
Необходимо открыть файл `data_preprocessing.py` и найди блок:

```
DATA_PATH = Path("data/dataset.csv")
TEXT_COL = "text"
LABEL_COL = "label"
LANGUAGE = "russian"
```

Изменить на свои значения:
- `DATA_PATH` — путь к файлу.
  Пример: если файл называется `mydata.csv`, то будет:
  ```
  DATA_PATH = Path("data/mydata.csv")
  ```
- `TEXT_COL` — точное имя колонки с текстом.
- `LABEL_COL` — точное имя колонки с меткой.
- `LANGUAGE` — оставь `"russian"`, если тексты на русском, иначе `"english"`.

## 5) Запускаем первый скрипт (создание 9 датасетов)
В терминале (в папке проекта):

```
python data_preprocessing.py
```

После запуска появится папка `outputs/` с датасетами, статистикой и графиками:
- `outputs/corpus_stats.json` — статистика по 3 корпусам
- `outputs/datasets_stats.json` — статистика по 9 датасетам

## 6) Запускаем второй скрипт (обучение моделей)
В терминале:

```
python train_models.py
```

После запуска появятся:
- `outputs/reports/` — отчёты по метрикам
- `outputs/confusion_matrices/` — матрицы ошибок (картинки)

## Дополнительно (бонус)
В `data_preprocessing.py` можно включить генерацию синтетического корпуса:
```
SYNTHETIC_DATASET = "raw_w2v"
```
Это создаст отдельный набор с суффиксом `_synthetic`.
