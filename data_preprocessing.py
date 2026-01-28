#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
import nltk
from nltk import download as nltk_download
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_PATH = Path("data/dataset.csv")

TEXT_COL = "text"
LABEL_COL = "label"

LANGUAGE = "russian"

MIN_ROWS_REQUIRED = 6_000

OUTPUT_DIR = Path("outputs")
DATASETS_DIR = OUTPUT_DIR / "datasets"
PLOTS_DIR = OUTPUT_DIR / "plots"
NLTK_DATA_DIR = OUTPUT_DIR / "nltk_data"
CORPUS_STATS_PATH = OUTPUT_DIR / "corpus_stats.json"
DATASETS_STATS_PATH = OUTPUT_DIR / "datasets_stats.json"

W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
W2V_SEED = 42

SYNTHETIC_DATASET = None  # например: "raw_w2v" или "lemma_w2v"
SYNTHETIC_PER_CLASS = 200
SYNTHETIC_POOL = 50
SYNTHETIC_NOISE = 0.01

SUPPORTED_LANGUAGES = {"english", "russian"}
if LANGUAGE not in SUPPORTED_LANGUAGES:
    raise ValueError(f"LANGUAGE must be one of {SUPPORTED_LANGUAGES}")

if LANGUAGE == "russian":
    try:
        import pymorphy3 as pymorphy
    except ImportError as exc:
        raise ImportError(
            "Для русской лемматизации требуется pymorphy3. "
            "Установите: pip install pymorphy3"
        ) from exc
    _MORPH = pymorphy.MorphAnalyzer()
else:
    _LEMMATIZER = WordNetLemmatizer()

_STEMMER = SnowballStemmer(LANGUAGE)

def ensure_dirs() -> None:
    """Создаёт каталоги, если их нет."""
    for p in (OUTPUT_DIR, DATASETS_DIR, PLOTS_DIR, NLTK_DATA_DIR):
        p.mkdir(parents=True, exist_ok=True)


def read_dataset(path: Union[Path, str]) -> pd.DataFrame:
    """Читает CSV/TSV/Excel. Принимает как Path, так и строку."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Поместите файл в указанную директорию или поправьте DATA_PATH."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str)
    return pd.read_csv(path, dtype=str)  # CSV по умолчанию


def remove_emojis(text: str) -> str:
    """Удаление эмодзи при помощи Unicode‑диапазонов."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Сжать несколько пробелов в один и удалить пробелы по краям."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Пошаговая очистка текста."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # ссылки
    text = re.sub(r"@\w+", " ", text)                     # упоминания
    text = re.sub(r"#\w+", " ", text)                     # хештеги
    text = re.sub(r"\d+", " ", text)                      # цифры
    text = remove_emojis(text)                            # эмодзи
    text = text.translate(str.maketrans("", "", string.punctuation))  # пунктуация
    return normalize_whitespace(text)


def tokenize(text: str) -> List[str]:
    """Токенизация NLTK (`word_tokenize`)."""
    try:
        return word_tokenize(text)
    except LookupError as exc:
        raise LookupError(
            "NLTK punkt resource is missing. "
            "Run: python -m nltk.downloader punkt"
        ) from exc


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    """Стемминг токенов."""
    return [_STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Лемматизация токенов в зависимости от языка."""
    if LANGUAGE == "russian":
        return [_MORPH.parse(t)[0].normal_form for t in tokens]
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def plot_length_distribution(lengths: pd.Series, title: str, filename: str) -> None:
    """Гистограмма + box‑plot распределения длины сообщений."""
    plt.figure(figsize=(10, 4))
    sns.histplot(lengths, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Длина сообщения (токенов)")
    plt.ylabel("Кол‑во")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()

    plt.figure(figsize=(6, 2))
    sns.boxplot(x=lengths)
    plt.title(f"{title} (boxplot)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename.replace(".png", "_box.png"))
    plt.close()


def filter_by_length(
    df: pd.DataFrame,
    text_col: str,
    min_q: float = 0.01,
    max_q: float = 0.99,
) -> pd.DataFrame:
    """
    Удаляем сообщения, которые находятся за пределами квантилей.
    Если после фильтрации осталось < MIN_ROWS_REQUIRED строк,
    возвращаем оригинальный очищенный DataFrame.
    """
    df["_tok_len"] = df[text_col].apply(lambda x: len(tokenize(x)))
    plot_length_distribution(df["_tok_len"], "Length distribution (before)", "length_before.png")

    min_len = df["_tok_len"].quantile(min_q)
    max_len = df["_tok_len"].quantile(max_q)

    filtered = df[(df["_tok_len"] >= min_len) & (df["_tok_len"] <= max_len)].copy()
    plot_length_distribution(filtered["_tok_len"], "Length distribution (after)", "length_after.png")
    filtered.drop(columns=["_tok_len"], inplace=True)

    if len(filtered) < MIN_ROWS_REQUIRED:
        log.warning(
            f"После отбора по длине осталось только {len(filtered)} строк (< {MIN_ROWS_REQUIRED}). "
            "Возвращаю оригинальный очищенный датасет."
        )
        df.drop(columns=["_tok_len"], inplace=True)
        return df
    return filtered


def build_corpora(texts: pd.Series) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    """
    Возвращает четыре объекта:
      * raw — чистый (но не токенизированный) текст,
      * stem — стеммированный,
      * lemma — лемматизированный,
      * tokenized — список токенов (нужен для Word2Vec).
    """
    tokenized = [tokenize(t) for t in texts]
    stemmed = [" ".join(stem_tokens(toks)) for toks in tokenized]
    lemmatized = [" ".join(lemmatize_tokens(toks)) for toks in tokenized]
    raw = texts.tolist()
    return raw, stemmed, lemmatized, tokenized


def build_token_counter(tokens_list: List[List[str]]) -> Counter:
    """Подсчёт частот токенов по всему корпусу."""
    counter: Counter = Counter()
    for doc in tokens_list:
        counter.update(doc)
    return counter


def filter_counter_by_vocab(counter: Counter, vocab: Iterable[str]) -> Counter:
    """Оставляет только токены, вошедшие в словарь векторизатора/модели."""
    vocab_set = set(vocab)
    return Counter({tok: cnt for tok, cnt in counter.items() if tok in vocab_set})


def token_stats_from_counter(counter: Counter, title: str, metric: str) -> dict:
    """Унифицированная статистика по токенам."""
    most_common = counter.most_common(10)
    least_common = list(counter.most_common()[:-11:-1])
    return {
        "title": title,
        "metric": metric,
        "total_tokens": int(sum(counter.values())),
        "unique_tokens": int(len(counter)),
        "most_common": most_common,
        "least_common": least_common,
    }


def token_stats_from_matrix(
    title: str,
    X: sparse.csr_matrix,
    vectorizer,
    metric: str,
) -> dict:
    """Статистика по матрице признаков (Count/TF-IDF)."""
    features = np.array(vectorizer.get_feature_names_out())
    weights = np.asarray(X.sum(axis=0)).ravel()
    order = np.argsort(weights)

    most_idx = order[::-1][:10]
    least_idx = order[:10]

    if metric == "count_sum":
        most = [(features[i], int(weights[i])) for i in most_idx]
        least = [(features[i], int(weights[i])) for i in least_idx]
        total = int(weights.sum())
    else:
        most = [(features[i], float(weights[i])) for i in most_idx]
        least = [(features[i], float(weights[i])) for i in least_idx]
        total = float(weights.sum())

    return {
        "title": title,
        "metric": metric,
        "total_tokens": total,
        "unique_tokens": int(len(features)),
        "most_common": most,
        "least_common": least,
    }


def save_stats(stats: List[dict], path: Path) -> None:
    """Сохраняет статистику в JSON."""
    path.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")


def vectorize_count(texts: List[str]) -> Tuple[sparse.csr_matrix, CountVectorizer]:
    """Count‑Vectorizer с фиксированным токен‑паттерном."""
    vec = CountVectorizer(token_pattern=r"\b\w+\b", lowercase=False)
    X = vec.fit_transform(texts)
    return X, vec


def vectorize_tfidf(texts: List[str]) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    """TF‑IDF‑Vectorizer с тем же токен‑паттерном."""
    vec = TfidfVectorizer(token_pattern=r"\b\w+\b", lowercase=False)
    X = vec.fit_transform(texts)
    return X, vec


def vectorize_word2vec(tokens_list: List[List[str]]) -> Tuple[np.ndarray, Word2Vec]:
    """
    Обучаем Word2Vec и получаем вектор документа усреднением
    векторных представлений слов. Возвращаем массив (n_docs, vector_size)
    и обученную модель.
    """
    w2v = Word2Vec(
        sentences=tokens_list,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=1,               
        epochs=W2V_EPOCHS,
        seed=W2V_SEED,
    )

    doc_vecs = []
    for toks in tokens_list:
        valid = [t for t in toks if t in w2v.wv]
        if not valid:
            doc_vecs.append(np.zeros(W2V_VECTOR_SIZE))
        else:
            doc_vecs.append(np.mean(w2v.wv[valid], axis=0))
    return np.vstack(doc_vecs), w2v


def save_dataset(
    name: str,
    X,
    y: np.ndarray,
    vectorizer_meta: dict | None,
) -> dict:
    """
    Сохраняет признаки и метки и возвращает метаданные,
    которые потом собираются в `datasets_meta.json`.
    """
    entry = {"name": name}

    if sparse.issparse(X):
        data_path = DATASETS_DIR / f"{name}.npz"
        sparse.save_npz(data_path, X)
        entry["format"] = "sparse_npz"
    else:
        data_path = DATASETS_DIR / f"{name}.npy"
        np.save(data_path, X)
        entry["format"] = "dense_npy"
    entry["data_path"] = str(data_path)

    labels_path = DATASETS_DIR / f"{name}_labels.npy"
    np.save(labels_path, y)
    entry["labels_path"] = str(labels_path)

    if vectorizer_meta:
        meta_path = DATASETS_DIR / f"{name}_vectorizer.json"
        meta_path.write_text(
            json.dumps(vectorizer_meta, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        entry["vectorizer_path"] = str(meta_path)

    return entry


def generate_synthetic_by_similarity(
    X: np.ndarray,
    y: np.ndarray,
    per_class: int = 200,
    seed: int = 42,
    candidate_pool: int = 50,
    noise: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Синтетика без готовых алгоритмов:
    берём вектор, ищем похожий в своём классе по косинусному сходству,
    смешиваем их и добавляем небольшой шум.
    """
    rng = np.random.default_rng(seed)
    syn_X, syn_y = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue
        class_vecs = X[idx]
        norms = np.linalg.norm(class_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        class_norm = class_vecs / norms

        for _ in range(per_class):
            anchor = rng.integers(len(class_vecs))
            pool_size = min(candidate_pool, len(class_vecs) - 1)
            if pool_size < 1:
                break
            candidates = rng.choice(len(class_vecs), size=pool_size, replace=False)
            candidates = candidates[candidates != anchor]
            if len(candidates) == 0:
                continue
            sims = class_norm[candidates] @ class_norm[anchor]
            nearest = candidates[int(np.argmax(sims))]

            vec = (class_vecs[anchor] + class_vecs[nearest]) / 2.0
            vec += rng.normal(0, noise, size=vec.shape)
            syn_X.append(vec)
            syn_y.append(cls)

    if not syn_X:
        return X, y

    X_out = np.vstack([X, np.array(syn_X)])
    y_out = np.concatenate([y, np.array(syn_y)])
    return X_out, y_out


def maybe_generate_synthetic(
    name: str,
    X,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Опционально создаёт синтетический датасет для выбранного корпуса."""
    if SYNTHETIC_DATASET != name:
        return None
    if sparse.issparse(X):
        log.warning(
            f"Синтетика запрошена для разреженной матрицы `{name}`. "
            "Пропускаю генерацию."
        )
        return None
    return generate_synthetic_by_similarity(
        X,
        y,
        per_class=SYNTHETIC_PER_CLASS,
        seed=W2V_SEED,
        candidate_pool=SYNTHETIC_POOL,
        noise=SYNTHETIC_NOISE,
    )


# Основная часть
def main() -> None:
    """Выполняет всю цепочку предобработки и векторизации."""
    ensure_dirs()

    nltk.data.path.append(str(NLTK_DATA_DIR))
    nltk_download("punkt", quiet=True, download_dir=str(NLTK_DATA_DIR))
    if LANGUAGE == "english":
        nltk_download("wordnet", quiet=True, download_dir=str(NLTK_DATA_DIR))

    df = read_dataset(DATA_PATH)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise KeyError(
            f"В датасете должны быть колонки '{TEXT_COL}' и '{LABEL_COL}'. "
            f"Найденные колонки: {list(df.columns)}"
        )
    df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
    log.info(f"Загружено строк после dropna: {len(df)}")

    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    df = filter_by_length(df, TEXT_COL)

    raw_texts, stem_texts, lemma_texts, tokenized_raw = build_corpora(df[TEXT_COL])
    tokenized_stem = [t.split() for t in stem_texts]
    tokenized_lemma = [t.split() for t in lemma_texts]
    y_base = df[LABEL_COL].values

    raw_counter = build_token_counter(tokenized_raw)
    stem_counter = build_token_counter(tokenized_stem)
    lemma_counter = build_token_counter(tokenized_lemma)

    stats = [
        token_stats_from_counter(raw_counter, "raw_cleaned", metric="token_count"),
        token_stats_from_counter(stem_counter, "stemmed", metric="token_count"),
        token_stats_from_counter(lemma_counter, "lemmatized", metric="token_count"),
    ]
    save_stats(stats, CORPUS_STATS_PATH)
    log.info("Сохранена статистика по корпусам.")

    meta_entries = []
    dataset_stats = []

    X_cnt, cnt_vec = vectorize_count(raw_texts)
    meta_entries.append(
        save_dataset(
            "raw_count",
            X_cnt,
            y_base,
            {"type": "count", "vocab_size": len(cnt_vec.vocabulary_)},
        )
    )
    dataset_stats.append(token_stats_from_matrix("raw_count", X_cnt, cnt_vec, "count_sum"))
    syn = maybe_generate_synthetic("raw_count", X_cnt, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "raw_count_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "raw_count"},
            )
        )

    X_tfidf, tfidf_vec = vectorize_tfidf(raw_texts)
    meta_entries.append(
        save_dataset(
            "raw_tfidf",
            X_tfidf,
            y_base,
            {"type": "tfidf", "vocab_size": len(tfidf_vec.vocabulary_)},
        )
    )
    dataset_stats.append(token_stats_from_matrix("raw_tfidf", X_tfidf, tfidf_vec, "tfidf_sum"))
    syn = maybe_generate_synthetic("raw_tfidf", X_tfidf, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "raw_tfidf_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "raw_tfidf"},
            )
        )

    X_w2v, w2v_model = vectorize_word2vec(tokenized_raw)
    meta_entries.append(
        save_dataset(
            "raw_w2v",
            X_w2v,
            y_base,
            {
                "type": "word2vec",
                "vector_size": W2V_VECTOR_SIZE,
                "min_count": W2V_MIN_COUNT,
            },
        )
    )
    w2v_counter = filter_counter_by_vocab(raw_counter, w2v_model.wv.key_to_index)
    dataset_stats.append(token_stats_from_counter(w2v_counter, "raw_w2v", "token_count"))
    syn = maybe_generate_synthetic("raw_w2v", X_w2v, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "raw_w2v_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "raw_w2v"},
            )
        )
    w2v_path = DATASETS_DIR / "raw_w2v.model"
    w2v_model.save(str(w2v_path))
    log.info(f"Word2Vec‑модель сохранена в {w2v_path}")

    X_cnt, cnt_vec = vectorize_count(stem_texts)
    meta_entries.append(
        save_dataset(
            "stem_count",
            X_cnt,
            y_base,
            {"type": "count", "vocab_size": len(cnt_vec.vocabulary_)},
        )
    )
    dataset_stats.append(token_stats_from_matrix("stem_count", X_cnt, cnt_vec, "count_sum"))
    syn = maybe_generate_synthetic("stem_count", X_cnt, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "stem_count_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "stem_count"},
            )
        )

    X_tfidf, tfidf_vec = vectorize_tfidf(stem_texts)
    meta_entries.append(
        save_dataset(
            "stem_tfidf",
            X_tfidf,
            y_base,
            {"type": "tfidf", "vocab_size": len(tfidf_vec.vocabulary_)},
        )
    )
    dataset_stats.append(token_stats_from_matrix("stem_tfidf", X_tfidf, tfidf_vec, "tfidf_sum"))
    syn = maybe_generate_synthetic("stem_tfidf", X_tfidf, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "stem_tfidf_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "stem_tfidf"},
            )
        )

    X_w2v, w2v_model_stem = vectorize_word2vec([t.split() for t in stem_texts])
    meta_entries.append(
        save_dataset(
            "stem_w2v",
            X_w2v,
            y_base,
            {
                "type": "word2vec",
                "vector_size": W2V_VECTOR_SIZE,
                "min_count": W2V_MIN_COUNT,
            },
        )
    )
    stem_w2v_counter = filter_counter_by_vocab(
        stem_counter, w2v_model_stem.wv.key_to_index
    )
    dataset_stats.append(token_stats_from_counter(stem_w2v_counter, "stem_w2v", "token_count"))
    syn = maybe_generate_synthetic("stem_w2v", X_w2v, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "stem_w2v_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "stem_w2v"},
            )
        )

    X_cnt, cnt_vec = vectorize_count(lemma_texts)
    meta_entries.append(
        save_dataset(
            "lemma_count",
            X_cnt,
            y_base,
            {"type": "count", "vocab_size": len(cnt_vec.vocabulary_)},
        )
    )
    dataset_stats.append(token_stats_from_matrix("lemma_count", X_cnt, cnt_vec, "count_sum"))
    syn = maybe_generate_synthetic("lemma_count", X_cnt, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "lemma_count_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "lemma_count"},
            )
        )

    X_tfidf, tfidf_vec = vectorize_tfidf(lemma_texts)
    meta_entries.append(
        save_dataset(
            "lemma_tfidf",
            X_tfidf,
            y_base,
            {"type": "tfidf", "vocab_size": len(tfidf_vec.vocabulary_)},
        )
    )
    dataset_stats.append(token_stats_from_matrix("lemma_tfidf", X_tfidf, tfidf_vec, "tfidf_sum"))
    syn = maybe_generate_synthetic("lemma_tfidf", X_tfidf, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "lemma_tfidf_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "lemma_tfidf"},
            )
        )

    X_w2v, w2v_model_lemma = vectorize_word2vec([t.split() for t in lemma_texts])
    meta_entries.append(
        save_dataset(
            "lemma_w2v",
            X_w2v,
            y_base,
            {
                "type": "word2vec",
                "vector_size": W2V_VECTOR_SIZE,
                "min_count": W2V_MIN_COUNT,
            },
        )
    )
    lemma_w2v_counter = filter_counter_by_vocab(
        lemma_counter, w2v_model_lemma.wv.key_to_index
    )
    dataset_stats.append(token_stats_from_counter(lemma_w2v_counter, "lemma_w2v", "token_count"))
    syn = maybe_generate_synthetic("lemma_w2v", X_w2v, y_base)
    if syn:
        X_syn, y_syn = syn
        meta_entries.append(
            save_dataset(
                "lemma_w2v_synthetic",
                X_syn,
                y_syn,
                {"type": "synthetic", "base": "lemma_w2v"},
            )
        )

    meta_path = OUTPUT_DIR / "datasets_meta.json"
    meta_path.write_text(
        json.dumps(meta_entries, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    save_stats(dataset_stats, DATASETS_STATS_PATH)
    log.info(f"Все наборы данных сохранены в {DATASETS_DIR}")
    log.info(f"Метаданные записаны в {meta_path}")
    log.info(f"Статистика по 9 датасетам записана в {DATASETS_STATS_PATH}")


if __name__ == "__main__":
    main()
