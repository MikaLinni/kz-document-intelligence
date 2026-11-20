"""Search the fixed legal corpus with BM25, BGE-M3 LoRA or their RRF hybrid."""

import argparse
import os
import sys
import textwrap
from pathlib import Path

# The baseline loads its pinned base revision from the local Hugging Face cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from huggingface_hub.errors import LocalEntryNotFoundError
from peft import PeftModel
from rank_bm25 import BM25Okapi

from scripts.train_bge_m3_lora import (
    CHECKPOINT_DIR, CORPUS_PATH, CORPUS_SHA256, DOCUMENT_IDS, LANGUAGES,
    PASSAGE_BATCH, PASSAGE_LENGTH, QUERY_BATCH, QUERY_LENGTH, RRF_K,
    encode_texts, load_candidates, load_encoder, sha256, tokenize_bm25,
)

ADAPTER = CHECKPOINT_DIR / "epoch-02"
KAZAKH_LETTERS = frozenset("әғқңөұүһі")
KAZAKH_WORDS = frozenset("жер туралы бойынша кандай калай жане меншик укугы бап деген кезде".split())
METHOD_LABELS = {"bm25": "BM25", "lora": "BGE-M3 LoRA", "hybrid": "BM25 + LoRA RRF"}


def detect_language(question):
    normalized = question.lower()
    if KAZAKH_LETTERS.intersection(normalized) or KAZAKH_WORDS.intersection(tokenize_bm25(question)):
        return "kk"
    return "ru"


class LegalSearch:
    def __init__(self, method):
        self.method = method
        if sha256(CORPUS_PATH) != CORPUS_SHA256:
            raise ValueError("SHA-256 корпуса отличается от SHA-256 baseline_v1.")
        candidates = load_candidates()
        metadata = pd.read_parquet(
            CORPUS_PATH, columns=["chunk_id", "article_number", "article_heading"],
            filters=[("document_id", "in", list(DOCUMENT_IDS))],
        )
        joined = candidates.merge(metadata, on="chunk_id", how="left", validate="one_to_one", sort=False)
        self.passages = {
            language: joined.loc[joined.language == language].reset_index(drop=True)
            for language in LANGUAGES
        }
        self.lexical_indexes = {}
        self.passage_embeddings = {}
        if method != "bm25":
            missing = [name for name in ("adapter_config.json", "adapter_model.safetensors")
                       if not (ADAPTER / name).is_file()]
            if missing:
                raise FileNotFoundError(f"В {ADAPTER} отсутствуют {', '.join(missing)}.")
            print("Загружаю BGE-M3 и адаптер epoch-02 на MPS…", file=sys.stderr, flush=True)
            self.tokenizer, base = load_encoder()
            self.model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False).to("mps").eval()

    def prepare(self, language):
        passages = self.passages[language]
        if self.method != "lora" and language not in self.lexical_indexes:
            self.lexical_indexes[language] = BM25Okapi([tokenize_bm25(text) for text in passages.text])
        if self.method != "bm25" and language not in self.passage_embeddings:
            print(f"Строю dense-индекс {language.upper()} в памяти ({len(passages)} чанков)…",
                  file=sys.stderr, flush=True)
            self.passage_embeddings[language] = encode_texts(
                self.model, self.tokenizer, passages.text.tolist(), PASSAGE_LENGTH, PASSAGE_BATCH,
            )

    def search(self, question, language):
        self.prepare(language)
        if self.method != "lora":
            lexical = self.lexical_indexes[language].get_scores(tokenize_bm25(question))
            if self.method == "bm25":
                scores = lexical
        if self.method != "bm25":
            query = encode_texts(self.model, self.tokenizer, [question], QUERY_LENGTH, QUERY_BATCH)[0]
            dense = query @ self.passage_embeddings[language].T
            if self.method == "lora":
                scores = dense
            else:
                # Convert stable sorted positions to one-based ranks as in baseline_v1.
                lexical_ranks = np.argsort(np.argsort(-lexical, kind="stable")) + 1
                dense_ranks = np.argsort(np.argsort(-dense, kind="stable")) + 1
                scores = 1 / (RRF_K + lexical_ranks) + 1 / (RRF_K + dense_ranks)
        ranked = np.argsort(-scores, kind="stable")[:5]
        passages = self.passages[language]
        return [(float(scores[i]), passages.iloc[i]) for i in ranked]


def print_results(search, question, language):
    selected = detect_language(question) if language == "auto" else language
    results = search.search(question, selected)
    print(f"Запрос {question}")
    print(f"{METHOD_LABELS[search.method]} · {selected.upper()} · TOP-5")
    for rank, (score, row) in enumerate(results, 1):
        article = row.article_number if pd.notna(row.article_number) and row.article_number else "—"
        heading = row.article_heading if pd.notna(row.article_heading) and row.article_heading else "—"
        excerpt = textwrap.shorten(row.text, width=300, placeholder="…")
        print(f"{rank:>2}. score={score:.6f}  document_id={row.document_id}  article={article}")
        print(f"    Заголовок  {heading}")
        print(f"    {excerpt}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Вопрос на русском или казахском")
    parser.add_argument("--method", choices=METHOD_LABELS, default="hybrid")
    parser.add_argument("--lang", choices=("auto", *LANGUAGES), default="auto")
    parser.add_argument("--interactive", action="store_true", help="Несколько вопросов с одним индексом в памяти")
    args = parser.parse_args()
    question = args.question.strip() if args.question else ""
    if not args.interactive and not question:
        parser.error("Введите вопрос или укажите --interactive.")

    try:
        search = LegalSearch(args.method)
        if args.interactive:
            for language in LANGUAGES if args.lang == "auto" else (args.lang,):
                search.prepare(language)
        if question:
            print_results(search, question, args.lang)
        if args.interactive:
            print("Введите вопрос. Пустая строка завершает работу.", flush=True)
            while question := input("> ").strip():
                print_results(search, question, args.lang)
    except (EOFError, KeyboardInterrupt):
        print()
    except (LocalEntryNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка запуска демо. {error}\n")


if __name__ == "__main__":
    main()
