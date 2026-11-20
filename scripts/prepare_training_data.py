import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from tqdm import tqdm


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

CORPUS_PATH = DATA_DIR / "legal_chunks_20k.parquet"
TRAIN_PATH = DATA_DIR / "retrieval_train_clean.parquet"

OUTPUT_PATH = DATA_DIR / "bge_m3_training_records_clean.jsonl"


# =========================================================
# SETTINGS
# =========================================================

NEGATIVES_PER_QUERY = 3

BM25_POOL_SIZE = 50

# Не берём самые верхние несколько результатов:
# среди них выше риск false negatives.
SKIP_TOP = 3

SEED = 42

np.random.seed(SEED)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()

    for char in [
        "\u2010",
        "\u2011",
        "\u2012",
        "\u2013",
        "\u2014",
        "\u2212",
    ]:
        text = text.replace(char, "-")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_article_number(value):
    if pd.isna(value):
        return ""

    value = normalize_text(value)

    match = re.search(
        r"(\d+(?:-\d+)*)",
        value,
    )

    if match:
        return match.group(1)

    return ""


def tokenize(text):
    return re.findall(
        r"\w+",
        normalize_text(text),
    )


def answer_overlap(answer, candidate):
    """
    Доля слов ответа, присутствующих в candidate.
    Используем для отсечения потенциальных false negatives.
    """

    answer_tokens = set(
        tokenize(answer)
    )

    candidate_tokens = set(
        tokenize(candidate)
    )

    if not answer_tokens:
        return 0.0

    return (
        len(
            answer_tokens
            & candidate_tokens
        )
        /
        len(answer_tokens)
    )


# =========================================================
# LOAD
# =========================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

corpus_df = pd.read_parquet(
    CORPUS_PATH
)

train_df = pd.read_parquet(
    TRAIN_PATH
)


corpus_df["chunk_id"] = (
    corpus_df["chunk_id"]
    .astype(str)
)

corpus_df["document_id"] = (
    corpus_df["document_id"]
    .astype(str)
)

train_df["document_id"] = (
    train_df["document_id"]
    .astype(str)
)

train_df["gold_relevant_chunk_id"] = (
    train_df["gold_relevant_chunk_id"]
    .astype(str)
)


corpus_df["normalized_article_number"] = (
    corpus_df["article_number"]
    .apply(
        normalize_article_number
    )
)


print(
    "Full corpus:",
    len(corpus_df)
)

print(
    "Clean train queries:",
    len(train_df)
)


# =========================================================
# GOLD LOOKUP
# =========================================================

chunk_lookup = (
    corpus_df
    .set_index("chunk_id")
    .to_dict("index")
)


missing_gold = (
    ~train_df[
        "gold_relevant_chunk_id"
    ].isin(
        set(
            corpus_df["chunk_id"]
        )
    )
).sum()


print(
    "Missing gold chunks:",
    missing_gold
)


if missing_gold > 0:
    raise ValueError(
        "Some gold chunks are missing from corpus."
    )


# =========================================================
# NEGATIVE MINING CORPUS
#
# Используем документы, встречающиеся в clean train.
# Это даёт сложные legal negatives и не требует держать
# BM25 по всем 305k chunks в RAM.
# =========================================================

train_document_ids = set(
    train_df[
        "document_id"
    ].unique()
)


mining_df = corpus_df[
    corpus_df[
        "document_id"
    ].isin(
        train_document_ids
    )
].copy()


mining_df = (
    mining_df
    .drop_duplicates(
        subset=["chunk_id"]
    )
    .reset_index(drop=True)
)


print()
print("=" * 80)
print("NEGATIVE MINING CORPUS")
print("=" * 80)

print(
    "Train documents:",
    len(train_document_ids)
)

print(
    "Mining chunks:",
    len(mining_df)
)

print(
    "Languages:"
)

print(
    mining_df[
        "language"
    ].value_counts()
)


# =========================================================
# BUILD SEPARATE BM25 INDEX FOR EACH LANGUAGE
#
# Same-language negatives are harder and more useful
# than trivially different-script negatives.
# =========================================================

bm25_indexes = {}


print()
print("=" * 80)
print("BUILDING BM25 INDEXES")
print("=" * 80)


for language in sorted(
    mining_df["language"]
    .dropna()
    .unique()
):

    lang_df = (
        mining_df[
            mining_df["language"]
            == language
        ]
        .copy()
        .reset_index(drop=True)
    )


    texts = (
        lang_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    tokenized = [
        tokenize(text)
        for text in tqdm(
            texts,
            desc=f"Tokenizing {language}",
        )
    ]


    bm25 = BM25Okapi(
        tokenized
    )


    bm25_indexes[
        language
    ] = {
        "df": lang_df,
        "texts": texts,
        "bm25": bm25,
    }


    print(
        language,
        "chunks:",
        len(lang_df),
    )


# =========================================================
# MINE NEGATIVES
# =========================================================

print()
print("=" * 80)
print("MINING SEMI-HARD NEGATIVES")
print("=" * 80)


training_records = []

skipped_no_index = 0
skipped_not_enough = 0


for _, row in tqdm(
    train_df.iterrows(),
    total=len(train_df),
    desc="Mining negatives",
):

    query = str(
        row["question"]
    ).strip()

    answer = str(
        row["answer"]
    ).strip()

    language = str(
        row["language"]
    )

    positive_id = str(
        row[
            "gold_relevant_chunk_id"
        ]
    )


    positive_info = (
        chunk_lookup[
            positive_id
        ]
    )


    positive_text = str(
        positive_info[
            "text"
        ]
    )


    positive_document_id = str(
        positive_info[
            "document_id"
        ]
    )


    positive_article = (
        normalize_article_number(
            positive_info[
                "article_number"
            ]
        )
    )


    if language not in bm25_indexes:
        skipped_no_index += 1
        continue


    lang_data = (
        bm25_indexes[
            language
        ]
    )


    lang_df = (
        lang_data[
            "df"
        ]
    )

    bm25 = (
        lang_data[
            "bm25"
        ]
    )


    scores = bm25.get_scores(
        tokenize(query)
    )


    pool_size = min(
        BM25_POOL_SIZE,
        len(scores),
    )


    # Быстро находим top pool_size.
    candidate_positions = np.argpartition(
        scores,
        -pool_size,
    )[
        -pool_size:
    ]


    # Сортируем найденный pool по score.
    candidate_positions = (
        candidate_positions[
            np.argsort(
                scores[
                    candidate_positions
                ]
            )[::-1]
        ]
    )


    # Semi-hard:
    # первые несколько BM25 результатов
    # специально пропускаем.
    candidate_positions = (
        candidate_positions[
            SKIP_TOP:
        ]
    )


    negatives = []
    negative_ids = []


    for position in candidate_positions:

        candidate = (
            lang_df.iloc[
                position
            ]
        )


        negative_id = str(
            candidate[
                "chunk_id"
            ]
        )


        negative_text = str(
            candidate[
                "text"
            ]
        )


        negative_document_id = str(
            candidate[
                "document_id"
            ]
        )


        negative_article = (
            normalize_article_number(
                candidate[
                    "article_number"
                ]
            )
        )


        # ---------------------------------------------
        # 1. Не gold
        # ---------------------------------------------

        if negative_id == positive_id:
            continue


        # ---------------------------------------------
        # 2. Не точный duplicate
        # ---------------------------------------------

        if (
            normalize_text(
                negative_text
            )
            ==
            normalize_text(
                positive_text
            )
        ):
            continue


        # ---------------------------------------------
        # 3. Если это тот же document + та же статья,
        # есть высокий риск, что это ещё один
        # реально релевантный passage.
        # ---------------------------------------------

        if (
            negative_document_id
            == positive_document_id
            and
            positive_article
            and
            negative_article
            == positive_article
        ):
            continue


        # ---------------------------------------------
        # 4. Если chunk содержит слишком много
        # слов gold answer, он может быть
        # false negative.
        # ---------------------------------------------

        overlap = answer_overlap(
            answer,
            negative_text,
        )


        if overlap >= 0.55:
            continue


        # ---------------------------------------------
        # 5. No duplicates
        # ---------------------------------------------

        if negative_id in negative_ids:
            continue


        negative_ids.append(
            negative_id
        )

        negatives.append(
            negative_text
        )


        if (
            len(negatives)
            >= NEGATIVES_PER_QUERY
        ):
            break


    if (
        len(negatives)
        <
        NEGATIVES_PER_QUERY
    ):
        skipped_not_enough += 1
        continue


    training_records.append(
        {
            "query": query,

            "positive_id":
                positive_id,

            "positive":
                positive_text,

            "negative_ids":
                negative_ids,

            "negatives":
                negatives,

            "language":
                language,

            "document_id":
                str(
                    row[
                        "document_id"
                    ]
                ),

            "context_id":
                str(
                    row[
                        "context_id"
                    ]
                ),

            "clean_reason":
                str(
                    row[
                        "clean_reason"
                    ]
                ),

            "repair_confidence":
                str(
                    row[
                        "repair_confidence"
                    ]
                ),
        }
    )


# =========================================================
# SAVE
# =========================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    for record in training_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            +
            "\n"
        )


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 80)
print("FINAL TRAINING DATA")
print("=" * 80)


print(
    "Clean train input:",
    len(train_df)
)

print(
    "Training records:",
    len(training_records)
)

print(
    "Skipped: no BM25 index:",
    skipped_no_index
)

print(
    "Skipped: not enough safe negatives:",
    skipped_not_enough
)


if training_records:

    stats_df = pd.DataFrame(
        training_records
    )


    print()
    print("Languages:")

    print(
        stats_df[
            "language"
        ].value_counts()
    )


    print()
    print("Clean reason:")

    print(
        stats_df[
            "clean_reason"
        ].value_counts()
    )


print()
print(
    "Saved:",
    OUTPUT_PATH
)


# =========================================================
# EXAMPLES
# =========================================================

print()
print("=" * 80)
print("TRAINING EXAMPLES")
print("=" * 80)


for i, record in enumerate(
    training_records[:5],
    start=1,
):

    print()
    print("#" * 80)
    print(
        f"EXAMPLE {i}"
    )
    print("#" * 80)


    print()
    print("QUERY:")
    print(
        record["query"]
    )


    print()
    print("POSITIVE:")
    print(
        record[
            "positive"
        ][:700]
    )


    for j, negative in enumerate(
        record[
            "negatives"
        ],
        start=1,
    ):

        print()
        print(
            f"NEGATIVE {j}:"
        )

        print(
            negative[:500]
        )