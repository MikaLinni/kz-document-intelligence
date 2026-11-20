import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm


# =========================================================
# PROJECT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CORPUS_PATH,
    TRAIN_PATH,
    DEV_PATH,
    TEST_PATH,
)


# =========================================================
# OUTPUT PATHS
# =========================================================

TRAIN_REPAIRED_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval_train_repaired.parquet"
)

DEV_REPAIRED_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval_dev_repaired.parquet"
)

TEST_REPAIRED_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval_test_repaired.parquet"
)


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


# =========================================================
# ARTICLE EXTRACTION
# =========================================================

def extract_article_numbers(text):
    text = normalize_text(text)

    numbers = set()

    patterns = [
        # Kazakh: 639-бап
        r"\b(\d+(?:-\d+)*)\s*-\s*бап",

        # Russian: Статья 680
        r"\bстатья\s+(\d+(?:-\d+)*)",

        # Russian: ст. 465
        r"\bст\.?\s*(\d+(?:-\d+)*)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        numbers.update(matches)

    return numbers


def get_target_article(question, answer):
    question_articles = extract_article_numbers(
        question
    )

    # Question has priority.
    if len(question_articles) == 1:
        return (
            next(iter(question_articles)),
            "question",
        )

    # Use answer only when question has no article
    # and answer contains exactly one article.
    if len(question_articles) == 0:

        answer_articles = extract_article_numbers(
            answer
        )

        if len(answer_articles) == 1:
            return (
                next(iter(answer_articles)),
                "answer",
            )

    return None, "none"


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

dev_df = pd.read_parquet(
    DEV_PATH
)

test_df = pd.read_parquet(
    TEST_PATH
)


for df in [
    corpus_df,
    train_df,
    dev_df,
    test_df,
]:
    if "document_id" in df.columns:
        df["document_id"] = (
            df["document_id"]
            .astype(str)
        )


corpus_df["chunk_id"] = (
    corpus_df["chunk_id"]
    .astype(str)
)


for df in [
    train_df,
    dev_df,
    test_df,
]:
    df["relevant_chunk_id"] = (
        df["relevant_chunk_id"]
        .astype(str)
    )


corpus_df[
    "normalized_article_number"
] = (
    corpus_df["article_number"]
    .apply(
        normalize_article_number
    )
)


print(
    "Corpus:",
    len(corpus_df),
)

print(
    "Train:",
    len(train_df),
)

print(
    "Dev:",
    len(dev_df),
)

print(
    "Test:",
    len(test_df),
)


# =========================================================
# CANDIDATE CORPUS
#
# Only documents that occur in our benchmark.
# =========================================================

all_document_ids = set(
    train_df["document_id"].unique()
)

all_document_ids.update(
    dev_df["document_id"].unique()
)

all_document_ids.update(
    test_df["document_id"].unique()
)


candidate_df = corpus_df[
    corpus_df["document_id"].isin(
        all_document_ids
    )
].copy()


candidate_df = (
    candidate_df
    .drop_duplicates(
        subset=["chunk_id"]
    )
    .reset_index(drop=True)
)


print()
print(
    "Benchmark documents:",
    len(all_document_ids),
)

print(
    "Candidate chunks:",
    len(candidate_df),
)


candidate_texts = (
    candidate_df["text"]
    .fillna("")
    .astype(str)
    .map(normalize_text)
    .tolist()
)


# =========================================================
# COMBINE TEXTS FOR ALIGNMENT TF-IDF
#
# This is used only for gold-label construction.
# It is NOT BGE-M3 training.
# =========================================================

all_splits_df = pd.concat(
    [
        train_df.assign(
            split="train"
        ),
        dev_df.assign(
            split="dev"
        ),
        test_df.assign(
            split="test"
        ),
    ],
    ignore_index=True,
)


questions = (
    all_splits_df["question"]
    .fillna("")
    .astype(str)
    .map(normalize_text)
    .tolist()
)


answers = (
    all_splits_df["answer"]
    .fillna("")
    .astype(str)
    .map(normalize_text)
    .tolist()
)


contexts = (
    all_splits_df["context"]
    .fillna("")
    .astype(str)
    .map(normalize_text)
    .tolist()
)


# =========================================================
# TF-IDF
# =========================================================

print()
print("=" * 80)
print("BUILDING ALIGNMENT TF-IDF")
print("=" * 80)


fit_texts = (
    candidate_texts
    + questions
    + answers
    + contexts
)


vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=150_000,
    sublinear_tf=True,
    dtype=np.float32,
)


vectorizer.fit(
    fit_texts
)


candidate_matrix = normalize(
    vectorizer.transform(
        candidate_texts
    )
)


question_matrix = normalize(
    vectorizer.transform(
        questions
    )
)


answer_matrix = normalize(
    vectorizer.transform(
        answers
    )
)


context_matrix = normalize(
    vectorizer.transform(
        contexts
    )
)


print(
    "TF-IDF features:",
    candidate_matrix.shape[1],
)


# =========================================================
# DOCUMENT + LANGUAGE INDEX
#
# Для основного benchmark:
# RU query -> RU passage
# KK query -> KK passage
#
# Cross-lingual training сделаем позже отдельно.
# =========================================================

document_language_to_positions = {}


for (
    document_id,
    language
), group in candidate_df.groupby(
    [
        "document_id",
        "language",
    ]
):

    document_language_to_positions[
        (
            str(document_id),
            str(language),
        )
    ] = group.index.to_numpy()


# =========================================================
# REPAIR FUNCTION
# =========================================================

def repair_dataframe(
    df,
    matrix_offset,
    split_name,
):

    records = []

    print()
    print("=" * 80)
    print(
        f"REPAIRING {split_name.upper()}"
    )
    print("=" * 80)


    for local_i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name,
    ):

        global_i = (
            matrix_offset
            + local_i
        )

        document_id = str(
            row["document_id"]
        )


        language = str(
            row["language"]
        )


        document_positions = (
            document_language_to_positions.get(
                (
                    document_id,
                    language,
                ),
                np.array(
                    [],
                    dtype=int,
                ),
            )
        )


        if len(document_positions) == 0:
            continue


        # ---------------------------------------------
        # Structural evidence
        # ---------------------------------------------

        target_article, article_source = (
            get_target_article(
                row["question"],
                row["answer"],
            )
        )


        used_article_filter = False

        positions = (
            document_positions
        )


        # ---------------------------------------------
        # Exact article filter
        # ---------------------------------------------

        if target_article is not None:

            exact_positions = []


            for pos in document_positions:

                candidate_article = (
                    candidate_df.iloc[
                        pos
                    ][
                        "normalized_article_number"
                    ]
                )


                if (
                    candidate_article
                    == target_article
                ):
                    exact_positions.append(
                        pos
                    )


            if exact_positions:

                positions = np.array(
                    exact_positions,
                    dtype=int,
                )

                used_article_filter = True


        # ---------------------------------------------
        # Similarities
        # ---------------------------------------------

        candidate_vectors = (
            candidate_matrix[
                positions
            ]
        )


        question_scores = (
            question_matrix[
                global_i
            ]
            @ candidate_vectors.T
        ).toarray().ravel()


        answer_scores = (
            answer_matrix[
                global_i
            ]
            @ candidate_vectors.T
        ).toarray().ravel()


        context_scores = (
            context_matrix[
                global_i
            ]
            @ candidate_vectors.T
        ).toarray().ravel()


        final_scores = (
            0.45 * answer_scores
            +
            0.30 * context_scores
            +
            0.25 * question_scores
        )


        order = np.argsort(
            final_scores
        )[::-1]


        best_local = (
            order[0]
        )


        if len(order) > 1:
            second_local = (
                order[1]
            )
        else:
            second_local = (
                order[0]
            )


        best_position = (
            positions[
                best_local
            ]
        )


        second_position = (
            positions[
                second_local
            ]
        )


        best_row = (
            candidate_df.iloc[
                best_position
            ]
        )


        second_row = (
            candidate_df.iloc[
                second_position
            ]
        )


        best_score = float(
            final_scores[
                best_local
            ]
        )


        second_score = float(
            final_scores[
                second_local
            ]
        )


        if len(order) > 1:

            margin = (
                best_score
                -
                second_score
            )

        else:
            margin = 1.0


        original_chunk_id = str(
            row[
                "relevant_chunk_id"
            ]
        )


        repaired_chunk_id = str(
            best_row[
                "chunk_id"
            ]
        )


        proposed_article_number = (
            best_row[
                "normalized_article_number"
            ]
        )


        article_correct = False


        if target_article is not None:

            article_correct = (
                target_article
                ==
                proposed_article_number
            )


        record = row.to_dict()


        record.update(
            {
                # Keep original gold
                "original_relevant_chunk_id":
                    original_chunk_id,

                # Proposed repaired gold
                "repaired_chunk_id":
                    repaired_chunk_id,

                "repair_changed":
                    (
                        original_chunk_id
                        != repaired_chunk_id
                    ),

                # Structural info
                "target_article":
                    (
                        target_article
                        if target_article
                        else ""
                    ),

                "article_source":
                    article_source,

                "used_article_filter":
                    used_article_filter,

                "repaired_article_number":
                    proposed_article_number,

                "article_correct":
                    article_correct,

                # Alignment scores
                "repair_best_score":
                    best_score,

                "repair_second_score":
                    second_score,

                "repair_margin":
                    margin,

                "repair_question_similarity":
                    float(
                        question_scores[
                            best_local
                        ]
                    ),

                "repair_answer_similarity":
                    float(
                        answer_scores[
                            best_local
                        ]
                    ),

                "repair_context_similarity":
                    float(
                        context_scores[
                            best_local
                        ]
                    ),

                # Texts
                "repaired_positive_text":
                    str(
                        best_row[
                            "text"
                        ]
                    ),

                "second_candidate_chunk_id":
                    str(
                        second_row[
                            "chunk_id"
                        ]
                    ),
            }
        )


        records.append(
            record
        )


    repaired_df = pd.DataFrame(
        records
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    def assign_confidence(row):

        # Exact structural article found
        if row[
            "used_article_filter"
        ]:

            if (
                row[
                    "repair_answer_similarity"
                ] >= 0.45

                and

                row[
                    "repair_context_similarity"
                ] >= 0.45
            ):
                return "high"


            if (
                row[
                    "repair_answer_similarity"
                ] >= 0.30

                and

                row[
                    "repair_context_similarity"
                ] >= 0.35
            ):
                return "medium"


            return "low"


        # No article evidence
        if (
            row[
                "repair_answer_similarity"
            ] >= 0.60

            and

            row[
                "repair_context_similarity"
            ] >= 0.55

            and

            row[
                "repair_margin"
            ] >= 0.05
        ):
            return "high"


        if (
            row[
                "repair_answer_similarity"
            ] >= 0.45

            and

            row[
                "repair_context_similarity"
            ] >= 0.45

            and

            row[
                "repair_margin"
            ] >= 0.02
        ):
            return "medium"


        return "low"


    repaired_df[
        "repair_confidence"
    ] = repaired_df.apply(
        assign_confidence,
        axis=1,
    )


    return repaired_df


# =========================================================
# PROCESS SPLITS
# =========================================================

train_offset = 0

dev_offset = len(
    train_df
)

test_offset = (
    len(train_df)
    +
    len(dev_df)
)


train_repaired = repair_dataframe(
    train_df.reset_index(drop=True),
    train_offset,
    "train",
)


dev_repaired = repair_dataframe(
    dev_df.reset_index(drop=True),
    dev_offset,
    "dev",
)


test_repaired = repair_dataframe(
    test_df.reset_index(drop=True),
    test_offset,
    "test",
)


# =========================================================
# SAVE
# =========================================================

train_repaired.to_parquet(
    TRAIN_REPAIRED_PATH,
    index=False,
)

dev_repaired.to_parquet(
    DEV_REPAIRED_PATH,
    index=False,
)

test_repaired.to_parquet(
    TEST_REPAIRED_PATH,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

def print_summary(
    name,
    df,
):

    print()
    print("=" * 80)
    print(
        name.upper()
    )
    print("=" * 80)

    print(
        "Rows:",
        len(df),
    )

    print(
        "Changed:",
        int(
            df[
                "repair_changed"
            ].sum()
        ),
    )

    print(
        "Changed %:",
        round(
            100
            * df[
                "repair_changed"
            ].mean(),
            2,
        ),
    )


    print(
        "Structural article evidence:",
        int(
            (
                df[
                    "target_article"
                ]
                != ""
            ).sum()
        ),
    )


    print(
        "Article-filtered:",
        int(
            df[
                "used_article_filter"
            ].sum()
        ),
    )


    print()
    print(
        "Confidence:"
    )

    print(
        df[
            "repair_confidence"
        ].value_counts()
    )


    print()
    print(
        "Article source:"
    )

    print(
        df[
            "article_source"
        ].value_counts()
    )


print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print_summary(
    "train",
    train_repaired,
)

print_summary(
    "dev",
    dev_repaired,
)

print_summary(
    "test",
    test_repaired,
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(
    TRAIN_REPAIRED_PATH
)

print(
    DEV_REPAIRED_PATH
)

print(
    TEST_REPAIRED_PATH
)