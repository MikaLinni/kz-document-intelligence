from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"


INPUT_PATHS = {
    "train": DATA_DIR / "retrieval_train_repaired.parquet",
    "dev": DATA_DIR / "retrieval_dev_repaired.parquet",
    "test": DATA_DIR / "retrieval_test_repaired.parquet",
}


OUTPUT_PATHS = {
    "train": DATA_DIR / "retrieval_train_clean.parquet",
    "dev": DATA_DIR / "retrieval_dev_clean.parquet",
    "test": DATA_DIR / "retrieval_test_clean.parquet",
}


# =========================================================
# CLEAN SELECTION
# =========================================================

def select_clean_rows(df):

    df = df.copy()

    # =====================================================
    # A. ARTICLE EXPLICITLY GIVEN IN QUESTION
    #
    # Берём только если exact article действительно
    # была найдена в corpus.
    #
    # Дополнительно требуем хотя бы умеренное
    # совпадение с answer/context.
    # =====================================================

    question_article_mask = (
        (df["used_article_filter"] == True)
        &
        (df["article_source"] == "question")
        &
        (df["repair_answer_similarity"] >= 0.30)
        &
        (df["repair_context_similarity"] >= 0.30)
    )


    # =====================================================
    # B. ARTICLE OBTAINED FROM ANSWER
    #
    # Здесь exact article также обязательна.
    # =====================================================

    answer_article_mask = (
        (df["used_article_filter"] == True)
        &
        (df["article_source"] == "answer")
        &
        (df["repair_answer_similarity"] >= 0.30)
        &
        (df["repair_context_similarity"] >= 0.30)
        &
        (df["repair_question_similarity"] >= 0.10)
    )


    # =====================================================
    # IMPORTANT:
    #
    # Semantic-only alignment больше НЕ используем
    # для supervised BGE-M3 training.
    #
    # Причина:
    # он уже показал false positives при ручной проверке.
    # =====================================================

    clean_mask = (
        question_article_mask
        |
        answer_article_mask
    )


    clean_df = (
        df[clean_mask]
        .copy()
        .reset_index(drop=True)
    )


    # =====================================================
    # WHY THIS ROW WAS KEPT
    # =====================================================

    def get_reason(row):

        if (
            row["article_source"]
            == "question"
        ):
            return "exact_article_from_question"

        return "exact_article_from_answer"


    clean_df["clean_reason"] = (
        clean_df.apply(
            get_reason,
            axis=1,
        )
    )


    # =====================================================
    # FINAL GOLD
    # =====================================================

    clean_df["gold_relevant_chunk_id"] = (
        clean_df["repaired_chunk_id"]
        .astype(str)
    )


    return clean_df


# =========================================================
# PROCESS
# =========================================================

clean_splits = {}


for split_name, input_path in INPUT_PATHS.items():

    print()
    print("=" * 80)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 80)

    df = pd.read_parquet(
        input_path
    )

    clean_df = select_clean_rows(
        df
    )

    clean_splits[
        split_name
    ] = clean_df


    print(
        "Original rows:",
        len(df)
    )

    print(
        "Clean rows:",
        len(clean_df)
    )

    print(
        "Coverage:",
        round(
            100
            * len(clean_df)
            / len(df),
            2,
        ),
        "%",
    )


    print()
    print(
        "Clean reason:"
    )

    print(
        clean_df[
            "clean_reason"
        ].value_counts()
    )


    print()
    print(
        "Languages:"
    )

    print(
        clean_df[
            "language"
        ].value_counts()
    )


    print()
    print(
        "Repair confidence:"
    )

    print(
        clean_df[
            "repair_confidence"
        ].value_counts()
    )


    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    output_path = (
        OUTPUT_PATHS[
            split_name
        ]
    )

    clean_df.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        "Saved:",
        output_path
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 80)
print("FINAL CLEAN DATASET")
print("=" * 80)


for split_name, df in clean_splits.items():

    original_count = len(
        pd.read_parquet(
            INPUT_PATHS[
                split_name
            ]
        )
    )

    print()
    print(
        split_name.upper()
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Coverage:",
        round(
            100
            * len(df)
            / original_count,
            2,
        ),
        "%",
    )


print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)


for split_name, path in OUTPUT_PATHS.items():
    print(
        split_name,
        "->",
        path
    )