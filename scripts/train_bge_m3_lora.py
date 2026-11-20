"""Train the frozen v1 dense retrieval baseline. Run with python -m scripts.train_bge_m3_lora."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import re
import tempfile
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import CORPUS_PATH, DATA_DIR, MODELS_DIR, RESULTS_DIR

FROZEN_DIR = DATA_DIR / "frozen" / "baseline_v1"
RUN_DIR = RESULTS_DIR / "baseline_v1_lora"
CHECKPOINT_DIR = MODELS_DIR / "baseline_v1_lora"
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DOCUMENT_IDS = ("1", "3559", "4054", "7833", "15924", "16159", "26556")
LANGUAGES = ("ru", "kk")
CORPUS_SHA256 = "b65485a62e7f9707049bd325f6c3b900c1b96eee565b3ac35fbedc10fc41515a"
SEED = 42
EPOCHS = 5
PATIENCE = 2
ACCUMULATION = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
QUERY_LENGTH, PASSAGE_LENGTH = 128, 320
QUERY_BATCH, PASSAGE_BATCH = 16, 8
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
TARGETS = ("query", "value")
RRF_K = 60
DASHES = str.maketrans({dash: "-" for dash in "‐‑‒–—−"})


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs():
    expected = {
        "retrieval_train_clean.parquet", "retrieval_dev_clean.parquet",
        "retrieval_test_clean.parquet", "bge_m3_training_records_clean.jsonl",
        "repair_all_splits.py", "build_clean_splits.py", "prepare_training_data.py",
    }
    manifest = FROZEN_DIR / "SHA256SUMS.txt"
    hashes = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        name = Path(name).name
        if name not in expected or name in hashes or sha256(FROZEN_DIR / name) != digest:
            raise ValueError(f"Invalid frozen manifest entry: {name}")
        hashes[name] = digest
    if set(hashes) != expected:
        raise ValueError("Frozen manifest is incomplete")
    if sha256(CORPUS_PATH) != CORPUS_SHA256:
        raise ValueError(f"Unexpected corpus SHA-256: {CORPUS_PATH}")
    return hashes


def load_candidates():
    columns = ["chunk_id", "document_id", "language", "text"]
    corpus = pd.read_parquet(CORPUS_PATH, columns=columns)
    candidates = corpus.loc[corpus.document_id.astype(str).isin(DOCUMENT_IDS)].copy()
    candidates["chunk_id"] = candidates.chunk_id.astype(str)
    candidates["document_id"] = candidates.document_id.astype(str)
    candidates = candidates.sort_values("chunk_id").reset_index(drop=True)
    if (
        len(candidates) != 3009 or not candidates.chunk_id.is_unique
        or candidates.language.value_counts().to_dict() != {"ru": 1692, "kk": 1317}
        or set(candidates.document_id) != set(DOCUMENT_IDS)
        or not candidates.text.fillna("").str.strip().ne("").all()
    ):
        raise ValueError("Unexpected fixed evaluation corpus")
    return candidates


def load_queries(split, candidates):
    columns = ["question", "language", "gold_relevant_chunk_id"]
    queries = pd.read_parquet(FROZEN_DIR / f"retrieval_{split}_clean.parquet", columns=columns)
    queries["gold_relevant_chunk_id"] = queries.gold_relevant_chunk_id.astype(str)
    gold = set(zip(queries.language, queries.gold_relevant_chunk_id))
    available = set(zip(candidates.language, candidates.chunk_id))
    if not gold <= available:
        raise ValueError(f"{split}: {len(gold - available)} missing same-language gold pairs")
    if queries.empty or not queries.question.fillna("").str.strip().ne("").all():
        raise ValueError(f"{split}: empty questions")
    return queries


def load_encoder():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Run in a macOS terminal with Metal access.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True)
    encoder = AutoModel.from_pretrained(
        MODEL, revision=REVISION, attn_implementation="sdpa", add_pooling_layer=False,
    )
    if not tokenizer.is_fast or tokenizer.padding_side != "right":
        raise ValueError("Expected the fast BGE-M3 tokenizer with right padding")
    matched = {
        name for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS
    }
    expected = {
        f"encoder.layer.{layer}.attention.self.{target}"
        for layer in range(24) for target in TARGETS
    }
    if encoder.config.model_type != "xlm-roberta" or encoder.config.hidden_size != 1024 or matched != expected:
        raise ValueError("Unexpected BGE-M3 encoder or LoRA target modules")
    encoder.config.use_cache = False
    return tokenizer, encoder.to("mps")


def trainable_encoder(encoder, checkpoint=None):
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if checkpoint:
        model = PeftModel.from_pretrained(encoder, checkpoint, is_trainable=True)
    else:
        config = LoraConfig(
            **LORA, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=list(TARGETS), revision=REVISION,
        )
        model = get_peft_model(encoder, config)
    model.to("mps")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if len(trainable) != 96 or any(".lora_" not in name for name, _ in trainable):
        raise ValueError("Unexpected trainable parameters")
    return model, [p for _, p in trainable]


def dense_embedding(model, batch):
    inputs = {key: tensor.to("mps") for key, tensor in batch.items()}
    hidden = model(**inputs, return_dict=True).last_hidden_state
    return F.normalize(hidden[:, 0].float(), p=2, dim=-1)


def tokenize_training(tokenizer, texts, max_length):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def compute_loss(model, tokenizer, record):
    query = tokenize_training(tokenizer, [record["query"]], QUERY_LENGTH)
    passages = [record["positive"], *record["negatives"]]
    batch = tokenize_training(tokenizer, passages, PASSAGE_LENGTH)
    scores = dense_embedding(model, query) @ dense_embedding(model, batch).T
    return F.cross_entropy(scores / TEMPERATURE, torch.zeros(1, dtype=torch.long, device="mps"))


@torch.inference_mode()
def encode_texts(model, tokenizer, texts, max_length, batch_size):
    model.eval()
    tokens = tokenizer(texts, truncation=True, max_length=max_length)
    order = sorted(range(len(texts)), key=lambda index: len(tokens["input_ids"][index]))
    embeddings = np.empty((len(texts), 1024), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        examples = [{key: values[index] for key, values in tokens.items()} for index in indices]
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
        embeddings[indices] = dense_embedding(model, batch).cpu().numpy()
    if not np.isfinite(embeddings).all():
        raise ValueError("Dense retrieval produced non-finite embeddings")
    return embeddings


def tokenize_bm25(text):
    normalized = unicodedata.normalize("NFKC", text).lower().translate(DASHES)
    return re.findall(r"\w+", normalized)


def bm25_rankings(candidates, queries):
    index = BM25Okapi([tokenize_bm25(text) for text in candidates.text])
    scores = np.stack([index.get_scores(tokenize_bm25(question)) for question in queries.question])
    return np.argsort(-scores, axis=1, kind="stable")


def dense_rankings(model, tokenizer, candidates, queries):
    passages = encode_texts(model, tokenizer, candidates.text.tolist(), PASSAGE_LENGTH, PASSAGE_BATCH)
    encoded = encode_texts(model, tokenizer, queries.question.tolist(), QUERY_LENGTH, QUERY_BATCH)
    return np.argsort(-(encoded @ passages.T), axis=1, kind="stable")


def fuse_rankings(left, right):
    ranks_left = np.argsort(left, axis=1)
    ranks_right = np.argsort(right, axis=1)
    scores = 1 / (RRF_K + 1 + ranks_left) + 1 / (RRF_K + 1 + ranks_right)
    return np.argsort(-scores, axis=1, kind="stable")


def evaluate(candidates, queries, model, tokenizer, dense_name, lexical=None):
    metrics = {"bm25": {}, dense_name: {}, f"bm25_rrf_{dense_name}": {}}
    if lexical is None:
        lexical = {}
    for language in LANGUAGES:
        passages = candidates.loc[candidates.language == language]
        questions = queries.loc[queries.language == language]
        if questions.empty:
            continue
        if language not in lexical:
            lexical[language] = bm25_rankings(passages, questions)
        bm25 = lexical[language]
        print(f"Evaluating {dense_name} {language}: {len(questions)} queries, {len(passages)} chunks", flush=True)
        dense = dense_rankings(model, tokenizer, passages, questions)
        hybrid = fuse_rankings(bm25, dense)
        ids = passages.chunk_id.to_numpy()
        gold = questions.gold_relevant_chunk_id.to_numpy()
        for name, ranking in zip(metrics, (bm25, dense, hybrid)):
            hits = ids[ranking[:, :10]] == gold[:, None]
            ranks = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, np.inf)
            metrics[name][language] = {
                "queries": len(gold),
                "recall@5": float(np.mean(ranks <= 5)),
                "recall@10": float(np.mean(ranks <= 10)),
                "mrr@10": float(np.mean(1 / ranks)),
                "ndcg@10": float(np.mean(1 / np.log2(ranks + 1))),
            }
    for by_language in metrics.values():
        groups = list(by_language.values())
        n = sum(group["queries"] for group in groups)
        by_language["overall"] = {"queries": n}
        for metric in ("recall@5", "recall@10"):
            hits = sum(round(group[metric] * group["queries"]) for group in groups)
            by_language["overall"][metric] = hits / n
        for metric in ("mrr@10", "ndcg@10"):
            by_language["overall"][metric] = sum(group[metric] * group["queries"] for group in groups) / n
    return metrics


def experiment_config(hashes):
    return {
        "seed": SEED, "base_model": MODEL, "base_revision": REVISION,
        "device": "mps", "dtype": "float32", "attention": "sdpa", "pooling": "normalized_cls",
        "versions": {
            "python": platform.python_version(), "macOS": platform.mac_ver()[0],
            **{p: version(p) for p in ("torch", "transformers", "peft", "accelerate", "rank-bm25",
                                      "numpy", "pandas", "pyarrow", "tokenizers", "huggingface-hub")},
        },
        "frozen_sha256": hashes, "corpus_sha256": CORPUS_SHA256,
        "scripts_sha256": {path.name: sha256(path) for path in (Path(__file__), Path(__file__).with_name("evaluate_test.py"))},
        "candidate_document_ids": list(DOCUMENT_IDS), "candidate_counts": {"ru": 1692, "kk": 1317},
        "candidate_filter": "same language across all seven documents; ties by ascending chunk_id",
        "query_max_length": QUERY_LENGTH, "passage_max_length": PASSAGE_LENGTH,
        "query_batch_size": QUERY_BATCH, "passage_batch_size": PASSAGE_BATCH,
        "bm25": {
            "k1": 1.5, "b": 0.75, "epsilon": 0.25,
            "tokenization": "NFKC, lowercase, dash normalization, Unicode word tokens",
        },
        "rrf_k": RRF_K, "rrf_depth": "all same-language candidates",
        "lora": {**LORA, "target_modules": list(TARGETS), "bias": "none", "trainable_parameters": 1572864},
        "gradient_checkpointing": {"use_reentrant": False},
        "max_epochs": EPOCHS, "patience": PATIENCE, "microbatch_queries": 1,
        "negatives_per_query": 3, "gradient_accumulation": ACCUMULATION,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE, "optimizer": "AdamW", "max_grad_norm": MAX_GRAD_NORM,
        "scheduler": "linear", "warmup_ratio": WARMUP_RATIO,
        "selection": "strict DEV Recall@10 improvement; earliest epoch wins ties",
        "patience_reference": "zero-shot or an improving LoRA epoch",
    }


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_epoch(checkpoint_root, epoch, model, optimizer, scheduler, metrics, loss):
    with tempfile.TemporaryDirectory(dir=checkpoint_root) as temporary:
        path = Path(temporary) / "checkpoint"
        model.save_pretrained(path, save_embedding_layers=False)
        torch.save({
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
        }, path / "resume.pt")
        write_json(path / "dev.json", {"epoch": epoch, "mean_loss": loss, "metrics": metrics})
        path.rename(checkpoint_root / f"epoch-{epoch:02d}")


def train_epoch(model, tokenizer, records, optimizer, scheduler, parameters, epoch):
    model.train()
    order = list(range(len(records)))
    random.Random(SEED + epoch).shuffle(order)
    running = torch.zeros((), device="mps")
    started = time.monotonic()
    for start in range(0, len(order), ACCUMULATION):
        batch = order[start:start + ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        for index in batch:
            loss = compute_loss(model, tokenizer, records[index])
            running += loss.detach()
            (loss / len(batch)).backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step = start // ACCUMULATION + 1
        if step == 1 or step % 10 == 0 or start + len(batch) == len(order):
            seen = start + len(batch)
            print(f"epoch={epoch} records={seen}/{len(order)} step={step} "
                  f"loss={running.item() / seen:.4f} grad={norm.item():.4f} "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    return running.item() / len(records)


def smoke_test(model, tokenizer, records, dev, candidates):
    queries = dev.groupby("language", sort=False).head(2)
    other = candidates.groupby("language", sort=False).head(12)
    passages = candidates[candidates.chunk_id.isin(set(other.chunk_id) | set(queries.gold_relevant_chunk_id))]
    zero = evaluate(passages, queries, model, tokenizer, "zero_shot")
    probe = queries.question.tolist()
    initial = encode_texts(model, tokenizer, probe, QUERY_LENGTH, QUERY_BATCH)
    model, parameters = trainable_encoder(model)
    if sum(p.numel() for p in parameters) != 1572864:
        raise ValueError("Unexpected LoRA parameter count")
    np.testing.assert_allclose(
        encode_texts(model, tokenizer, probe, QUERY_LENGTH, QUERY_BATCH), initial, rtol=0, atol=1e-6,
    )
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    subset = [next(r for r in records if r["language"] == language) for language in LANGUAGES]
    loss = train_epoch(model, tokenizer, subset, optimizer, torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
                       parameters, 1)
    adapted = encode_texts(model, tokenizer, probe, QUERY_LENGTH, QUERY_BATCH)
    if not math.isfinite(loss) or np.array_equal(initial, adapted):
        raise ValueError("Smoke test did not produce a finite, effective adapter update")
    with tempfile.TemporaryDirectory() as temporary:
        model.save_pretrained(temporary, save_embedding_layers=False)
        # Load the serialized adapter under another name in the same encoder.
        model.load_adapter(temporary, adapter_name="smoke_reload")
        model.set_adapter("smoke_reload")
        np.testing.assert_allclose(
            encode_texts(model, tokenizer, probe, QUERY_LENGTH, QUERY_BATCH), adapted, rtol=0, atol=1e-6,
        )
    tuned = evaluate(passages, queries, model, tokenizer, "lora")
    return {"records": len(subset), "queries": len(queries), "candidates": len(passages),
            "mean_training_loss": loss, "zero_shot": zero, "after_update": tuned}


def select(history):
    zero_recall = history[0]["metrics"]["zero_shot"]["overall"]["recall@10"]
    best, lora, stale = 0, 0, 0
    best_recall = zero_recall
    highest_lora = -1
    for row in history[1:]:
        current = row["metrics"]["lora"]["overall"]["recall@10"]
        if current > highest_lora:
            lora, highest_lora = row["epoch"], current
        if current > best_recall:
            best, best_recall, stale = row["epoch"], current, 0
        else:
            stale += 1
    return {"best_epoch": best, "best_dev_recall@10": best_recall, "lora_comparison_epoch": lora,
            "lora_comparison_dev_recall@10": highest_lora if lora else None,
            "epochs_without_improvement": stale}


def run_training(config, model, tokenizer, records, dev, candidates):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    config_path = RUN_DIR / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Run configuration changed. Use the original configuration to resume.")
    else:
        write_json(config_path, config)
    lexical = {}
    zero_path = RUN_DIR / "zero_shot_dev.json"
    if not zero_path.exists():
        print("Evaluating zero-shot DEV", flush=True)
        write_json(zero_path, {"epoch": 0, "metrics": evaluate(candidates, dev, model, tokenizer, "zero_shot", lexical)})
    history = [json.loads(zero_path.read_text())]
    saved = sorted(CHECKPOINT_DIR.glob("epoch-*"))
    history.extend(json.loads((path / "dev.json").read_text()) for path in saved)
    if [row["epoch"] for row in history] != list(range(len(history))):
        raise ValueError("Missing or invalid epoch checkpoints")
    selection = select(history)
    write_json(RUN_DIR / "dev_history.json", history)
    print(json.dumps({"epochs_completed": len(saved), **selection}), flush=True)
    if len(saved) < EPOCHS and selection["epochs_without_improvement"] < PATIENCE:
        model, parameters = trainable_encoder(model, saved[-1] if saved else None)
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = EPOCHS * math.ceil(len(records) / ACCUMULATION)
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        if saved:
            state = torch.load(saved[-1] / "resume.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["cpu_rng"])
            torch.mps.set_rng_state(state["mps_rng"])
        for epoch in range(len(history), EPOCHS + 1):
            loss = train_epoch(model, tokenizer, records, optimizer, scheduler, parameters, epoch)
            print(f"Evaluating LoRA DEV after epoch {epoch}", flush=True)
            metrics = evaluate(candidates, dev, model, tokenizer, "lora", lexical)
            save_epoch(CHECKPOINT_DIR, epoch, model, optimizer, scheduler, metrics, loss)
            history.append({"epoch": epoch, "mean_loss": loss, "metrics": metrics})
            write_json(RUN_DIR / "dev_history.json", history)
            selection = select(history)
            print(json.dumps(selection), flush=True)
            if selection["epochs_without_improvement"] >= PATIENCE:
                break
    write_json(RUN_DIR / "selection.json", {"epochs_completed": len(history) - 1, **selection})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Check loading, one update and adapter round-trip")
    args = parser.parse_args()
    hashes = verify_inputs()
    config = experiment_config(hashes)
    candidates = load_candidates()
    dev = load_queries("dev", candidates)
    with (FROZEN_DIR / "bge_m3_training_records_clean.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != 1086 or any(len(r["negatives"]) != 3 for r in records):
        raise ValueError("Unexpected frozen training records")
    RUN_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    tokenizer, model = load_encoder()
    if args.smoke:
        result = smoke_test(model, tokenizer, records, dev, candidates)
        write_json(RUN_DIR / "smoke.json", result)
        print(json.dumps(result), flush=True)
    else:
        with (RUN_DIR / "training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_training(config, model, tokenizer, records, dev, candidates)


if __name__ == "__main__":
    main()
