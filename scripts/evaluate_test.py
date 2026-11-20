"""Evaluate all five baselines on locked TEST. Run with python -m scripts.evaluate_test."""

import fcntl
import json

from peft import PeftModel

from scripts.train_bge_m3_lora import (
    CHECKPOINT_DIR, RUN_DIR, evaluate, experiment_config, load_candidates,
    load_encoder, load_queries, verify_inputs, write_json,
)


def main():
    result = RUN_DIR / "test_metrics.json"
    with (RUN_DIR / "test.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result.exists():
            print(result.read_text())
            return
        selection = json.loads((RUN_DIR / "selection.json").read_text())
        config = json.loads((RUN_DIR / "config.json").read_text())
        if config != experiment_config(verify_inputs()):
            raise ValueError("Configuration differs from the completed DEV run")
        epoch = selection["lora_comparison_epoch"]
        if not 1 <= epoch <= selection["epochs_completed"]:
            raise ValueError("DEV selection does not identify a trained LoRA comparison checkpoint")
        checkpoint = CHECKPOINT_DIR / f"epoch-{epoch:02d}"
        candidates = load_candidates()
        tokenizer, encoder = load_encoder()
        # Validate the selected adapter before opening TEST queries.
        tuned = PeftModel.from_pretrained(encoder, checkpoint, is_trainable=False)
        tuned.to("mps")
        queries = load_queries("test", candidates)
        lexical = {}
        with tuned.disable_adapter():
            metrics = evaluate(candidates, queries, tuned, tokenizer, "zero_shot", lexical)
        tuned_metrics = evaluate(candidates, queries, tuned, tokenizer, "lora", lexical)
        metrics.update({name: scores for name, scores in tuned_metrics.items() if name != "bm25"})
        write_json(result, {"selection": selection, "configuration": config, "metrics": metrics})
        print(result.read_text())


if __name__ == "__main__":
    main()
