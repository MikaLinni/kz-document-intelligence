# Model and checkpoint

- Base encoder — `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`. [Pinned model card](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)
- Selected adapter — LoRA **epoch 2**, trained on `query` and `value` projections using Apple M4 and PyTorch MPS.
- Selection — **DEV Recall@10 0.9918**. Training stopped after epoch 4 with patience 2. [Recorded selection](../results/baseline/selection.json)

| TEST method using epoch 2 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BGE-M3 LoRA | 0.9216 | 0.9590 | 0.8111 | 0.8474 |
| BM25 + LoRA RRF | 0.9403 | 0.9627 | 0.8624 | 0.8873 |

## Local setup

Base weights and adapter checkpoints are kept outside Git. After activating the environment, use the Hub CLI to populate the local cache with the pinned base revision for offline demo use.

```bash
hf download BAAI/bge-m3 config.json pytorch_model.bin tokenizer.json tokenizer_config.json \
  special_tokens_map.json sentencepiece.bpe.model \
  --revision 5617a9f61b028005a4858fdac845db406aefb181
```

Put the privately available `adapter_config.json` and `adapter_model.safetensors` in `models/baseline_v1_lora/epoch-02/`. Add the [exact local corpus](../data/README.md#local-layout) before starting the demo. The demo loads weights from cache and requires MPS for dense search.

To reproduce the adapter with the [pinned configuration](../results/baseline/config.json), use the exact frozen inputs and run from the project root.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.train_bge_m3_lora
```

A local run writes checkpoint directories under `models/baseline_v1_lora/` and logs under `results/baseline_v1_lora/`. The GitHub-facing archived results are in [results/baseline](../results/baseline/).
