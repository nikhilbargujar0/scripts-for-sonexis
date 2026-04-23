# Sonexis Conversational Audio Dataset Pipeline

Offline Python pipeline for turning messy two-speaker conversations into
training-ready audio datasets with rich annotations, manifests, validation, and
interaction metrics.

Primary CLI:

```bash
python main.py \
  --input ./conversation_0001 \
  --output ./dataset \
  --input_type separate \
  --output_mode both \
  --offline_mode true
```

Model loading is local by default. Populate `./models` first:

```bash
python download_models.py --model-dir ./models --whisper-size small
```

See [OFFLINE_SETUP.md](OFFLINE_SETUP.md) for local model setup.
