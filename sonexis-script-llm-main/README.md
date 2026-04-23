# Sonexis Conversational Audio Dataset Pipeline

Offline Python pipeline for turning messy two-speaker conversations into
training-ready audio datasets with rich annotations, manifests, validation, and
interaction metrics.

Primary CLI:

```bash
python main.py --input ./data --output ./dataset --mode both
```

Model loading is local by default. Populate `./models` first:

```bash
python download_models.py --model-dir ./models --whisper-size small
```

See [pipeline/README.md](pipeline/README.md) and [OFFLINE_SETUP.md](OFFLINE_SETUP.md).
