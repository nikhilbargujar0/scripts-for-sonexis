# Colab Messy Drive Input

Sonexis accepts messy Drive uploads and normalizes them into canonical speaker-separated input before running ASR.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/nikhilbargujar0/scripts-for-sonexis.git
%cd /content/scripts-for-sonexis
!pip install -r requirements.txt
!apt-get install -y ffmpeg
```

Audit input first:

```bash
!python scripts/main.py \
  --input "/content/drive/MyDrive/sonexis/sample/sonexis_audio" \
  --output "/content/drive/MyDrive/sonexis/output" \
  --input_type auto \
  --normalise_messy_input true \
  --audit_input_only true \
  --colab true
```

Run pipeline:

```bash
!python scripts/main.py \
  --input "/content/drive/MyDrive/sonexis/sample/sonexis_audio" \
  --output "/content/drive/MyDrive/sonexis/output" \
  --input_type auto \
  --normalise_messy_input true \
  --colab true \
  --model_size large-v3 \
  --pipeline_mode premium_accuracy \
  --require_human_review true \
  --word_accuracy_target 0.99 \
  --speaker_accuracy_target 0.99 \
  --timestamp_accuracy_target 0.98 \
  --code_switch_accuracy_target 0.99 \
  --asr_batched true
```

Expected output:

```text
/content/drive/MyDrive/sonexis/output/
├── english/
├── hindi/
├── hinglish/
├── marwadi/
└── punjabi/
```

Each language folder contains its own `annotations/`, `review/`, `transcripts/`, and `manifests/` directories.

The pipeline run creates ASR output and human review packages. The 99%+ claim only applies after mandatory human review and `review_finalize.py` approval. ASR-vs-review WER/CER are diagnostic only and are not the final delivery accuracy claim.
