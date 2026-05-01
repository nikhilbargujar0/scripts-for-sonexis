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

## Premium Review Workflow

The premium path does not claim automatic 99% ASR accuracy. It creates a
multi-engine ASR and consensus review package, then requires human QA before a
record can be approved for client delivery. Estimated accuracy and verified
accuracy are stored separately.

Generate a premium review package:

```bash
python main.py \
  --input ./raw_dataset/conversation_0001 \
  --output ./dataset_output \
  --input_type speaker_folders \
  --output_mode both \
  --pipeline_mode premium_accuracy \
  --allow_paid_apis true \
  --require_human_review true \
  --word_accuracy_target 0.99 \
  --speaker_accuracy_target 0.99 \
  --timestamp_accuracy_target 0.98 \
  --code_switch_accuracy_target 0.99 \
  --model_dir ./models
```

This writes:

- `review/conversation_0001/human_review_template.json`
- `review/conversation_0001/final_reviewed_transcript.json`
- `review/conversation_0001/qa_report.json`

After a reviewer completes `final_reviewed_transcript.json`, finalise the
record:

```bash
python review_finalize.py \
  --annotation ./dataset_output/annotations/conversation_0001.json \
  --reviewed_transcript ./dataset_output/review/conversation_0001/final_reviewed_transcript.json \
  --output ./dataset_output \
  --reviewer_id reviewer_001 \
  --approve_if_passed true
```

`approved_for_client_delivery` only becomes `true` after finalisation verifies
that validation passed, review is completed, reviewed text exists, and verified
QA metrics meet the configured thresholds. Pending or failed review keeps the
record out of delivery manifests.
