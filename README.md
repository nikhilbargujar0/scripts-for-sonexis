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

The pipeline does not claim automatic 99% ASR accuracy. Premium workflow
targets 99%+ reviewed delivery accuracy: multi-engine ASR plus consensus, then
human QA and finalisation before client delivery. ASR-vs-review WER is reported
separately from reviewed delivery confidence. Estimated accuracy and verified
accuracy are stored separately.

Workflow A: generate a premium review package.

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

Workflow B: human reviewer edits `final_reviewed_transcript.json`, filling
`reviewed_text` and any `unresolved_issue_types`. `review_reasons` are
informational prompts from ASR/QA and do not mean an issue remains. Use
`unresolved_issue_types` only for problems still unresolved after review.
`issue_types` is deprecated and is not emitted in new review templates.

Use internal speaker IDs `SPEAKER_00` and `SPEAKER_01` when possible.
Human-friendly aliases `speaker_1`/`spk1` and `speaker_2`/`spk2` are accepted.
Avoid `speaker_01`; it is ambiguous unless explicitly configured in
`speaker_map`.

Workflow C: finalise reviewed transcript.

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
