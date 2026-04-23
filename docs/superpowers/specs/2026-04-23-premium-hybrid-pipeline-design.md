# Premium Hybrid Pipeline Design

Date: 2026-04-23
Repo: `nikhilbargujar0/scripts-for-sonexis`
Scope: First production-usable premium hybrid upgrade for the existing offline conversational audio dataset pipeline

## Objective

Upgrade the current offline conversational audio dataset pipeline into a hybrid premium pipeline that:

- preserves the existing offline local-first path
- supports `offline_standard` and `premium_accuracy` modes
- uses paid APIs only when explicitly enabled and only when routing decides they are needed
- improves transcript and timestamp quality through multi-engine ASR, consensus selection, and timestamp refinement
- preserves code-switching across Hindi, English, Hinglish, Marwadi, and Punjabi
- attaches human review metadata to every transcript output
- generates multiple product-specific dataset outputs from one collection flow

This design aims toward high target thresholds such as 98% word and timestamp accuracy, but it must not claim achieved performance unless benchmarked. The pipeline stores target thresholds, measured or estimated quality placeholders, and review state without fabricating outcomes.

## Business Constraints

- `offline_standard` must remain fully local
- premium paid API usage must never happen silently
- all premium provider usage must be controlled through config
- local fallback must still work when premium mode is enabled but paid APIs are disabled or fail
- every output transcript must be human-reviewable
- outdoor or noisy audio must influence routing, QA expectations, TTS suitability, and review priority
- the system should support future language expansion without hardcoding provider behavior across the repo

## Design Summary

The premium pipeline is an extension layer beside the current processor-based architecture, not a rewrite of the core local pipeline.

Existing ownership remains unchanged for:

- input ingestion
- preprocessing
- speaker-pair alignment
- VAD
- diarisation
- mono mixing
- downstream metadata extraction

The premium extension owns:

- multi-engine ASR routing
- selective paid API escalation
- transcript candidate normalization
- transcript consensus selection
- timestamp refinement
- human review metadata
- benchmark-ready premium quality metadata
- purpose-specific product exports

The mono and pair processors decide whether to call the premium layer based on `pipeline_mode`. Both paths still converge on one canonical session record built through the existing output formatter.

## Pipeline Modes

Supported modes:

- `offline_standard`
- `premium_accuracy`

Rules:

- `offline_standard` always stays local
- `premium_accuracy` may use paid APIs only if `premium.enabled=true` and `premium.allow_paid_apis=true`
- premium mode still runs the local engine first
- paid engines are called only when routing says they are needed
- missing premium credentials must fail clearly if a requested provider is selected
- a premium-mode run with paid APIs disabled or unavailable still produces a complete local-fallback record

## Module Layout

### New premium package

- `pipeline/premium/types.py`
  Shared dataclasses and JSON-safe serializers for transcript candidates, consensus results, alignment refinement results, routing reports, and premium processing reports.
- `pipeline/premium/asr_router.py`
  Runs local-first ASR, evaluates difficulty, selectively invokes paid adapters, and returns normalized transcript candidates plus routing metadata.
- `pipeline/premium/consensus.py`
  Scores transcript candidates, picks either the best single candidate or a merged consensus transcript, and emits review signals.
- `pipeline/premium/alignment_router.py`
  Chooses the timestamp source and optionally refines timestamps with local forced alignment or trusted vendor timestamps.
- `pipeline/premium/review.py`
  Builds the required human review block and review-priority metadata.
- `pipeline/premium/quality.py`
  Builds target thresholds, benchmark-ready metrics placeholders, code-switch metadata, and premium processing provenance.

### Adapter package

- `pipeline/premium/adapters/base.py`
  Shared adapter interface, provider configuration helpers, and output normalization helpers.
- `pipeline/premium/adapters/whisper_local.py`
  Wraps the current faster-whisper local path into the premium candidate contract.
- `pipeline/premium/adapters/deepgram_api.py`
  Official Deepgram SDK integration with normalized output.
- `pipeline/premium/adapters/google_stt_v2_api.py`
  Official Google Speech-to-Text v2 client integration with normalized output.
- `pipeline/premium/adapters/azure_speech_api.py`
  Scaffold only for this phase: config and env validation, normalized response contract, and safe not-implemented behavior.

### Product exporters

- `pipeline/products/stt_product.py`
- `pipeline/products/tts_export_product.py`
- `pipeline/products/diarisation_product.py`
- `pipeline/products/evaluation_gold_product.py`
- `pipeline/products/__init__.py`

### Routing helper

- `pipeline/utils/premium_routing.py`
  Difficulty scoring and `should_escalate_to_paid_asr(context)` logic.

## Integration Points

### `pipeline/config.py`

Add premium-aware config fields while preserving current defaults:

- `pipeline_mode`
- `export_products`
- `require_human_review`
- `premium_config`
- premium provider enable flags
- premium credentials env names
- alignment preferences
- routing thresholds
- supported premium languages

Defaults should be conservative and local-first.

### `pipeline/processors/downstream.py`

Refactor the current helper so it can accept a prebuilt transcript instead of always running local ASR internally.

New behavior:

- offline mode passes no override and uses the current local transcribe flow
- premium mode passes the selected or merged transcript from the premium layer
- all downstream language, interaction, speaker metadata, and formatting logic remains shared

### `pipeline/processors/pair_processor.py` and `pipeline/processors/mono_processor.py`

Keep existing ingestion and audio preparation flow. Add a premium branch after preprocessing, VAD, diarisation, and mono mix are ready:

- build routing context
- run premium ASR if configured
- choose and refine final transcript
- continue through shared downstream metadata extraction
- attach premium metadata and generate product exports before write-out

### `pipeline/output_formatter.py`

Extend the canonical record builder to support the premium fields while keeping offline records valid.

### `schema/dataset_schema.json`

Add premium-aware fields as optional or conditionally required by mode. Offline records must still validate without premium-only fields.

## Premium Control Flow

For each session:

1. The existing processor performs preprocessing and current audio preparation steps.
2. The processor computes a routing context from:
   - input mode
   - recording condition
   - noise level
   - overlap amount
   - speaker count
   - local ASR quality if available
   - language and code-switch difficulty signals
3. The premium ASR router always runs the local `whisper_local` adapter first.
4. If `pipeline_mode=offline_standard`, the local candidate is selected and the flow stays fully local.
5. If `pipeline_mode=premium_accuracy`, the router evaluates whether to escalate.
6. If escalation is warranted and paid APIs are allowed, the router invokes enabled providers in order:
   - `deepgram`
   - `google_stt_v2`
7. The router normalizes all candidates into a shared candidate contract.
8. Consensus scoring compares candidates and selects:
   - `best_single_engine`, or
   - `merged_consensus`
9. The alignment router picks the best timestamp source:
   - trusted vendor word timestamps when strong
   - `whisperx` local refinement when configured and available
   - local fallback timestamps otherwise
10. The downstream metadata path runs on the final selected transcript.
11. Premium review, QA, and provenance metadata are attached.
12. Product exporters derive additional product files from the same canonical record.

## Difficulty Routing

The premium router should not call every paid engine by default. Escalation should be selective and evidence-based.

Routing factors:

- outdoor audio
- noisy audio
- code-switch density
- strong Romanized Indic signals
- low local transcript confidence
- low local mean segment quality
- sparse or unstable word timestamps
- disagreement between token language cues and local language classification
- high review-priority conditions

Routing rule:

- if `pipeline_mode=offline_standard`, never escalate
- if `premium.allow_paid_apis=false`, never escalate
- if local quality appears sufficient, keep local-only
- if the difficulty score crosses threshold, call one or more paid providers in the configured order

The routing helper returns:

- difficulty score
- reasons
- whether escalation happened
- engines attempted
- engines skipped

## Transcript Candidate Contract

Each adapter must normalize into one internal candidate shape. The rest of the repo should only consume this shared contract.

Candidate fields:

- `engine`
- `provider`
- `paid_api`
- `transcript`
- `confidence`
- `avg_word_confidence`
- `language_hint`
- `detected_languages`
- `code_switch_signals`
- `timing_source`
- `timestamp_confidence`
- `warnings`
- `adapter_metadata`

Normalized example:

```json
{
  "engine": "deepgram",
  "provider": "deepgram",
  "paid_api": true,
  "transcript": {
    "raw": "...",
    "language": "hinglish",
    "segments": [],
    "words": []
  },
  "confidence": 0.91,
  "avg_word_confidence": 0.89,
  "language_hint": "hinglish",
  "detected_languages": ["Hindi", "English"],
  "code_switch_signals": {
    "detected": true,
    "dominant_languages": ["Hindi", "English"],
    "switch_count": 7,
    "switch_patterns": ["hi->en", "en->hi"],
    "romanized_indic_present": true
  },
  "timing_source": "vendor_word_timestamps",
  "timestamp_confidence": 0.9,
  "warnings": [],
  "adapter_metadata": {
    "request_id": "...",
    "model": "...",
    "sdk": "official"
  }
}
```

Implementation rule:

- in memory, the transcript portion should remain compatible with the repo’s existing `Transcript`, `TranscriptSegment`, and `Word` structures so the current downstream metadata path can be reused cleanly
- in persisted JSON, candidates are serialized to a stable schema-safe structure under `transcript_candidates`

## Consensus Design

Consensus should remain practical. The first pass does not need token-graph decoding or complicated lattice merging.

Scoring factors:

- token agreement across engines
- segment timing consistency
- word timestamp availability
- engine confidence
- candidate timestamp confidence
- code-switch preservation
- language-switch agreement
- presence of Indic and Romanized Indic tokens
- audio-condition weighting

Outputs:

- selected engine
- compared engines
- consensus score
- strategy
- review recommendation
- candidate-level rationale

Supported strategies:

- `best_single_engine`
- `merged_consensus`

The merged path should stay conservative and operate only where alignment between candidates is good enough to merge safely.

## Timestamp Refinement Design

Timestamp refinement produces the final timestamped transcript plus confidence metadata.

Priority order:

1. trusted vendor word timestamps from the selected premium candidate
2. local `whisperx` refinement when enabled and available
3. existing local timestamps from the winning transcript candidate

Output metadata:

- `timestamp_method`
- `timestamp_confidence`
- `word_timestamps_available`
- `segment_timestamps_available`
- `refinement_applied`
- `alignment_notes`

The system should not overclaim timestamp quality. If timestamps are only partially available, the metadata should say so explicitly.

## Human Review Workflow

Every transcript output must include a required review block:

```json
"human_review": {
  "required": true,
  "status": "pending",
  "review_stage": "transcript_review",
  "reviewer_id": null,
  "notes": null
}
```

Supported statuses:

- `pending`
- `in_review`
- `approved`
- `corrected`
- `rejected`

Review metadata should be easy to update later without forcing a schema redesign.

Additional review priority metadata may include:

- priority score
- review reasons
- premium disagreement indicators

Review priority should rise for:

- outdoor audio
- low consensus score
- weak timestamp confidence
- code-switch detection
- mismatch between local and paid candidates

## Quality Targets and Metrics

The system stores premium targets without pretending they were achieved.

Required targets block:

```json
"quality_targets": {
  "word_accuracy_target": 0.98,
  "timestamp_accuracy_target": 0.98,
  "code_switch_accuracy_target": 0.98,
  "human_review_required": true
}
```

Required metrics block:

```json
"quality_metrics": {
  "estimated_word_accuracy": null,
  "estimated_timestamp_accuracy": null,
  "estimated_code_switch_accuracy": null,
  "benchmark_evaluated": false,
  "human_review_completed": false
}
```

Rules:

- premium-facing schema should use `word_accuracy_target` and `estimated_word_accuracy`
- existing local fields such as `transcript_accuracy_target` may be mapped during transition, but the canonical premium record should converge on the premium names
- no fabricated achieved metrics
- null until a benchmark or evaluation workflow populates them
- benchmark readiness should be explicit in the record

## Code-Switch Support

Premium mode should explicitly preserve mixed-language content and treat code-switching as a quality and review dimension, not just a descriptive label.

Required metadata block:

```json
"code_switch": {
  "detected": true,
  "dominant_languages": ["Hindi", "English"],
  "switch_count": 7,
  "switch_patterns": ["hi->en", "en->hi"],
  "review_required": true
}
```

Rules:

- mixed-language tokens must be preserved
- do not force English-only normalization
- retain Romanized Indic forms where available
- current supported languages: Hindi, English, Hinglish, Marwadi, Punjabi
- future expansion must be config-driven and provider-agnostic

## Audio Condition Handling

The current business requirement uses `indoor` and `outdoor` as supported recording conditions. The existing metadata extraction already infers richer environmental signals. The premium layer should normalize these into conservative premium routing labels:

- `indoor`
- `outdoor`
- `unknown`

These labels influence:

- premium routing
- review priority
- TTS suitability
- quality expectations

Examples:

- outdoor noisy audio increases escalation likelihood
- outdoor code-switching audio increases manual review priority
- outdoor audio is less likely to pass TTS subset filtering

## Product Generation

The system should produce multiple product-specific outputs from one canonical session record.

Supported products:

- `stt`
- `tts_export`
- `diarisation`
- `evaluation_gold`
- `speaker_separated`
- `mono_mixed`

### STT product

Includes:

- final transcript
- timestamps
- speaker turns
- language segments
- code-switch metadata
- speaker-separated or mono context

### TTS export product

Conservative subset only.

Eligibility factors:

- single speaker segment
- no overlap
- stable timing
- low noise
- stable loudness
- clean transcript alignment
- acceptable review state

Required metadata:

```json
"tts_suitability": {
  "eligible": false,
  "reasons": ["overlap_detected", "conversation_style_audio"],
  "confidence": 0.88
}
```

### Diarisation product

Includes:

- speaker turns
- overlap markers
- timing quality metadata
- input alignment provenance

### Evaluation or gold product

Includes:

- all transcript candidates
- selected final transcript
- consensus report
- timestamp refinement metadata
- review metadata
- quality metadata
- validation report

### Storage model

One canonical session record remains the source of truth.

Product-specific files are written under:

- `products/stt/<session>.json`
- `products/tts_export/<session>.json`
- `products/diarisation/<session>.json`
- `products/evaluation_gold/<session>.json`

The canonical record stores artifact references for downstream discovery.

## Premium Processing Provenance

Each record should describe how it was produced:

```json
"premium_processing": {
  "pipeline_mode": "premium_accuracy",
  "paid_api_used": true,
  "engines_used": ["whisper_local", "deepgram"],
  "consensus_applied": true,
  "timestamp_refinement_applied": true,
  "human_review_required": true
}
```

This allows clients to separate local-only outputs from premium hybrid outputs and audit paid API usage clearly.

## Provider Configuration and Secrets

Premium provider configuration must live in config and read secrets from environment variables only.

Representative shape:

```yaml
premium:
  enabled: true
  allow_paid_apis: true
  paid_budget_mode: smart
  preferred_asr_engines:
    - deepgram
    - google_stt_v2
    - whisper_local
  preferred_alignment_engines:
    - whisperx
    - vendor_word_timestamps
  require_human_review: true
  asr_engines:
    whisper_local:
      enabled: true
    deepgram:
      enabled: true
      api_key_env: DEEPGRAM_API_KEY
    google_stt_v2:
      enabled: true
      credentials_env: GOOGLE_APPLICATION_CREDENTIALS
    azure_speech:
      enabled: false
      api_key_env: AZURE_SPEECH_KEY
      region_env: AZURE_SPEECH_REGION
  alignment:
    whisperx_enabled: true
    vendor_word_timestamps_enabled: true
```

Rules:

- no hardcoded keys
- fail clearly when a selected provider lacks credentials
- log which engines were used
- keep provider assumptions inside adapters

## CLI Changes

Extend the existing CLI conservatively with optional flags:

- `--pipeline_mode`
- `--allow_paid_apis`
- `--premium_config`
- `--require_human_review`
- `--export_products`

Representative usage:

```bash
python main.py \
  --input ./conversation_0001 \
  --output ./dataset \
  --input_type separate \
  --output_mode both \
  --pipeline_mode premium_accuracy \
  --allow_paid_apis true \
  --export_products stt,tts_export,evaluation_gold
```

Defaults should preserve current behavior.

## Schema Changes

Extend `schema/dataset_schema.json` for:

- `human_review`
- `quality_targets`
- `quality_metrics`
- `code_switch`
- `transcript_candidates`
- `timestamp_method`
- `timestamp_refinement`
- `tts_suitability`
- `dataset_products`
- `premium_processing`

Schema principles:

- strict enough for consistency
- premium-only fields optional in offline mode
- review and quality-target fields required where business policy requires them

## Testing Strategy

Add tests for:

- premium routing
  - local-only path
  - escalation path
  - missing credentials path
- consensus
  - best candidate selected
  - merged consensus path
- timestamp refinement
  - word timestamps available
  - refinement metadata attached
- human review
  - every premium output includes the review block
- product generation
  - STT product written
  - TTS export filters conservatively
  - evaluation product includes transcript candidates
- schema
  - offline and premium records both validate

Use fixtures and mocks for provider clients so unit tests do not require real API credentials.

## Out of Scope for This Phase

- full Azure Speech execution path
- claimed benchmark achievement
- human review UI or reviewer tooling
- advanced consensus lattices or token-graph decoding
- broad language expansion beyond the currently required set

## Risks and Mitigations

### Risk: provider-specific logic leaks into the core pipeline

Mitigation:

- adapters expose one normalized candidate contract
- processors and formatter consume only normalized premium objects

### Risk: premium mode destabilizes offline mode

Mitigation:

- offline mode keeps the current path
- premium behavior is branch-based and config-gated
- offline validation remains covered by schema and tests

### Risk: timestamp confidence is overstated

Mitigation:

- refinement metadata explicitly reports method and confidence
- no benchmark claims unless measured

### Risk: conversational audio is over-exported into TTS

Mitigation:

- TTS export stays conservative
- overlap, noise, and alignment stability gate eligibility

## Implementation Recommendation

Implement in this order:

1. premium types and config plumbing
2. premium routing helper and local adapter wrapper
3. Deepgram and Google adapters
4. consensus and alignment router
5. processor integration
6. output formatter and schema updates
7. product exporters
8. tests and README updates

This order keeps the premium path incremental and reduces regressions in the current offline system.
