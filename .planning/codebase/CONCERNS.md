# Concerns

- CLI boolean parsing accepts unknown values as false, so typos like `--offline_mode treu` silently change behavior.
- `PipelineConfig.output_format` supports `json`, `jsonl`, and `parquet`, but CLI does not currently expose it.
- CLI premium config loading accepts YAML via optional PyYAML, though PyYAML is not listed in dependencies; JSON is the safer documented default.
- Test suite uses `unittest`; `pytest` is not installed in the current system Python environment.
- Large model-dependent paths are hard to exercise in lightweight tests, so CLI/config behavior benefits from direct unit tests.

