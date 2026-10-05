# Headless model routing

Model selection, parameter validation, admission, and sequential residency are reusable engine components. Their original GUI wrappers have been removed.

The renamed modules are `runtime_settings` (formerly gui_config), `model_catalog` (gui_models), `model_runtime` (gui_runtime), and `process_lock` (gui_instance). `runtime_job` retains only the guarded chat-job path, including unload-before-measure admission and immutable effective worker settings.

`AppSettings` remains the versioned source contract. Existing configuration environment names are retained for compatibility. Load a JSON settings snapshot into `PROMETHEIST_GUI_JOB_CONFIG_FILE` to use it in fresh workers; a plain terminal invocation otherwise uses the source project's environment-driven Ollama path.

A programmatic caller can create a private job directory containing `job.json`:

```json
{"action":"chat","payload":{"text":"Explain this function","conversation_id":"11111111-1111-4111-8111-111111111111","task":"coding","fallback":"default"}}
```

Set the configuration-file environment variable and run `uv run python -m prometheist.runtime_job PATH_TO_JOB_DIRECTORY`. The job records admission, progress, effective settings, and result or failure. It does not activate a person profile, start a GUI, collect sensors, modify OS security, or sync devices.

The default model and optional stage routes must fit actual CPU/RAM admission. The sequential residency controller verifies unload boundaries; a preview of reclaimable memory cannot authorize execution. Remote-provider routes remain explicit and require their source consent/credential checks.

The source namespace and versioned settings type names are compatibility details. No web-server framework or browser automation dependency is installed by this package.
