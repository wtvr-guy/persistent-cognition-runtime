# Headless model routing

Model selection, parameter validation, admission, and sequential residency are reusable engine components. Their original GUI wrappers have been removed.

The renamed modules are `runtime_settings` (formerly gui_config), `model_catalog` (gui_models), `model_runtime` (gui_runtime), and `process_lock` (gui_instance). `runtime_job` retains only the guarded chat-job path, including unload-before-measure admission and immutable effective worker settings.

`AppSettings` remains the versioned source contract. Runtime configuration uses `PCR_*` environment names; legacy GUI-shaped suffixes describe the retained settings transport, not an installed GUI. Load a JSON settings snapshot into `PCR_GUI_JOB_CONFIG_FILE` to use it in fresh workers; a plain terminal invocation otherwise uses the environment-driven Ollama path.

A programmatic caller can create a private job directory containing `job.json`:

```json
{"action":"chat","payload":{"text":"Explain this function","conversation_id":"11111111-1111-4111-8111-111111111111","task":"coding","fallback":"default"}}
```

Set the configuration-file environment variable and run `uv run python -m persistent_cognition.runtime_job PATH_TO_JOB_DIRECTORY`. The job records admission, progress, effective settings, and result or failure. It does not activate application-profile machinery, start a GUI, collect sensors, modify OS security, or sync devices.

Create the settings snapshot explicitly; omission is rejected before opening the
database or changing model residency:

```python
import os
from pathlib import Path
from persistent_cognition.api import AppSettings, ModelSelection
from persistent_cognition.operator_state import write_private_policy

settings_path = Path("private-jobs/settings.json").absolute()
settings = AppSettings(selection=ModelSelection(model="qwen3:4b-instruct-2507-q4_K_M"))
write_private_policy(settings_path, settings.model_dump(mode="json"))
os.environ["PCR_GUI_JOB_CONFIG_FILE"] = str(settings_path)
os.environ["DATABASE_URL"] = "postgresql://pcr_app:YOUR_PASSWORD@localhost:5432/pcr"
os.environ["PCR_ARTIFACT_ROOT"] = str(Path("private-history").absolute())
# Write the above job.json into a private job directory, then invoke runtime_job.
```

Use [the supported API](EMBEDDING.md) for direct embedding. Process-scoped settings
must be established once at application startup. [Deployment limits and remote
enrollment](SECURITY_AND_STORAGE.md) apply to headless jobs and their workers.

The default model and optional stage routes must fit actual CPU/RAM admission. The sequential residency controller verifies unload boundaries; a preview of reclaimable memory cannot authorize execution. Remote-provider routes remain explicit and require their source consent/credential checks.

The source namespace and versioned settings type names are compatibility details. No web-server framework or browser automation dependency is installed by this package.

The `persistent_cognition` runtime has no persona or self-model prompt/schema/stage. Structured non-user reactions can be entirely model-free. Semantic triage and natural-language responses retain guarded, stateless model calls only when their application source policy requires them. Trusted application reactions use explicitly injected worker startup, not model-selected import paths; see [Setup](SETUP.md#trusted-application-callbacks).
