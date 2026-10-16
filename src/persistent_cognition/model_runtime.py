"""Resource admission for routed local/remote workers; never borrow wrong-model credit."""
from persistent_cognition.runtime_settings import job_settings
from persistent_cognition.ollama_runtime import OllamaRuntimeState


class RoutedRuntimeProbe:
    def capture(self):
        return OllamaRuntimeState(model="routed-models", probe_ok=False, resident=False,
                                  error="App model/context plans use cold estimates without unverified residency credit")


def configured_runtime_probe():
    settings = job_settings()
    if settings is None:
        return None
    return RoutedRuntimeProbe()
