"""Validate the active contract catalog and print its reproducible manifest."""
import json
from persistent_cognition.contract_registry import contract_manifest

if __name__ == "__main__":
    print(json.dumps(contract_manifest(), indent=2, sort_keys=True))
