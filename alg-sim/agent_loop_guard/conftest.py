"""Test-harness setup: make `open_webui` importable.

The pipe runs inside Open WebUI, where `open_webui` is an installed package.
In this checkout it is not available, so the monkey-patch tests
(test_reasoning_replay_ab.py::test_patch_*) register a FAKE
`open_webui.utils.middleware` module in sys.modules — but the leaf module
alone is not enough: `from open_webui.utils import middleware` also needs the
`open_webui` and `open_webui.utils` packages to import.

This conftest puts a minimal fake package on sys.path (before the test
modules load). The fake's own `middleware` module is never imported here —
the tests replace it wholesale via sys.modules.
"""

import sys
from pathlib import Path

FAKE_OWUI = Path(__file__).resolve().parents[1] / "fake_owui"
if str(FAKE_OWUI) not in sys.path:
    sys.path.insert(0, str(FAKE_OWUI))
