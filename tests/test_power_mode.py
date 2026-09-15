"""J-VIS Power Mode tests — 'change my power' toggles red mode persistently."""

from __future__ import annotations

import os

import pytest

from power_mode import get_power_mode, set_power_mode


@pytest.fixture(autouse=True)
def isolated_power_state(tmp_path):
    """Point power mode at a per-test state file and clean it up."""
    state_path = str(tmp_path / "power_mode.json")
    old = os.environ.get("POWER_MODE_PATH")
    os.environ["POWER_MODE_PATH"] = state_path
    yield state_path
    if old is None:
        os.environ.pop("POWER_MODE_PATH", None)
    else:
        os.environ["POWER_MODE_PATH"] = old


class TestPowerMode:
    def test_defaults_to_normal(self, isolated_power_state):
        assert get_power_mode() == "normal"

    def test_change_my_power_sets_red(self, isolated_power_state):
        mode = set_power_mode("red")
        assert mode == "red"
        assert get_power_mode() == "red"

    def test_restore_to_normal(self, isolated_power_state):
        set_power_mode("red")
        assert set_power_mode("normal") == "normal"
        assert get_power_mode() == "normal"

    def test_persists_across_reads(self, isolated_power_state):
        set_power_mode("red")
        state_path = isolated_power_state
        os.environ.pop("POWER_MODE_PATH", None)
        os.environ["POWER_MODE_PATH"] = state_path
        assert get_power_mode() == "red"

    def test_invalid_mode_rejected(self, isolated_power_state):
        with pytest.raises(ValueError):
            set_power_mode("purple")
        assert get_power_mode() == "normal"

    def test_mode_case_insensitive(self, isolated_power_state):
        assert set_power_mode("RED") == "red"
        assert get_power_mode() == "red"