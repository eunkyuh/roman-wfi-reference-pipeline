# tests/test_dark_simulation.py

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan import (
    BaseDarkSimulation,
    WeeklyDarkSimulation,
)

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "darks"


@pytest.fixture
def basic_dark(output_dir):
    return BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=2,
        scas=[1, "WFI03"],
        ma_table_number=1010,
    )


@pytest.fixture
def config_dict():

    return {
        "defaults": {
            "gain": 2.0,
            "seed": 123,
        },
        "sca_overrides": {
            3: {
                "gain": 5.0,
            }
        },
    }


# ============================================================
# SCA parsing
# ============================================================

def test_parse_all_wfi(output_dir):

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas="ALL_WFI",
        ma_table_number=1010,
    )

    assert sim.scas == list(range(1, 19))


def test_parse_mixed_scas(basic_dark):

    assert basic_dark.scas == [1, 3]


@pytest.mark.parametrize(
    "bad_scas",
    [
        "WFI01",
        [0],
        [19],
        ["BAD"],
        [None],
    ],
)
def test_invalid_scas_raise(output_dir, bad_scas):

    with pytest.raises(ValueError):
        BaseDarkSimulation(
            output_dir=output_dir,
            program="00999",
            start_time="2026-01-01T00:00:00",
            num_exposures=1,
            scas=bad_scas,
            ma_table_number=1010,
        )


# ============================================================
# MA table config
# ============================================================

def test_get_ma_table_config_weekly(basic_dark):

    cfg = basic_dark._get_ma_table_config()

    assert cfg["truncate"] is None
    assert cfg["n_reads"] == 8


def test_invalid_ma_table(output_dir):

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[1],
        ma_table_number=9999,
    )

    with pytest.raises(ValueError):
        sim._get_ma_table_config()


# ============================================================
# Config loading
# ============================================================

def test_load_config_and_overrides(output_dir, config_dict, monkeypatch):

    monkeypatch.setattr(
        BaseDarkSimulation,
        "_load_config",
        lambda self, _: config_dict,
    )

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[3],
        ma_table_number=1010,
        config_file="dummy.yml",
    )

    params = sim._get_sca_dark_params(3)

    assert params["gain"] == 5.0


def test_missing_config_raises(output_dir):

    with pytest.raises(FileNotFoundError):
        BaseDarkSimulation(
            output_dir=output_dir,
            program="00999",
            start_time="2026-01-01T00:00:00",
            num_exposures=1,
            scas=[1],
            ma_table_number=1010,
            config_file="does_not_exist.yml",
        )


# ============================================================
# Filename generation
# ============================================================

def test_make_filename_uncal(basic_dark):

    filename = basic_dark._make_filename(exp=5, sca=3)

    expected = (
        "r0099901001001001001_0005_wfi03_f213_uncal.asdf"
    )

    assert filename.name == expected


def test_make_filename_cal(output_dir):

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[1],
        ma_table_number=1010,
        level=2,
    )

    filename = sim._make_filename(exp=1, sca=1)

    assert filename.name.endswith("_cal.asdf")


# ============================================================
# Romanisim command construction
# ============================================================

def test_run_romanisim_calls_subprocess(monkeypatch, basic_dark):

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    monkeypatch.setattr(
        "subprocess.run",
        mock_run,
    )

    basic_dark._run_romanisim(
        filename=Path("test.asdf"),
        current_time=basic_dark.start_time,
        sca=1,
    )

    assert mock_run.called

    cmd = mock_run.call_args[0][0]

    assert "romanisim-make-image" in cmd
    assert "--sca" in cmd
    assert "1" in cmd


def test_run_romanisim_failure(monkeypatch, basic_dark):

    mock_run = MagicMock()
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "failure"

    monkeypatch.setattr(
        "subprocess.run",
        mock_run,
    )

    with pytest.raises(RuntimeError):
        basic_dark._run_romanisim(
            filename=Path("bad.asdf"),
            current_time=basic_dark.start_time,
            sca=1,
        )


# ============================================================
# run()
# ============================================================

def test_run_calls_internal_methods(monkeypatch, basic_dark):

    run_mock = MagicMock()
    post_mock = MagicMock()

    monkeypatch.setattr(
        basic_dark,
        "_run_romanisim",
        run_mock,
    )

    monkeypatch.setattr(
        basic_dark,
        "_post_process",
        post_mock,
    )

    basic_dark.run()

    expected_calls = (
        basic_dark.num_exposures * len(basic_dark.scas)
    )

    assert run_mock.call_count == expected_calls
    assert post_mock.call_count == expected_calls


# ============================================================
# WeeklyDarkSimulation subclass
# ============================================================

def test_weekly_dark_defaults(output_dir):

    sim = WeeklyDarkSimulation(
        output_dir=output_dir,
        num_exposures=5,
        scas=[1],
    )

    assert sim.program == "00901"
    assert sim.ma_table_number == 1010
    assert sim.num_exposures == 5


# ============================================================
# _post_process smoke test
# ============================================================

def test_post_process_smoke(monkeypatch, tmp_path, basic_dark):

    fake_cube = np.ones((8, 4, 4), dtype=np.uint16)

    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.simulate_dark_reads",
        lambda n_reads, **kwargs: (fake_cube, None),
    )

    mock_asdf = MagicMock()

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_asdf

    monkeypatch.setattr(
        "asdf.open",
        lambda *args, **kwargs: mock_ctx,
    )

    mock_asdf.tree = {
        "roman": {
            "meta": {
                "instrument": {}
            }
        }
    }

    basic_dark._post_process(
        filename=tmp_path / "fake.asdf",
        sca=1,
    )

    assert (
        mock_asdf.tree["roman"]["meta"]["instrument"]["optical_element"]
        == "DARK"
    )

    assert "data" in mock_asdf.tree["roman"]