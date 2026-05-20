
import pytest

from wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_flat_cal_plan import (
    FLAT_FILTERS,
    FlatSimulation,
)

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sim(tmp_path):
    """
    Basic FlatSimulation instance for testing.
    """
    return FlatSimulation(
        output_dir=tmp_path,
        auto_run=False,
        config_file=None,
    )


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_initialization_defaults(sim):

    assert sim.num_exposures == 20
    assert sim.truncate == 20

    assert sim.scas == list(range(1, 19))

    assert sim.filters == FLAT_FILTERS


# ---------------------------------------------------------
# SCA parsing
# ---------------------------------------------------------

def test_parse_scas_ints(sim):

    result = sim._parse_scas([1, 5, 8])

    assert result == [1, 5, 8]


def test_parse_scas_strings(sim):

    result = sim._parse_scas(["WFI01", "WFI05"])

    assert result == [1, 5]


def test_parse_scas_duplicates_removed(sim):

    result = sim._parse_scas([1, 1, "WFI01"])

    assert result == [1]


def test_parse_scas_invalid(sim):

    with pytest.raises(ValueError):
        sim._parse_scas([None])


# ---------------------------------------------------------
# Filter parsing
# ---------------------------------------------------------

def test_parse_all_filters(sim):

    result = sim._parse_filters("ALL")

    assert result == FLAT_FILTERS


def test_parse_single_filter(sim):

    result = sim._parse_filters("F106")

    assert result == ["F106"]


def test_parse_multiple_filters(sim):

    result = sim._parse_filters(["F106", "F158"])

    assert result == ["F106", "F158"]


def test_parse_invalid_filter(sim):

    with pytest.raises(ValueError):
        sim._parse_filters(["BADFILTER"])


# ---------------------------------------------------------
# Filename generation
# ---------------------------------------------------------

def test_make_filename(sim):

    filename = sim._make_filename(
        exp=3,
        sca=7,
        filt="F106",
    )

    expected = (
        sim.output_dir /
        "r0090401001001001001_0003_wfi07_f106_uncal.asdf"
    )

    assert filename == expected


# ---------------------------------------------------------
# Config handling
# ---------------------------------------------------------

def test_get_sca_flat_params_no_config(sim):

    params = sim._get_sca_flat_params(1)

    assert params == {}


# ---------------------------------------------------------
# Mock Romanisim subprocess
# ---------------------------------------------------------

def test_run_romanisim_success(monkeypatch, sim, tmp_path):

    class MockResult:
        returncode = 0
        stderr = ""

    def mock_run(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(
        "subprocess.run",
        mock_run,
    )

    filename = tmp_path / "test.asdf"

    sim._run_romanisim(
        filename=filename,
        current_time=sim.start_time,
        sca=1,
    )


def test_run_romanisim_failure(monkeypatch, sim, tmp_path):

    class MockResult:
        returncode = 1
        stderr = "romanisim failed"

    def mock_run(*args, **kwargs):
        return MockResult()

    monkeypatch.setattr(
        "subprocess.run",
        mock_run,
    )

    filename = tmp_path / "test.asdf"

    with pytest.raises(RuntimeError):
        sim._run_romanisim(
            filename=filename,
            current_time=sim.start_time,
            sca=1,
        )
