import subprocess
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
import yaml
from astropy.time import Time

from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_REF_OPTICAL_ELEMENT_F062,
    WFI_REF_OPTICAL_ELEMENT_F087,
    WFI_REF_OPTICAL_ELEMENT_F106,
    WFI_REF_OPTICAL_ELEMENT_F129,
    WFI_REF_OPTICAL_ELEMENT_F146,
    WFI_REF_OPTICAL_ELEMENT_F158,
    WFI_REF_OPTICAL_ELEMENT_F184,
    WFI_REF_OPTICAL_ELEMENT_F213,
)
from wfi_reference_pipeline.utilities.simulate_reads import simulate_flat_reads

FLAT_FILTERS = [
    WFI_REF_OPTICAL_ELEMENT_F062,
    WFI_REF_OPTICAL_ELEMENT_F087,
    WFI_REF_OPTICAL_ELEMENT_F106,
    WFI_REF_OPTICAL_ELEMENT_F129,
    WFI_REF_OPTICAL_ELEMENT_F146,
    WFI_REF_OPTICAL_ELEMENT_F158,
    WFI_REF_OPTICAL_ELEMENT_F184,
    WFI_REF_OPTICAL_ELEMENT_F213,
]

class FlatSimulation:
    """
    Simulate full flat calibration plan for all detectors

    Example usage:
    from wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_flat_cal_plan import FlatSimulation

    flat = FlatSimulation(
    output_dir="/grp/roman/RFP/DEV/sim_inflight_calplan/romanisim_flats",
    config_file="simulate_flats_config.yml",
    scas=[3], or a list of specific detectors [1,5,8,12], or ALL by default
    num_exposures=5,
    truncate=20,
    flat_rate=1000,
    auto_run=True
    )


    """

    def __init__(
        self,
        output_dir,
        scas="ALL_WFI",
        num_exposures=20,
        truncate=20,
        start_time="2026-10-31T00:00:00",
        program="00904",
        config_file='simulate_flats_config.yml',
        flat_rate=1000,
        filters="ALL",   # <-- NEW
        auto_run=False,   
    ):
        self.output_dir = Path(output_dir)
        self.scas = self._parse_scas(scas)
        self.num_exposures = num_exposures
        self.truncate = truncate
        self.start_time = Time(start_time)
        self.program = program
        self.flat_rate = flat_rate
        self.auto_run = auto_run
        self.filters = self._parse_filters(filters)

        self.config = self._load_config(config_file)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.auto_run:
            self.run()

    # ---------------------------------------------------------
    # Filter parsing
    # ---------------------------------------------------------
    def _parse_filters(self, filters):

        if filters == "ALL":
            return FLAT_FILTERS

        if isinstance(filters, str):
            filters = [filters]

        parsed = []

        valid_filters = set(FLAT_FILTERS)

        for filt in filters:

            filt = filt.upper()

            if filt not in valid_filters:
                raise ValueError(
                    f"Invalid filter {filt}. "
                    f"Valid filters are: {sorted(valid_filters)}"
                )

            parsed.append(filt)

        return parsed

    # ---------------------------------------------------------
    # SCA parsing aka WFI01 to WFI18
    # ---------------------------------------------------------
    def _parse_scas(self, scas):
        if scas == "ALL_WFI":
            return list(range(1, 19))

        parsed = []
        for sca in scas:
            if isinstance(sca, int):
                parsed.append(sca)
            elif isinstance(sca, str):
                parsed.append(int(sca.replace("WFI", "")))
            else:
                raise ValueError(f"Invalid SCA: {sca}")

        return sorted(set(parsed))

    # ---------------------------------------------------------
    # Config handling - override defaults from config file
    # ---------------------------------------------------------
    def _load_config(self, config_file):
        if config_file is None:
            return None

        config_path = Path(__file__).resolve().parent / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_sca_flat_params(self, sca):
        if self.config is None:
            return {}

        params = dict(self.config.get("defaults", {}))
        overrides = self.config.get("sca_overrides", {})

        if sca in overrides:
            params.update(overrides[sca])

        return params

    # ---------------------------------------------------------
    # Filename
    # ---------------------------------------------------------
    def _make_filename(self, exp, sca, filt):
        return self.output_dir / (
            f"r{self.program}01001001001001_"
            f"{exp:04d}_wfi{sca:02d}_"
            f"{filt.lower()}_uncal.asdf"
        )

    # ---------------------------------------------------------
    # Romanisim call
    # ---------------------------------------------------------
    def _run_romanisim(self, filename, current_time, sca):
        command = [
            "romanisim-make-image",
            "--date", current_time.isot,
            "--nobj", "0",
            "--usecrds",
            "--sca", str(sca),
            "--level", "1",
            "--ma_table_number", "9003",
            "--truncate", str(self.truncate),
            str(filename),
        ]

        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    # ---------------------------------------------------------
    # Post process and inject flat cube made from RFP
    # ---------------------------------------------------------
    def _post_process(self, filename, sca, filt):
        params = self._get_sca_flat_params(sca)

        # Extract scaling factor or default to 1.0
        scale = params.pop("l_flat_pseudo_factor", 1.0)


        flat_cube, _ = simulate_flat_reads(
            n_reads=self.truncate,
            flat_rate=self.flat_rate * scale,  # apply L-flat pseudo scaling here
            **params,  # and now other overrides to make detectors like WFI03 noisy
        )

        with asdf.open(filename, mode="rw") as af:
            af.tree["roman"]["meta"]["instrument"]["optical_element"] = filt
            af.tree["roman"]["data"] = flat_cube.astype(np.uint16)
            af.update()


    # ---------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------
    def run(self):

        time_overhead = 10.0 * u.s   # skip 10 seconds between exposures
        filter_gap = 1.0 * u.hour   # skip hour between filters - just for checking

        current_time = self.start_time.copy()

        for filt in self.filters:
            print("\n==============================")
            print(f"Running filter {filt}")
            print("==============================")

            # ALL SCAs start together for this filter
            sca_times = {sca: current_time.copy() for sca in self.scas}

            for exp in range(1, self.num_exposures + 1):

                for sca in self.scas:
                    print(f"\n--- SCA {sca} | EXP {exp} ---")

                    filename = self._make_filename(exp, sca, filt)
                    t = sca_times[sca]

                    try:
                        self._run_romanisim(filename, t, sca)
                        self._post_process(filename, sca, filt)

                        print(f"✔ Created {filename.name}")

                    except Exception as e:
                        print(f"✘ Failed exp {exp}: {e}")

                    sca_times[sca] += (
                        self.truncate * WFI_FRAME_TIME[WFI_MODE_WIM] * u.s
                        + time_overhead
                    )

            # after finishing filter advance clock by hour
            current_time += filter_gap