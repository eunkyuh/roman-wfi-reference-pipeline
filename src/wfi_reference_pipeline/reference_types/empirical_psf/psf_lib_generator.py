from copy import deepcopy

import numpy as np
import roman_datamodels as rdm
import stsynphot as stsyn
from astropy import units as u
from crds import getreferences
from scipy import ndimage
from stpsf import roman


# ============================================================
# Utilities help functions
# ============================================================
def ensure_list(value):
    """
    Convert scalar input to list.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    return [value]


def build_source_dictionary(
    spectral_types,
    temperatures,
    logg_values,
):
    """
    Construct source spectrum dictionary.
    """

    spectral_types = ensure_list(spectral_types)
    temperatures = ensure_list(temperatures)
    logg_values = ensure_list(logg_values)

    if not (
        len(spectral_types)
        == len(temperatures)
        == len(logg_values)
    ):
        raise ValueError(
            "spectral_types, temperatures, and logg_values "
            "must have identical lengths."
        )

    source_dict = {}

    for stype, teff, logg in zip(
        spectral_types,
        temperatures,
        logg_values,
    ):

        source_dict[stype] = {
            "teff": teff,
            "logg": logg,
            "spectrum": stsyn.grid_to_spec(
                "phoenix",
                teff,
                0.0,
                logg,
            ),
        }
    return source_dict

# ============================================================
# IPC / PRF
# ============================================================
def apply_ipc(
    psf_array,
    oversample=4,
    ipc_kernel=None,
):

    prf = np.ones((oversample, oversample))
    result_prf = ndimage.convolve(
        psf_array,
        prf,
        mode="constant",
        cval=0,
    )

    if ipc_kernel is None:
        return result_prf.astype(np.float32)
    ipc_sparse = np.zeros(
        (oversample * 2 + 1, oversample * 2 + 1)
    )
    ipc_sparse[::oversample, ::oversample] = ipc_kernel
    result = ndimage.convolve(
        result_prf,
        ipc_sparse,
        mode="constant",
        cval=0,
    )

    return result.astype(np.float32)

# ============================================================
# Core PSF Generator
# ============================================================
def generate_wfi_psf_library(
    detector,
    optical_element,
    pixel_x,
    pixel_y,
    spectral_types=("A0V", "G2V", "M5V"),
    temperatures=(9550, 5778, 2900),
    logg_values=(3.95, 4.44, 5.0),
    defocus_values=(0, 1, 2),
    oversample=4,
    fov_pixels=361,
    nlambda=10,
    include_extended_psf=True,
    extended_fov_pixels=3641,
    include_noipc=True,
    use_crds_ipc=True,
    ipc_file=None,
    verbose=True,
):
    """
    Generate Roman WFI ePSF library.

    Returns
    -------
    dict
        Dictionary containing:
            psf
            psf_noipc
            extended_psf
            extended_psf_noipc
            metadata
    """

    pixel_x = ensure_list(pixel_x)
    pixel_y = ensure_list(pixel_y)

    if len(pixel_x) != len(pixel_y):
        raise ValueError(
            "pixel_x and pixel_y must have identical lengths."
        )

    spectral_types = ensure_list(spectral_types)
    temperatures = ensure_list(temperatures)
    logg_values = ensure_list(logg_values)
    defocus_values = ensure_list(defocus_values)

    # --------------------------------------------------------
    # Build source spectra
    # --------------------------------------------------------
    source_dict = build_source_dictionary(
        spectral_types,
        temperatures,
        logg_values,
    )

    # --------------------------------------------------------
    # IPC kernel
    # --------------------------------------------------------
    ipc_kernel = None
    if use_crds_ipc:
        meta = {
            "ROMAN.META.INSTRUMENT.NAME": "WFI",
            "ROMAN.META.INSTRUMENT.DETECTOR": f"WFI{detector:02d}",
            "ROMAN.META.EXPOSURE.START_TIME": "2020-01-01T00:00:00",
        }
        ipc_file = getreferences(
            meta,
            reftypes=["ipc"],
            observatory="roman",
        )["ipc"]

    if ipc_file is not None:
        ipc_dm = rdm.open(ipc_file)
        ipc_kernel = ipc_dm.data

    # --------------------------------------------------------
    # Allocate arrays
    # --------------------------------------------------------
    nfocus = len(defocus_values)
    nspec = len(spectral_types)
    npos = len(pixel_x)

    psf = np.zeros(
        (nfocus,
         nspec,
         npos,
         fov_pixels,
         fov_pixels,
        ),
        dtype=np.float32,
    )
    psf_noipc = None

    if include_noipc:
        psf_noipc = np.zeros_like(psf)

    # --------------------------------------------------------
    # Main PSF loop
    # --------------------------------------------------------
    for fidx, focus in enumerate(defocus_values):
        for sidx, stype in enumerate(spectral_types):
            spectrum = source_dict[stype]["spectrum"]
            for pidx, (xpos, ypos) in enumerate(
                zip(pixel_x, pixel_y)
            ):

                if verbose:
                    print(
                        f"WFI{detector:02d} "
                        f"{optical_element} "
                        f"focus={focus} "
                        f"spec={stype} "
                        f"pos=({xpos}, {ypos})"
                    )

                wfi = roman.WFI()
                band = wfi._get_synphot_bandpass(
                    optical_element
                )
                wave = band.pivot().to(u.m).value
                wfi.options["parity"] = "odd"
                wfi.detector = f"WFI{detector:02d}"
                wfi.filter = optical_element
                wfi.options["defocus_waves"] = focus
                wfi.options["defocus_wavelength"] = wave
                wfi.detector_position = (xpos, ypos)
                scale = wfi.pixelscale / oversample
                wfi.pixelscale = scale

                result = wfi.calc_psf(
                    fov_pixels=fov_pixels,
                    source=spectrum,
                    oversample=1,
                    nlambda=nlambda,
                )

                stamp = result["OVERSAMP"].data.astype(
                    np.float32
                )

                if include_noipc:
                    psf_noipc[fidx, sidx, pidx] = deepcopy(
                        stamp
                    )

                psf[fidx, sidx, pidx] = apply_ipc(
                    deepcopy(stamp),
                    oversample=oversample,
                    ipc_kernel=ipc_kernel,
                )

    # --------------------------------------------------------
    # Extended PSF
    # --------------------------------------------------------
    extended_psf = None
    extended_psf_noipc = None

    if include_extended_psf:

        if verbose:
            print("Generating extended PSF...")

        wfi = roman.WFI()
        band = wfi._get_synphot_bandpass(
            optical_element
        )
        wave = band.pivot().to(u.m).value
        wfi.options["parity"] = "odd"
        wfi.detector = f"WFI{detector:02d}"
        wfi.filter = optical_element
        wfi.detector_position = (2047.5, 2047.5)
        scale = wfi.pixelscale / oversample
        wfi.pixelscale = scale
        ext = wfi.calc_psf(
            fov_pixels=extended_fov_pixels,
            monochromatic=wave,
            oversample=1,
        )
        ext_stamp = ext["OVERSAMP"].data.astype(
            np.float32
        )

        if include_noipc:
            extended_psf_noipc = deepcopy(ext_stamp)

        extended_psf = apply_ipc(
            deepcopy(ext_stamp),
            oversample=oversample,
            ipc_kernel=ipc_kernel,
        )

    # --------------------------------------------------------
    # Return PSF library object
    # --------------------------------------------------------
    return {
        "psf": psf,
        "psf_noipc": psf_noipc,
        "extended_psf": extended_psf,
        "extended_psf_noipc": extended_psf_noipc,
        "metadata": {
            "detector": detector,
            "optical_element": optical_element,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "spectral_types": spectral_types,
            "temperatures": temperatures,
            "logg_values": logg_values,
            "defocus_values": defocus_values,
            "oversample": oversample,
        },
    }


