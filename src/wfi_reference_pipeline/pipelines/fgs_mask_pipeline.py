import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import asdf
import numpy as np
import roman_datamodels as rdm
from astropy.io import fits
from romancal.dq_init import DQInitStep
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep

from wfi_reference_pipeline.config.config_access import get_pipelines_config
from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_FGS_MASK,
)
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.fgs_mask.fgs_mask import FGSMask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# from wfi_reference_pipeline.utilities.logging_functions import log_info


class FGSMaskPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all FGS Mask Pipeline functionality

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save output
    run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    fgs_mask_pipeline = FGSMaskPipeline("<detector string>")
    fgs_mask_pipeline.select_uncal_files()
    fgs_mask_pipeline.prep_pipeline()
    fgs_mask_pipeline.run_pipeline()
    fgs_mask_pipeline.pre_deliver()
    fgs_mask_pipeline.deliver()

    or

    fgs_mask_pipeline.restart_pipeline()

    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_FGS_MASK, detector)
        self.config = get_pipelines_config(REF_TYPE_FGS_MASK)

        self.flat_filelist = []
        self.dark_filelist = []

    def select_uncal_files(self):
        """Select the uncal files to be run through the RFP"""
        # Clearing from previous run
        self.uncal_files.clear()

        # TODO: how would users go about specifying which detector they want
        # to focus on? The paths here are specified in the config file so idk
        files = list(self.ingest_path.glob("*_uncal.asdf"))

        self.uncal_files = files

        logging.info(f"Ingesting {len(files)} files: {files}")

    def prep_pipeline(self, file_list=None):
        """
        Prepare calibration data files by running data through select romancal steps.
        Sort the filelist, then create the superdark and superflat from the calibrated files.
        """
        logging.info("FGS_MASK PREP")

        # Clean up previous runs
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = list(map(Path, self.uncal_files))

        # TODO: will file_list also contain the flat rate images? 
        for file in file_list:
            if "flat" in os.path.basename(file):
                logging.info(f"Skipping {os.path.basename(file)} since flats are already run through romancal")
                prep_output_file_path = self.file_handler.format_prep_output_file_path(
                    os.path.basename(file)
                )
                shutil.copy(file, prep_output_file_path)
                self.prepped_files.append(prep_output_file_path)
                continue

            logging.info("OPENING - " + file.name)

            # TODO: only need to prep dark calibration (not flats)
            # The romancal flat file will be called L2. Rick made a PARS file to auto run romancal

            self._run_romancal(file)

        logging.info("Finished PREPPING files to make FGS_MASK reference file from RFP")

        logging.info("Sorting prepped files into darks and flats")
        self._sort_filelist()

        logging.info("Creating a superdark using files: ", self.dark_filelist)
        self.prep_superdark()

        logging.info("Creating super rate image using files: ", self.flat_filelist)
        self.prep_super_rate()

        logging.info("All files prepped, ready to run FGSMask")

    def prep_superdark(self):
        """
        Create a superdark from the prepped self.dark_filelist files.
        This function uses the DarkPipeline superdark code. 
        """
        # Need the number of reads to run the superdark code
        nreads = self._get_nreads()

        # Setting the superdark path to be in the same dir as the prepped files
        self.superdark_path = os.path.join(self.prep_path, "superdark.asdf")

        logging.info("Creating superdark and writing file to", self.superdark_path)

        # Creating the dark pipeline object and creating the superdark
        dark_pipe = DarkPipeline(self.detector)
        dark_pipe.prep_superdark_file(
            short_file_list=self.dark_filelist,
            outfile=self.superdark_path,
            short_dark_num_reads=nreads,
        )

        # Loading the superdark and setting as attr
        self._load_superdark()

        return
    
    def prep_super_rate(self):
        """
        This function creates a super rate image by averaging the inputted flat rate files.
        The super rate image is then set as the attribtue `self.super_rate_image`
        """
        rate_images = np.zeros((len(self.flat_filelist), DETECTOR_PIXEL_Y_COUNT, DETECTOR_PIXEL_X_COUNT))

        for i, file in enumerate(self.flat_filelist):
            with asdf.open(file, memmap=True) as af:
                
                data = af["roman"]["data"]
                data = data.value if hasattr(data, "value") else data

                rate_images[i, :, :] = data

        # Calculating the super rate image
        self.super_rate_image = np.nanmean(rate_images, axis=0)
        

    def run_pipeline(self, file_list=None):
        """
        With the superdark and super rate image, run the FGS workflow on the
        files to create a DQ bitmask using the FGSFlags in fgs_mask.py.
        """
        logging.info("FGS_MASK PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.prepped_files

        tmp = MakeDevMeta(ref_type=self.ref_type)

        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_fgs_mask.mode,
            tmp.meta_fgs_mask.instrument_detector,
        )

        self.rfp_fgs_mask = FGSMask(
            meta_data=tmp.meta_fgs_mask,
            superdark=self.superdark,
            super_rate_image=self.super_rate_image,
            outfile=out_file_path,
            clobber=True,
        )

        # Running the actual FGS mask algorithms
        self.rfp_fgs_mask.make_fgs_mask_image()

        self.rfp_fgs_mask.generate_outfile()

        logging.info("Finished RFP to make FGS_MASK")

    def pre_deliver(self, file_change_note=None):
        """
        Converting the new bitmask to DETECTOR coordinates and
        creating a boolean mask where all non-zero values = 1.
        NOTE: This is the format that will be delivered to PSS.

        Parameters
        ----------
        file_change_note : str, optional
            String that indicates if any major changes are associated with the creation
            of this mask. Default is None.
        """
        self.convert_mask_to_pss_format()
        self.save_pss_mask(file_change_note=file_change_note)

    def deliver(self):
        pass
    
    def convert_mask_to_pss_format(self):
        """
        Put the DQ mask into detector coordinates, then convert to boolean.
        The converted PSS mask is set as `self.mask_pss`.
        """
        mask_det_coords = self._change_coord_to_det(self.rfp_fgs_mask.mask_image)
        self.mask_pss = self._convert_mask_to_boolean(mask_det_coords)

    # TODO: Is it the standard for the default value to be explicitly set in 
    # subsequent functions? or just the first time it's used? (file_change_note)
    def save_pss_mask(self, file_change_note):
        """
        Using the PSS mask, create the FITS file with the correct header.
        The outpath of the new mask file is set as `self.pss_mask_outpath`.
        """

        mask_header = self._make_fits_header(file_change_note=file_change_note)
        hdu = fits.PrimaryHDU(data=self.mask_pss,
                              header=mask_header)
        
        self.pss_mask_outpath = os.path.join(self.file_handler.pipeline_out_path,
                                             f"fgs_mask_{self.detector}.fits")
    
        hdu.writeto(self.pss_mask_outpath,
                    overwrite=True)

    def _make_fits_header(self, file_change_note):
        """
        Create the FITS header for the PSS mask.
        """
        hdr = fits.Header()

        hdr['DETECTOR'] = (self.detector, 'WFI detector number')
        hdr['AUTHOR']   = (self.rfp_fgs_mask.meta_data.author, 'Author of file')
        hdr['DATETIME'] = (datetime.now().strftime('%Y-%m-%d'), 'Date of file creation')
        hdr['NBADPIX']  = (int(np.count_nonzeros(self.mask_pss)), 'Number of bad pixels')
        hdr['CHANGE_NOTE'] = (file_change_note, 'High-level changes to mask derivation')

        return hdr
        

    def _convert_mask_to_boolean(self, mask):
        """Return a mask where any non-zero value = 1"""
        return (mask != 0).astype("uint8")

    def _change_coord_to_det(self, arr):
        """
        Change the detector coordinates from DETECTOR to SCIENCE (run again to undo). Dependent on detector.
        Code from Sarah Betti
        """
        # Detector coordinate positions; GSFC uses detector, SOC uses science
        detector_pos = {
            "WFI01": "upper left",
            "WFI02": "upper left",
            "WFI03": "lower right",
            "WFI04": "upper left",
            "WFI05": "upper left",
            "WFI06": "lower right",
            "WFI07": "upper left",
            "WFI08": "upper left",
            "WFI09": "lower right",
            "WFI10": "upper left",
            "WFI11": "upper left",
            "WFI12": "lower right",
            "WFI13": "upper left",
            "WFI14": "upper left",
            "WFI15": "lower right",
            "WFI16": "upper left",
            "WFI17": "upper left",
            "WFI18": "lower right",
        }

        position = detector_pos[self.detector]

        if position == "lower right":
            return arr[:, ::-1]

        else:
            return arr[::-1]

    def restart_pipeline(self):

        self.select_uncal_files()
        self.prep_pipeline()
        self.run_pipeline()
        self.pre_deliver()
        self.deliver()

        return
    
    def _run_romancal(self, file):
        """
        Run romancal on a single file. Add the output filepath to `self.prepped_files`.
        """
        with rdm.open(file) as f:

            result = DQInitStep.call(f, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = RefPixStep.call(result, save_results=False)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

    
    def _sort_filelist(self):
        """
        Sort the prepped files into flats and darks based on filename.
        """
        logging.info("Sorting the files into flats vs darks in self.file_list")

        invalid_files = []

        for file in self.prepped_files:
            filename = os.path.basename(file).lower()

            if "flat" in filename:
                self.flat_filelist.append(file)

            elif "dark" in filename:
                self.dark_filelist.append(file)

            else:
                invalid_files.append(file)

        if invalid_files:
            # TODO: should we set this as an attr instead of raising an error?
            raise ValueError("The following files can not be sorted in prepped flats or darks:", invalid_files)

    def _get_nreads(self):
        """Using the first file in self.dark_filelist, get the number of reads in the ramp."""
        if not self.dark_filelist:
            raise TypeError("No prepped dark files found in self.dark_filelist. Cannot make superdark.")
        
        with asdf.open(self.dark_filelist[0], memmap=True) as af:
            data = af["roman"]["data"]
            dark = data.value if hasattr(data, "value") else data
            nreads = dark.shape[0]

        return nreads
    
    def _load_superdark(self):
        """Load the newly-created superdark file"""
        logging.info("Loading superdark from", self.superdark_path)

        with asdf.open(self.superdark_path, memmap=True) as af:
            data = af["roman"]["data"]
            superdark = data.value if hasattr(data, "value") else data
            self.superdark = np.asarray(superdark)