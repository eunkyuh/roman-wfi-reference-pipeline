import logging
from pathlib import Path
import os
import asdf
import numpy as np

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep

from wfi_reference_pipeline.config.config_access import get_pipelines_config
from wfi_reference_pipeline.constants import REF_TYPE_FGS_MASK
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.fgs_mask.fgs_mask import FGSMask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT

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

            logging.info("OPENING - " + file.name)

            # TODO: only need to prep dark calibration (not flats)
            # The romancal flat file will be called L2. Rick made a PARS file to auto run romancal

            result = self._run_romancal(file)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

        logging.info("Finished PREPPING files to make FGS_MASK reference file from RFP")

        logging.info("Sorting prepped files into darks and flats")
        self._sort_filelist()

        logging.info("Creating a superdark using files: ", self.dark_filelist)
        self.prep_superdark()

        logging.info("Creating super rate image using files: ", self.flat_filelist)
        self.prep_super_rate()

        logging.info("All files prepped, ready to run FGSMask")

    def prep_superdark(self):
        """Create a superdark from the prepped self.dark_filelist files."""
        # Need the number of reads to run the superdark code
        nreads = self._get_nreads()

        # Setting the superdark path to be in the same dir as the prepped files
        self.superdark_path = os.path.join(self.prep_path, "superdark.asdf")

        logging.info("Creating superdark and writing file to", self.superdark_path)

        # Creating the dark pipeline object and creating the superdark
        dark_pipe = DarkPipeline(self.detector)
        dark_pipe.prep_superdark_file(
            full_file_list=self.dark_filelist,
            outfile=self.superdark_path,
            full_file_num_reads=nreads,
        )

        # Loading the superdark and setting as attr
        self._load_superdark()

        return
    
    def prep_super_rate(self):
        """The prepped flats will already be rate images. Create a super rate image."""
        rate_images = np.zeros((len(self.flat_filelist, DETECTOR_PIXEL_Y_COUNT, DETECTOR_PIXEL_X_COUNT)))

        for i, file in (self.flat_filelist):
            with asdf.open(file, memmap=True) as af:
                
                data = af["roman"]["data"]
                data = data.value if hasattr(data, "value") else data

                rate_images[i, :, :] = data

        # Calculating the super rate image
        self.super_rate_image = np.nanmean(rate_images, axis=0)
        

    def run_pipeline(self, file_list=None):

        logging.info("FGS_MASK PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.prepped_files

        tmp = MakeDevMeta(
            ref_type=self.ref_type
        )  # TODO replace with MakeMeta which gets actual information from files
        # fgs_mask_dev_meta = tmp.meta_fgs_mask.export_asdf_meta()
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_fgs_mask.mode,
            tmp.meta_fgs_mask.instrument_detector,
        )

        rfp_fgs_mask = FGSMask(
            meta_data=tmp.meta_fgs_mask,
            superdark=self.superdark,
            normalized_super_rate=self.normalized_super_rate,
            outfile=out_file_path,
            clobber=True,
        )

        rfp_fgs_mask.make_fgs_mask_image()
        rfp_fgs_mask.generate_outfile()
        logging.info("Finished RFP to make FGS_MASK")

    def pre_deliver(self):
        """This is where the coord transformation + boolean impl goes"""
        self._change_coord_to_det()
        self._change_to_boolean()

        # PSS expects the mask as FITS boolean file in DETECTOR coordinates (not SCIENCE)
        logging.info("Transforming mask to boolean mask in DETECTOR coordinates")
        binary_mask = (mask != 0).astype("uint8")
        binary_mask_det = change_coord_to_det(binary_mask, det)

        logging.info("Writing transformed boolean mask to FITS")
        binary_mask_path = os.path.join(basedir, f"binary_mask_{det}.fits")
        fits.writeto(binary_mask_path, data=binary_mask_det, overwrite=True)
        mask_path = os.path.join(basedir, f"mask_{det}.fits")
        fits.writeto(mask_path, data=mask, overwrite=True)
        return 
    
    def _change_coord_to_det(self):
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
        Run romancal on a single file. Created so multiprocessing's Pool can be implemented.
        """
        with rdm.open(file) as f:
            result = DQInitStep.call(f, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = RefPixStep.call(result, save_results=False)

        return result
    
    def _sort_filelist(self):
        """
        Sort the prepped files into flats and darks.
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
        
        with asdf.open(self.dark_filelist[0], memmap=True) as dm:
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