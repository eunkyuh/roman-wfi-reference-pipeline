import logging
import os
import shutil
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import asdf
import crds
import numpy as np
import roman_datamodels as rdm
from astropy.time import Time
from crds.client import api as crds_api
from romancal.dark_decay import DarkDecayStep
from romancal.dq_init import DQInitStep
from romancal.linearity import LinearityStep
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep
from romancal.wfi18_transient import WFI18TransientStep

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_MASK,
)
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta


class MaskPipeline(Pipeline):
    """
    Derived from the Pipeline Base Class
    This is the entry point for Mask Pipeline functionality.

    Gives user access to:
    select_uncal_files : Selecting Level 1 uncalibration asdf files
    prep_pipeline : Prepare the pipeline using romancal routines and save outputs
    run_pipeline : Process the data and create a new calibration asdf file for CRDS delivery
    restart_pipeline : Run all steps from scratch (derived from Pipeline)

    Usage:
    mask_pipeline = MaskPipeline("<detector string>")
    mask_pipeline.select_uncal_files()
    mask_pipeline.prep_pipeline()
    mask_pipeline.run_pipeline()
    mask_pipeline.pre_deliver()
    mask_pipeline.deliver()

    or

    mask_pipeline.restart_pipeline()
    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_MASK, detector)

        self.mask_file = None

        self.flat_filelist = []
        self.dark_filelist = []

        self.superdark = None
        self.super_rate_image = None


    def select_uncal_files(self):
        # Clearing from previous run
        self.uncal_files.clear()

        files = list(self.ingest_path.glob("*_uncal.asdf"))
        self.uncal_files = files

        logging.info(f"Ingesting {len(files)} files: {files}")


    def prep_pipeline(self, file_list=None):
        """
        Prepare calibration data files by running data through select romancal steps.
        Sort the filelist, then create the superdark and superflat from the calibrated files.
        """
        # Clearing up the previous run
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # This will be a directory for the CRDS masks, which will be cleared each mask run
        crds_directory = os.path.join(self.file_handler.prep_path, "crds_mask")
        if not os.path.exists(crds_directory):
            os.makedirs(crds_directory)

        self.crds_directory = crds_directory
        logging.info(f"Previous CRDS Mask file for {self.detector} will be in {crds_directory}")

        # Download the previous mask from CRDS and store it in self.crds_directory
        self._get_previous_mask_from_crds()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = list(map(Path, self.uncal_files))

        for file in file_list:
            logging.info("OPENING - " + file.name)
            perform_additional_steps = False

            if "flat" in os.path.basename(file):
                perform_additional_steps = True

            self._run_romancal(file,
                               perform_additional_steps=perform_additional_steps)

        try:
            romancal_args = [(file, perform_additional_steps) for file in file_list]

            with Pool() as pool:
                [] = pool.starmap(self._run_romancal, romancal_args)

        except Exception as e:
            print(f"Error processing files: {e}")

        finally:
            pool.close()
            pool.join()

        #self.prepped_files = prepped_files

        logging.info(f"The following files for {self.detector} have been prepped to run through the Mask pipeline: {self.prepped_files}")

        logging.info("Sorting prepped files into darks and flats")
        self._sort_filelist()

        if self.dark_filelist:
            logging.info(f"Creating a superdark using files: {self.dark_filelist}")
            self.prep_superdark()

        if self.flat_filelist:
            logging.info(f"Creating super rate image using files: {self.flat_filelist}")
            self.prep_super_rate()

        logging.info("Finished prepping the files to make Mask reference file from the RFP")


    def run_pipeline(self, save_intermed=True):
        """Run the Mask pipeline on self.prepped_files."""
        logging.info(f"Beginning to run Mask pipeline for {self.detector}")

        tmp = MakeDevMeta(ref_type=self.ref_type)

        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_mask.mode,
            tmp.meta_mask.instrument_detector,
        )

        rfp_mask = Mask(
            meta_data=tmp.meta_mask,
            superdark=self.superdark,
            super_rate_image=self.super_rate_image,
            ref_type_data=None,
            outfile=out_file_path,
            clobber=True,
        )

        logging.info("Beginning to run `make_mask_image`")
        rfp_mask.make_mask_image()

        logging.info(f"Generating the outfile Mask for {self.detector}")
        rfp_mask.generate_outfile()

        logging.info("Mask pipeline run is complete.")

        if save_intermed:
            self.save_intermediate_products(rfp_mask)


    def pre_deliver(self):
        pass


    def deliver(self):
        pass


    def prep_superdark(self):
        """
        Create a superdark from the prepped self.dark_filelist files.
        This function uses the DarkPipeline superdark code. 
        """
        # Need the number of reads to run the superdark code
        nreads = self._get_nreads()

        # Setting the superdark path to be in the same dir as the prepped files
        self.superdark_path = os.path.join(self.prep_path, f"superdark_for_mask_{self.detector}.asdf")

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

                # TODO: are we getting rate images ? 
                rate_images[i, :, :] = data[i, :, :]

        # Calculating the super rate image
        self.super_rate_image = np.nanmean(rate_images, axis=0)


    def save_intermediate_products(self, rfp_mask):
        """
        After the Mask module has run, save various intermediate products in the same
        directory as the superdark.
        """
        if self.super_rate_image is not None:
            self.super_rate_path = os.path.join(self.prep_path, f"super_rate_for_mask_{self.detector}.asdf")
            self._save_intermed_product(intermed_type="super_rate",
                                        data_tree={"data": self.super_rate_image},
                                        outpath=self.super_rate_path,
                                        filelist=self.flat_filelist)
        
        if hasattr(rfp_mask, "jump_count_img"):
            self.jump_path = os.path.join(self.prep_path, f"jump_products_{self.detector}.asdf")
            data_tree = {"jump_mask_cube": rfp_mask.jump_mask_cube, "jump_count_img": rfp_mask.jump_count_img}
            self._save_intermed_product(intermed_type="jump_products",
                                        data_tree=data_tree,
                                        outpath=self.jump_path)
            
        if hasattr(rfp_mask, "metrics_df"):
            self.metrics_df_path = os.path.join(self.prep_path, f"metrics_df_{self.detector}.csv")
            rfp_mask.metrics_df.to_csv(self.metrics_df_path)
            logging.info("Saved", self.metrics_df_path)


    def _get_previous_mask_from_crds(self):
        """
        Get the older mask from CRDS to be used in the DQInit step. This function uses the CRDS API
        to download the last mask file for a given detector. 
        The following attributes are set or used in this function: 
            self.crds_directory : The path to the directory that the CRDS mask is downloaded to.
                                  This directory is created in `prep_pipeline` and cleared after each run.
            
            self.prev_mask_filepath : The path to the given detector's CRDS mask.

            self.prev_mask_image : The 2D DQ array of the previous CRDS mask.
        """
        logging.info(f"Downloading the latest Mask file for {self.detector} from CRDS")

        # This is where the CRDS files will be downloaded to
        os.environ["CRDS_PATH"] = self.crds_directory
        logging.info(f"CRDS_PATH: {os.environ.get('CRDS_PATH')}")

        if len(os.listdir(self.crds_directory)) > 0:
            logging.info(f"Clearing out previous Mask Pipeline's CRDS files in {self.crds_directory}")
            shutil.rmtree(self.crds_directory)
            os.makedirs(self.crds_directory)

        crds_api.set_crds_server(os.environ["CRDS_SERVER_URL"])
        logging.info(f"CRDS_SERVER_URL: {os.environ.get('CRDS_SERVER_URL')}")

        crds_context = crds.get_default_context()
        logging.info(f"CRDS context: {crds_context}")

        logging.info(f"Syncing CRDS reference files for {self.detector}...")
        params = {
            "ROMAN.META.INSTRUMENT.NAME": "WFI",
            "ROMAN.META.INSTRUMENT.DETECTOR": self.detector,
            "ROMAN.META.EXPOSURE.START_TIME": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        }

        try:
            mask_refs = crds.getreferences(params,
                                           reftypes=["mask"],
                                           context=crds_context,
                                           observatory="roman")

        except Exception as e:
            logging.info(f"Unable to retrieve the latest Mask reference file for {self.detector} using parameters {params}, {e}")

        mask_file = mask_refs["mask"]
        self.prev_mask_filepath = mask_file

        logging.info(f"Loading the previous Mask from {mask_file}")
        try:
            with rdm.open(mask_file, memmap=True) as dm:

                dq_array = np.array(dm.dq)
                self.prev_mask_image = dq_array

        except Exception as e:
            logging.info(f"Unable to load the Mask from CRDS, {e}")


    def _run_romancal(self, file, perform_additional_steps=False):
        """
        Run romancal on a single file. Add the output filepath to `self.prepped_files`.

        Parameters:
        -----------
        file : str
            The L1 file to be run through romancal.
        perform_additional_steps : bool, default=False
            If True, run all steps of romancal up to the Linearity step.
        """
        with rdm.open(file) as f:

            result = DQInitStep.call(f, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = RefPixStep.call(result, save_results=False)

            if perform_additional_steps:
                result = DarkDecayStep.call(result, save_results=False)
                result = WFI18TransientStep.call(result, save_results=False)
                result = LinearityStep.call(result, save_results=False)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

    
    def _sort_filelist(self):
        """
        Sort the prepped files into flats and darks based on filename.
        """
        logging.info("Sorting the files into flats vs darks in self.prepped_files")

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


    def _save_intermed_product(self, intermed_type, data_tree, outpath, filelist=None, file_permission=0o666):
        """
        Save the intermediate product in `self.prep_path` directory.
        The file is saved as an ASDF file, with structure similar to
        usual Roman data products.
        """
        # Building metadata similar to SuperDark class
        meta = {
            "pedigree": "DUMMY",
            "description": f"{intermed_type} intermediate calibration product generated from RFP Mask module.",
            "date": Time(datetime.now()),
            "detector": self.detector,
        }

        if filelist:
            meta["filelist"] = filelist
        
        datamodel_tree = {
            "meta": meta
        }

        for key, value in data_tree.items():
            datamodel_tree[key] = value
        
        af = asdf.AsdfFile()
        af.tree = {"roman": datamodel_tree}

        af.write_to(outpath)
        os.chmod(outpath, file_permission)

        logging.info("Saved", outpath)
