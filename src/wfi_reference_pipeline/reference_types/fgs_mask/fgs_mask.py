import logging
import math
from enum import Enum

import numpy as np
from astropy.stats import sigma_clip

from wfi_reference_pipeline.constants import WFI_TYPE_IMAGE
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.resources.wfi_meta_fgs_mask import WFIMetaFGSMask

from ..reference_type import ReferenceType


# SRG: Add comments, put in order. Remove TFPN
# How to coordinate w RDMT is more important than choosing a base for bitwi 
class FGSFlags(np.uint32, Enum):
    """
    These are the flags that are used ONLY FOR THE FGS MASK.
    For flags used in the SCIENCE BPM, see the roman_datamodels repo.
    """
    GOOD = 0
    GW_AFFECTED_DATA = 2**4
    PERSISTENCE = 2**5
    DEAD = 2**10
    HOT_PIXEL = 2**11
    SUPERHOT_PIXEL = 2**12
    FLAT_FIELD = 2**18
    HIGH_CDS_NOISE = 2**26
    LOW_QE_OPTICAL = 2**27
    OTHER_BAD_PIXEL = 2**30
    REFERENCE_PIXEL = 2**31
    HOT_FROM_GW = 2**29


class FGSMask(ReferenceType):
    """
    Class FGSMask() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(
        self,
        meta_data,
        file_list=None, # TODO can this line be deleted since unused?
        ref_type_data=None, # TODO can this line be deleted since unused?
        superdark=None,
        super_rate_image=None,
        outfile="roman_fgs_mask.asdf",
        clobber=False,
    ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        TODO: delete from docstring? file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        TODO: delete from docstring? ref_type_data: numpy array; default = None
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        superdark: np.ndarray; default = None
            The superdark that will be used to calculate the CDS noise and dark rate images.
        super_rate_image: np.ndarray; default = None
            This is dataproduct generated using flat-field exposures. It is a Super Flat that has been slope-fitted.
            The super_rate_image is used to identify low QE, dead, and bad flat-field pixels.
        outfile: string; default = roman_flat.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            outfile=outfile,
            clobber=clobber,
            file_list=[""],
        )

        # The superdark and super rate images are created in the pipeline file
        self.superdark = superdark
        self.super_rate_image = super_rate_image

        # Creating an empty mask to be filled in
        self.mask_image = np.zeros((4096, 4096), dtype=np.uint32)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaFGSMask):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaFGSMask"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI FGS mask reference file."


    def make_fgs_mask_image(self, do_sigma_clip=True, sig_clip_cds_low=5.0, sig_clip_cds_high=5.0, dead_sigma_thr=5.0, hot_thr=2.5, superhot_thr=20.0, high_cds_thr=11.0, low_qe_thr=0.3, bad_flat_thr=0.0):
        """
        Run the full FGS mask generation workflow.

        Then create the normalized super rate image and CDS noise/dark
        rate images, and identify and flag bad pixels of each type.
        
        NOTE: The final product is a bitmask, not the boolean mask that PSS expects.

        Parameters
        ----------
        do_sigma_clip : bool, optional
            If True, apply sigma clipping when computing CDS noise. Default is True.
        sig_clip_cds_low : float, optional
            Lower-sigma threshold for CDS noise sigma clipping. Default is 5.0.
        sig_clip_cds_high : float, optional
            Upper-sigma threshold for CDS noise sigma clipping. Default is 5.0.
        dead_sigma_thr : float, optional
            Number of standard deviations below the median slope used to identify
            dead pixels. Default is 5.0.
        hot_thr : float, optional
            Dark rate threshold in DN above which pixels are flagged as hot. Default is 2.5.
        superhot_thr : float, optional
            Dark rate threshold in DN above which pixels are flagged as superhot. Default is 20.0.
        high_cds_thr : float, optional
            CDS noise threshold in DN above which pixels are flagged as high CDS noise. Default is 11.0.
        low_qe_thr : float, optional
            Normalized super rate threshold below which pixels are flagged as low QE. Default is 0.3.
        bad_flat_thr : float, optional
            Normalized super rate threshold below which pixels are flagged as bad flat field. Default is 0.0.
        """
        logging.info("Creating the normalized super rate image")
        self.create_normalized_super_rate_im()
        
        logging.info("Creating the CDS noise and dark rate images")
        self.create_cds_noise_darkrate_im(do_sigma_clip=do_sigma_clip,
                                          sig_clip_cds_low=sig_clip_cds_low,
                                          sig_clip_cds_high=sig_clip_cds_high)

        logging.info("Beginning bad pixel identification")

        logging.info(f"Setting DEAD pixels using a threshold of {dead_sigma_thr} sigma")
        self.set_dead_pixels(dead_sigma_thr=dead_sigma_thr)

        logging.info(f"Setting HOT and SUPERHOT pixels using a threshold of {hot_thr} DN and {superhot_thr} DN")
        self.set_hot_superhot_pixels(hot_thr=hot_thr, 
                                     superhot_thr=superhot_thr)
        
        logging.info(f"Setting HIGH_CDS_NOISE pixels using threshold of {high_cds_thr} DN")
        self.set_high_cds_noise_pixels(high_cds_thr=high_cds_thr)

        logging.info(f"Setting LOW_QE pixels using a threshold of {low_qe_thr}")
        self.set_low_qe_pixels(low_qe_thr=low_qe_thr)
        
        logging.info(f"Setting BAD_FLAT_FIELD pixels using a threshold of {bad_flat_thr}")
        self.set_bad_flat_field_pixels(bad_flat_thr=bad_flat_thr)

        logging.info("Finished running FGS mask workflow!")

    def create_normalized_super_rate_im(self):
        """
        Normalize the super rate image by its nanmean.

        Computes the normalized super rate image by dividing the super rate
        image by its nanmean and stores the result in self.normalized_super_rate.
        """
        normalized_super_rate = self.super_rate_image / np.nanmean(self.super_rate_image)
        self.normalized_super_rate = normalized_super_rate

        return

    def set_dead_pixels(self, dead_sigma_thr=5.0):
        """
        Flag pixels as DEAD in the FGS mask image.

        A pixel is flagged as DEAD if its value in the super rate image falls
        more than dead_sigma_thr standard deviations below the median slope.

        Parameters
        ----------
        dead_sigma_thr : float, optional
            Number of standard deviations below the median used as the dead
            pixel threshold. Default is 5.0.
        """
        median_slope = np.median(self.super_rate_image)
        std_slope = np.std(self.super_rate_image)

        dead_threshold = median_slope - (dead_sigma_thr * std_slope)
        dead_mask = self.super_rate_image < dead_threshold

        self.mask_image[dead_mask] += FGSFlags.DEAD

        return
    
    def set_hot_superhot_pixels(self, hot_thr=2.5, superhot_thr=20.0):
        """
        Flag pixels as HOT_PIXEL in the mask image.

        Pixels with a dark rate above hot_thr are flagged as HOT_PIXEL.
        Pixels with a dark rate above superhot_thr are flagged as SUPERHOT_PIXEL.

        Parameters
        ----------
        hot_thr : float, optional
            Dark rate threshold in DN above which pixels are flagged as hot. Default is 2.5.
        superhot_thr : float, optional
            Dark rate threshold in DN above which pixels are flagged as superhot. Default is 20.0.
        """
        hot_mask = self.darkrate_image > hot_thr
        self.mask_image[hot_mask] += FGSFlags.HOT_PIXEL

        superhot_mask = self.darkrate_image > superhot_thr
        self.mask_image[superhot_mask] += FGSFlags.SUPERHOT_PIXEL

        return

    def set_high_cds_noise_pixels(self, high_cds_thr=11.0):
        """
        Flag pixels as HIGH_CDS_NOISE in the mask image.

        Parameters
        ----------
        high_cds_thr : float, optional
            CDS noise threshold in DN above which pixels are flagged. Default is 11.0.
        """
        cds_mask = self.cds_noise > high_cds_thr
        self.mask_image[cds_mask] += FGSFlags.HIGH_CDS_NOISE

        return

    def set_low_qe_pixels(self, low_qe_thr=0.3):
        """
        Flag pixels as LOW_QE_OPTICAL in the mask image.

        Parameters
        ----------
        low_qe_thr : float, optional
            Normalized super rate threshold below which pixels are flagged as
            low QE. Default is 0.3.
        """
        qe_mask = self.normalized_super_rate < low_qe_thr
        self.mask_image[qe_mask] += FGSFlags.LOW_QE_OPTICAL

        return

    def set_bad_flat_field_pixels(self, bad_flat_thr=0.0):
        """
        Flag pixels as FLAT_FIELD in the mask image.

        Parameters
        ----------
        bad_flat_thr : float, optional
            Normalized super rate threshold below which pixels are flagged as
            bad flat field. Default is 0.0.
        """
        flat_mask = self.normalized_super_rate < bad_flat_thr
        self.mask_image[flat_mask] += FGSFlags.FLAT_FIELD

        return
    
    # TODO: should this just be in make_fgs_mask_image?
    def create_cds_noise_darkrate_im(self, do_sigma_clip=True, sig_clip_cds_low=5.0, sig_clip_cds_high=5.0):
        """
        Create the CDS noise and dark rate images from the superdark.

        Builds a ReadNoise data cube from self.superdark, fits a ramp model,
        computes the CDS noise image, and stores the dark rate image in
        self.darkrate_image.

        Parameters
        ----------
        do_sigma_clip : bool, optional
            If True, apply sigma clipping when computing CDS noise. Default is True.
        sig_clip_cds_low : float, optional
            Lower sigma threshold for CDS noise sigma clipping. Default is 5.0.
        sig_clip_cds_high : float, optional
            Upper sigma threshold for CDS noise sigma clipping. Default is 5.0.
        """
        logging.info("Creating ReadNoise data cube")
        self.readnoise_cube = ReadNoise.ReadNoiseDataCube(self.superdark,
                                                          WFI_TYPE_IMAGE)
        # Prep CDS noise computations
        self.readnoise_cube.fit_cube(degree=1)
        self.readnoise_cube.make_ramp_model(order=1)

        # Compute and write CDS noise image
        # TODO the comp_cds_noise function is for the ReadNoise class,
        # so I would need to create that obj to use the function. IMO it's
        # more lines to do that than to just redefine the function
        self.compute_cds_noise_from_datacube(do_sigma_clip=do_sigma_clip,
                                             sig_clip_cds_low=sig_clip_cds_low, 
                                             sig_clip_cds_high=sig_clip_cds_high)
        
        logging.info("Creating darkrate image")
        self.darkrate_image = self.readnoise_cube.rate_image

    def compute_cds_noise_from_datacube(self, do_sigma_clip=True, sig_clip_cds_low=5.0, sig_clip_cds_high=5.0):
        """
        Compute the CDS noise image from the readnoise data cube.

        Computes pairwise read differences from the ramp-subtracted data cube,
        optionally sigma-clips them, and stores the per-pixel standard deviation
        as self.cds_noise.

        Parameters
        ----------
        do_sigma_clip : bool, optional
            If True, apply sigma clipping across reads before computing noise. Default is True.
        sig_clip_cds_low : float, optional
            Lower sigma threshold for clipping. Default is 5.0.
        sig_clip_cds_high : float, optional
            Upper sigma threshold for clipping. Default is 5.0.
        """
        logging.info("Computing CDS noise image")
        read_diff_cube = np.zeros(
            (math.ceil(self.readnoise_cube.num_reads / 2),
             self.readnoise_cube.num_i_pixels,
             self.readnoise_cube.num_j_pixels,),
            dtype=np.float32,)

        for i_read in range(0, self.readnoise_cube.num_reads - 1, 2):
            # Avoid index error if num_reads is odd and disregard the last read because it does not form a pair.
            rd1 = self.readnoise_cube.ramp_model[i_read, :, :] - self.readnoise_cube.data[i_read, :, :]
            rd2 = self.readnoise_cube.ramp_model[i_read + 1, :, :] - self.readnoise_cube.data[i_read + 1, :, :]

            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1

        if do_sigma_clip:
            logging.info("Performing sigma-clipping on read differences before calculating CDS noise")
            clipped_diff_cube = sigma_clip(
                read_diff_cube,
                sigma_lower=sig_clip_cds_low,
                sigma_upper=sig_clip_cds_high,
                cenfunc=np.mean,
                axis=0,
                masked=False,
                copy=False)

            cds_noise = np.std(clipped_diff_cube, axis=0)

        else:
            cds_noise = np.std(read_diff_cube, axis=0)

        self.cds_noise = cds_noise

    def calculate_error(self):
        """
        Abstract method not applicable to FGSMask.
        """

        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized by FGSMask().

        NOTE - Similar to Mask(), this would be redundant to make_mask_image(). 
        The attribute mask is reserved specifically setting the data quality arrays
        of other reference file types.
        """

        pass

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.

        NOTE: This is the "intermediate DQ product", based off the Mask datamodel.
        The actual FGS mask that is delievered to PSS is created in fgs_mask_pipeline.py.
        """

        datamodel_tree = {
            'meta': self.meta_data.export_asdf_meta(),
            'dq': self.mask_image
        }

        return datamodel_tree
