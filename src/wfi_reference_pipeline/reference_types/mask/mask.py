import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from astropy.convolution import Box2DKernel, convolve
from roman_datamodels.datamodels import MaskRefModel
from roman_datamodels.dqflags import pixel as dqflags
from scipy.optimize import curve_fit
from scipy.stats import anderson, kurtosis, linregress, skew

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
)
from wfi_reference_pipeline.reference_types.reference_type import ReferenceType
from wfi_reference_pipeline.resources.wfi_meta_mask import WFIMetaMask


class Mask(ReferenceType):
    """
    Mask() generates a Roman WFI bad pixel mask reference file.

    This class identifies and flags detector pixels exhibiting anomalous
    behavior using IRRC-corrected flat-field and dark integrations, producing a static
    data quality (DQ) mask suitable for delivery to CRDS.

    The following pixel classes are identified:
        - DEAD / LOW_QE / OPEN / ADJ_OPEN (from flat fields)
        - RC (ramp exhibits double exponential behavior)
        - TELEGRAPH (multi-level, random, switching behavior)
        - OTHER_BAD_PIXEL (ambiguous or unclassifiable behavior)

    Threshold values are empirically chosen based on simulated and pre-flight
    data and are expected to be refined as on-orbit behavior becomes better
    characterized.

    Note: The following bad pixel classes do not have a designated flag in romancal (yet).
        - OPEN : RESERVED_5
        - ADJ_OPEN : RESERVED_6
        - RC_IRC : RESERVED_7
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        superdark=None,
        super_rate_image=None,
        bit_mask=None,
        outfile="roman_mask.asdf",
        clobber=False,
    ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.

        file_list: List of strings; default = None
            List of file names with absolute paths to Darks and Flats

        ref_type_data: numpy array; default = None
            Input data cube. Intended as input for Mask not generated from a file list.

        superdark: np.ndarray; default = None
            The superdark that will be used to calculate the dark rate images / ramps.

        super_rate_image: np.ndarray; default = None
            This is a data product generated using flat-field exposures. It is a Super Flat that has been slope-fitted.

        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.

        outfile: string; default = roman_mask.asdf
            File path and name for saved reference file.

        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType.
        super().__init__(
            meta_data=meta_data,
            outfile=outfile,
            clobber=clobber,
            file_list=[""],
            ref_type_data=ref_type_data
        )

        # Initialize attributes
        self.mask_image = None
        self.super_rate_image = super_rate_image
        self.superdark = superdark

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaMask):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaMask."
            )

        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI mask reference file."

        logging.info(f"Default mask reference file object: {outfile}.")

        # Checking for valid input types
        if ref_type_data is not None:
            if not isinstance(ref_type_data, np.ndarray):
                raise ValueError("Mask ref_type_data must be a numpy array.")

            if ref_type_data.dtype != np.uint32:
                raise ValueError("Mask ref_type_data must be of type np.uint32. Current type is:", type(ref_type_data))

            if not ref_type_data.shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT):
                raise ValueError(f"Mask ref_type_data must have shape ({DETECTOR_PIXEL_X_COUNT}, {DETECTOR_PIXEL_Y_COUNT}). Current shape is: {ref_type_data.shape}")

        if ref_type_data is None and self.super_rate_image is None and self.superdark is None:
            raise ValueError("Mask requires user to supply either ref_type_data, superdark, or super rate image.")

        else:
            logging.info("The input 2D data array is now self.mask_image.")
            self.mask_image = ref_type_data
            logging.info("Ready to generate reference file.")


    def make_mask_image(self,
                        boxwidth=4,
                        dead_sigma=5.,
                        max_low_qe_signal=0.5,
                        min_open_adj_signal=1.05,
                        do_not_use_flags=["DEAD", "TELEGRAPH", "OTHER_BAD_PIXEL", "RESERVED_7"],
                        from_smoothed=False,
                        sigma_thresh_jump=5.0,
                        min_jumps_for_rc_telegraph=1,
                        noise_sigma=None,
                        rc_thresh_ratio_tiebreaker=0.1,
                        tg_thresh_ratio_tiebreaker=1.5,
                        min_per_level=10):
        """
        This method contains the full mask-generation workflow:
            1. Identify flat-field based defects (DEAD, LOW_QE, OPEN/ADJ).
            2. Detect statistically significant jumps in each pixel ramp.
            3. Classify jumpy pixels as RC or TELEGRAPH using model-based diagnostics.
            4. Apply reference pixel and DO_NOT_USE flags.

        See docstring for each step for more information on the flag algorithms.

        NOTE: This method is intended to be the module's internal pipeline where each method's internal
        variables and parameters are set and this is the single call to populate all attributes needed
        for the reference file data model.

        Parameters:
        -----------
        boxwidth : int, optional
            Width of the boxcar smoothing kernel used when generating the
            locally-normalized flat-field image. Only used when
            from_smoothed=True. Default is 4.

        dead_sigma : float, optional
            Sigma threshold below the mean of the normalized flat-field image
            at which a pixel is classified as DEAD. Default is 5.

        max_low_qe_signal : float, optional
            Maximum normalized signal value for a pixel to be considered
            LOW_QE or OPEN. Default is 0.5.

        min_open_adj_signal : float, optional
            Minimum normalized signal value required for all adjacent pixels
            when identifying OPEN and ADJ_OPEN pixels. Default is 1.05.

        do_not_use_flags : list of str, optional
            List of DQ flag names whose pixels should also be marked as
            DO_NOT_USE. This prevents downstream romancal steps from using
            known bad pixels. Default is ["DEAD", "TELEGRAPH", "OTHER_BAD_PIXEL", "RESERVED_7" (RC/IRC)].

        from_smoothed : bool, optional
            If True, generate normalized flat-field images using local
            (smoothed) normalization. If False, use global mean normalization.
            Default is False.

        sigma_thresh_jump : float, optional
            Threshold (in robust sigma units) used by the MAD-based jump
            detector to identify statistically significant read-to-read
            discontinuities in pixel ramps. Default is 5.0.

        min_jumps_for_rc_telegraph : int, optional
            Minimum number of detected jumps in a pixel ramp required for the
            pixel to be considered a candidate for RC or TELEGRAPH
            classification. Default is 1.

        noise_sigma : float or None, optional
            Effective per-read noise estimate used for chi-square and
            level-separation SNR calculations. If None, the noise is estimated
            empirically from the residuals of the best-fit RC model for each
            pixel.

        rc_thresh_ratio_tiebreaker : float, optional
            Lower threshold on the chi-square ratio (chi2_rc / chi2_tg) used to
            resolve ties in the voting classifier in favor of RC behavior.
            Default is 0.1.

        tg_thresh_ratio_tiebreaker : float, optional
            Upper threshold on the chi-square ratio (chi2_rc / chi2_tg) used to
            resolve ties in the voting classifier in favor of TELEGRAPH
            behavior. Default is 1.5.

        min_per_level : int, optional
            Minimum number of samples required in each level when estimating
            the low and high states of the two-level telegraph model. This
            prevents single outliers or smooth RC ramps from producing
            artificially large level separations. Default is 10.
        """
        if self.super_rate_image is not None:
            logging.info("Running update_mask_from_flats()")
            self.update_mask_from_flats(from_smoothed=from_smoothed,
                                        boxwidth=boxwidth,
                                        dead_sigma=dead_sigma,
                                        max_low_qe_signal=max_low_qe_signal,
                                        min_open_adj_signal=min_open_adj_signal)
        if self.superdark is not None:
            logging.info("Running update_mask_from_darks()")
            self.update_mask_from_darks(sigma_thresh_jump=sigma_thresh_jump,
                                        min_jumps_for_rc_telegraph=min_jumps_for_rc_telegraph,
                                        noise_sigma=noise_sigma,
                                        rc_thresh_ratio_tiebreaker=rc_thresh_ratio_tiebreaker,
                                        tg_thresh_ratio_tiebreaker=tg_thresh_ratio_tiebreaker,
                                        min_per_level=min_per_level)

        # These functions can be implemented without input files
        logging.info("Setting REFERENCE pixels")
        self.update_mask_ref_pixels()

        logging.info("Setting DO_NOT_USE pixels")
        self.set_do_not_use_pixels(do_not_use_flags=do_not_use_flags)

        # Updating the Mask object with calculated mask
        self.mask_image = self.dq_mask


    def update_mask_from_flats(self,
                               from_smoothed,
                               boxwidth,
                               dead_sigma,
                               max_low_qe_signal,
                               min_open_adj_signal):
        """
        This function is used when ID'ing bad pixels from a super_rate_image.
        The following bad pixels classes are identified:
            - DEAD: set_dead_pixels()
            - LOW_QE: set_low_qe_pixels()
            - OPEN and ADJ: set_open_adj_pixels()

        The super_rate_image is generated in the MaskPipeline class from flat files.
        Since flat images are evenly illuminated pixels across the entire detector, they are ideal for
        identifying low sensitivity pixels (such as DEAD).
        """
        logging.info("Creating normalized image with super_rate_image")
        self.normalized_super_rate = self.create_normalized_image(from_smoothed,
                                                                  boxwidth)

        logging.info("Identifying DEAD pixels")
        self.set_dead_pixels(dead_sigma)

        logging.info("Identifying LOW_QE and OPEN/ADJ pixels")
        self.set_low_qe_open_adj_pixels(max_low_qe_signal,
                                        min_open_adj_signal)

        logging.info("All DEAD, OPEN/ADJ, LOW_QE pixels identified!")


    def create_normalized_image(self, from_smoothed, boxwidth):
        """
        Use the super_rate_image to create a normalized super rate image.
        If `from_smoothed` is True, then the super_rate_image will be 
        smoothed with a 2D box kernel before being normalized.

        Parameters
        ----------
        from_smoothed : bool
            If True, then smooth the super_rate_image before normalizing.
        boxwidth : int
            Used as the 2D box kernel width if `from_smoothed` is True.

        Returns
        -------
        Returns the normalized super_rate_image.
        """
        if from_smoothed:

            smoothing_kernel = Box2DKernel(boxwidth)
            smoothed_image = convolve(self.super_rate_image,
                                      smoothing_kernel,
                                      boundary="fill",
                                      fill_value=np.nanmedian(self.super_rate_image),
                                      nan_treatment="interpolate")

            denominator = smoothed_image

        else:
            denominator = np.nanmean(self.super_rate_image)

        return self.super_rate_image / denominator


    def set_dead_pixels(self, dead_sigma):
        """
        Identify the DEAD pixels using the normalized super rate image.
        A pixel is considered DEAD if it is 5 sigma below the mean
        of the normalized image.
        """
        norm_mean = np.nanmean(self.normalized_super_rate)
        norm_std = np.nanstd(self.normalized_super_rate)

        threshold = norm_mean - (dead_sigma * norm_std)

        logging.info(f"Pixels with normalized countrate value < {threshold} are marked as DEAD")

        dead_mask = (self.normalized_super_rate < threshold).astype(np.uint32)
        dead_mask[dead_mask == 1] = dqflags.DEAD.value

        self.dq_mask += dead_mask


    def _get_adjacent_pix(self, x_coor, y_coor, im):
        """
        Identify the pixels adjacent to a given pixel. Copied from Webb's RFP.
        This is used in set_low_qe_open_adj() function.

        Ex: note that x are the returned coordinates.
        [ ][x][ ]
        [x][o][x]
        [ ][x][ ]
        TODO: should we modify this function to return the corners too?
              Also, Tim brought up the case of two adjacent open pixels,
              currently if two open pixels are adjacent to each other then
              they're marked as LOW_QE since all four corners must be >1.05 norm im value.
        """
        y_dim, x_dim = im.shape

        if ((x_coor > 0) and (x_coor < (x_dim-1))):

            if ((y_coor > 0) and (y_coor < y_dim-1)):
                adj_x = np.array([x_coor, x_coor+1, x_coor, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor, y_coor-1, y_coor])

            elif y_coor == 0:
                adj_x = np.array([x_coor, x_coor+1, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor, y_coor])

            elif y_coor == (y_dim-1):
                adj_x = np.array([x_coor+1, x_coor, x_coor-1])
                adj_y = np.array([y_coor, y_coor-1, y_coor])

        elif x_coor == 0:

            if ((y_coor > 0) and (y_coor < y_dim-1)):
                adj_x = np.array([x_coor, x_coor+1, x_coor])
                adj_y = np.array([y_coor+1, y_coor, y_coor-1])

            elif y_coor == 0:
                adj_x = np.array([x_coor, x_coor+1])
                adj_y = np.array([y_coor+1, y_coor])

            elif y_coor == (y_dim-1):
                adj_x = np.array([x_coor+1, x_coor])
                adj_y = np.array([y_coor, y_coor-1])

        elif x_coor == (x_dim-1):

            if ((y_coor > 0) and (y_coor < y_dim-1)):

                adj_x = np.array([x_coor, x_coor, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor-1, y_coor])

            elif y_coor == 0:

                adj_x = np.array([x_coor, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor])

            elif y_coor == (y_dim-1):

                adj_x = np.array([x_coor, x_coor-1])
                adj_y = np.array([y_coor-1, y_coor])

        return adj_y, adj_x


    def set_low_qe_open_adj_pixels(self, max_low_qe_signal, min_open_adj_signal):
        """
        Identify LOW_QE, OPEN and ADJ pixels using the normalized super rate image.
        First, a list of coordinates of low signal pixels (defined as having a normalized
        rate value less than max_low_qe_signal) is created. The code then iterates through
        each of these low signal pixels, getting the four adject pixels and seeing
        if ALL of these four pixels are >1.05 norm im. If so, then this is a OPEN/ADJ
        pixel. Otherwise, then just the center is marked as LOW_QE.
        """
        low_qe_map = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)
        open_map = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)
        adj_map = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)

        low_sig_y, low_sig_x = np.where(self.normalized_super_rate < max_low_qe_signal)

        logging.info("Looping through low signal pixels to identify OPEN/ADJ/LOW_QE pixels")
        for x, y in zip(low_sig_x, low_sig_y):

            # Skip calculations if this is a DEAD pixel
            if self.dq_mask[y, x] & dqflags.DEAD.value == dqflags.DEAD.value:
                continue

            adj_coor = self._get_adjacent_pix(
                x_coor=x,
                y_coor=y,
                im=self.normalized_super_rate
            )

            adj_pix = self.normalized_super_rate[adj_coor]
            all_adj = (adj_pix > min_open_adj_signal)

            # TODO: update with OPEN/ADJ flags when determined
            if all(all_adj):
                adj_map[y-1:y+2, x-1:x+2] = dqflags.RESERVED_5.value
                adj_map[y, x] = 0
                open_map[y, x] = dqflags.RESERVED_6.value

            else:
                low_qe_map[y, x] = dqflags.LOW_QE.value

        self.dq_mask += low_qe_map.astype(np.uint32)
        self.dq_mask += open_map.astype(np.uint32)
        self.dq_mask += adj_map.astype(np.uint32)


    def set_do_not_use_pixels(self, do_not_use_flags):
        """
        This function adds the DO_NOT_USE flag to pixels with flags:
            DEAD
        DO_NOT_USE pixels are excluded in subsequent pipeline processing.
        More flags may be added after further analyses.
        """
        dnupix_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT),
                               dtype=np.uint32)

        # Going through each DNU flag
        for flag in do_not_use_flags:

            logging.info(f"Setting {flag} pixels as DO_NOT_USE")

            # Bitval for the current flag
            bitval = dqflags[flag].value

            # The indices of pixels with the current iteration's flag
            flagged_pix = np.where((self.dq_mask & bitval) == bitval)

            # Setting flagged pix to DNU bitval
            dnupix_mask[flagged_pix] = dqflags.DO_NOT_USE.value

        # Adding to mask
        self.dq_mask += dnupix_mask.astype(np.uint32)

        return


    def update_mask_from_darks(self, sigma_thresh_jump, min_jumps_for_rc_telegraph, noise_sigma, rc_thresh_ratio_tiebreaker, tg_thresh_ratio_tiebreaker, min_per_level):
        """
        This function is used when identifying bad pixels from long dark files (350 reads).
        The following bad pixel classes are identified:
            - RC and TELEGRAPH: set_rc_tel_pixels()

        The filelist is a list of IRRC-corrected dark files, and a superdark must be created.
        Since both RC and telegraph pixels have anomalous behavior as they accumulate
        charge, a superdark is ideal for detecting weird ramp shapes.

        Notes
        -----
        Classification relies on ramp morphology rather than
        absolute signal level. Pixels are first screened using a robust
        MAD-based jump detector; only pixels with detected jumps
        are subjected to RC/telegraph classification.
        """
        logging.info("Computing MAD-based jump counts for all pixels")
        self.mad_based_jump_counter_cube(sigma_thresh_jump=sigma_thresh_jump)

        logging.info("Identifying TELEGRAPH and RC pixels")
        self.set_rc_tel_pixels(min_jumps_for_rc_telegraph=min_jumps_for_rc_telegraph, 
                               noise_sigma=noise_sigma,
                               rc_thresh_ratio_tiebreaker=rc_thresh_ratio_tiebreaker,
                               tg_thresh_ratio_tiebreaker=tg_thresh_ratio_tiebreaker,
                               min_per_level=min_per_level)


    def set_rc_tel_pixels(self, min_jumps_for_rc_telegraph, noise_sigma, rc_thresh_ratio_tiebreaker, tg_thresh_ratio_tiebreaker, min_per_level):
        """
        Classify pixels exhibiting jump behavior as RC (double exponential behavior)
        or telegraph (multi-level switching) using per-pixel ramp diagnostics.

        For pixels that show at least one jump, we try two competing physical models for the ramp:
        an RC-like exponential decay and a telegraph-like two-level signal.
        We then compare how well each model explains the ramp.
        Statistics computed from the residuals are used as evidence of structure in the data
        and are not treated as formal hypothesis tests.

        Workflow
        --------
        For each pixel with jump_count >= min_jumps_for_rc_telegraph:

            1. RC Model Fit
                - Attempt a double-exponential fit to capture RC behavior.
                  If the fit fails to converge, a single-exponential fallback is attempted.
                - Compute the reduced chi-square of the RC model.
                - Compute the Anderson-Darling statistic and the bimodality coefficient
                  of the residual differences. RC-like ramps show low AD and BC values
                  because residuals are approximately Gaussian and unimodal.

            2. Telegraph (Two-Level) Model Fit
                - Fit a two-level model by splitting the ramp about its median value.
                - Compute the reduced chi-square of the telegraph model.
                - Compute the level-separation SNR, defined as |H - L| / noise_sigma,
                  where H and L are the high and low states.
                  Small level-SNR strongly favors telegraph behavior; very large values
                  tend to occur for RC-like ramps whose exponential span covers a wide DN range.
                - Compute the median jump amplitude using the absolute DN differences
                  at jump locations.

        Classification Criteria
        -----------------------
        A pixel is labeled RC when:
            - The telegraph (two-level) model provides a poor fit.
            - Residuals of the RC fit lack Gaussian structure (low AD statistic and low BC).
            - Jump amplitudes are small.
            - The RC model outperforms the telegraph model (chi-square comparison).

        A pixel is labeled TELEGRAPH when:
            - The two-level model has very low reduced chi-square.
            - Residuals of the RC fit are non-Gaussian or multi-modal (high AD and BC).
            - The pixel exhibits a large jump amplitude.
            - The telegraph model provides a better fit than the RC model.

        Inputs
        ------
        superdark : ndarray
            The (nreads, ny, nx) superdark cube
        jump_count : ndarray
            Integer (ny, nx) array giving the number of MAD-based jumps per pixel

        Returns
        -------
        rc_mask : ndarray (bool)
            Mask of pixels classified as RC.
        telegraph_mask : ndarray (bool)
            Mask of pixels classified as telegraph.
        metrics_table : pandas.DataFrame
            Per-pixel diagnostic metrics used during classification
            (chi-square values, AD/BC residual statistics, level SNR,
            jump amplitudes, slopes, etc.).

        Notes
        -----
        - Pixels with zero jumps are excluded from this classification routine.
        - Even though by definition telegraphs can have more than two-levels of switching,
          the two-level model that is applied will approximate > 2 level switching more
          accurately than a double exponential model.
        - All thresholds are empirically chosen to maximize separation
          between known RC and telegraph populations.
        - Ambiguous cases are conservatively flagged as OTHER_BAD_PIXEL.
        """
        # Empty masks to be populated with the positions of the RC and TELEGRAPH pixels, and OTHER
        rc_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)
        telegraph_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)
        other_bad_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)

        cand_y, cand_x = np.where(self.jump_count_img >= min_jumps_for_rc_telegraph)
        logging.info(f"Found {cand_x.size} candidates for 'jumpy' pixels with >= {min_jumps_for_rc_telegraph} jumps")

        # Create coord list to run in parallel
        cand_coords = list(zip(cand_y.tolist(), cand_x.tolist()))

        # Having this function within set_rc_tel_pixels allows us to not have to pass in superdark; saves mem
        # TODO brad probably knows a better way to save memory :)
        def _process_pixel_rc_tel(coord):
            """
            For a single pixel, extract the ramp and jumps in ramp. Then,
            create a dictionary called metrics which has the chi2s, residual stats,
            jump stats, and two-level model metrics used to classify pixel as RC or TELEGRAPH.
            """
            y, x = coord
            nreads = self.superdark.shape[0]
            t = np.arange(1, nreads + 1) * 3.16247 # seconds
            ramp = self.superdark[:, y, x]

            jump_mask_ramp = self.jump_mask_cube[:, y, x]
            jump_idx = np.where(jump_mask_ramp)[0] + 1
            jump_count_pix = int(jump_mask_ramp.sum())

            metrics = self.compute_metrics_for_pixel_rc_tel(ramp, t, jump_idx, jump_count_pix, noise_sigma, min_per_level)

            metrics["y"] = y
            metrics["x"] = x

            label_new, rc_votes, tg_votes = self.classify_rc_vs_tele(metrics, rc_thresh_ratio_tiebreaker, tg_thresh_ratio_tiebreaker)

            metrics["label_new"] = label_new
            metrics["rc_votes"] = rc_votes
            metrics["tg_votes"] = tg_votes

            return metrics

        logging.info("Beginning pixel processing")
        with ThreadPoolExecutor(max_workers=16) as ex:
            rows = list(ex.map(_process_pixel_rc_tel, cand_coords))

        logging.info("Updating dict rows with bad pixel label")
        for row in rows:
            x, y = row["x"], row["y"]
            label = row["label_new"]

            # TODO: need specific flag for RC/IRC
            if label == "RC_new":
                rc_mask[y, x] = dqflags.RESERVED_7.value

            elif label == "Tele_new":
                telegraph_mask[y, x] = dqflags.TELEGRAPH.value

            elif label == "Ambig" or label == "UNKNOWN":
                other_bad_mask[y, x] = dqflags.OTHER_BAD_PIXEL.value

        # Updating the full mask
        self.dq_mask += rc_mask
        self.dq_mask += telegraph_mask
        self.dq_mask += other_bad_mask

        self.metrics_df = pd.DataFrame(rows)


    def classify_rc_vs_tele(self, metrics_row, rc_thresh_ratio_tiebreaker, tg_thresh_ratio_tiebreaker):
        """
        Voting-based classifier using multiple diagnostics.
        No single metric is decisive; agreement across metrics determines
        the final classification, with chi-square ratios used as a tie-breaker.

        Returns:
            label_new : "RC_new", "Tele_new", "Ambig", or "UNKNOWN"
            rc_votes  : int
            tg_votes  : int
        """
        chi2_tg = metrics_row.get("chi2_tg", np.nan)
        chi2_rc = metrics_row.get("chi2_rc", np.nan)
        ad_dr = metrics_row.get("ad_dr", np.nan)
        level_snr = metrics_row.get("level_snr", np.nan)
        jump_med = metrics_row.get("jump_amp_med", np.nan)

        # Put chi2 ratio in log10 space since the ratios differ on order of magnitudes
        if np.isfinite(chi2_tg) and chi2_tg > 0 and np.isfinite(chi2_rc):
            chi2_ratio = chi2_rc / (chi2_tg + 1e-6)
            log10_ratio = np.log10(chi2_ratio)
        else:
            log10_ratio = np.nan

        vals = [chi2_tg, chi2_rc, ad_dr, level_snr, jump_med, log10_ratio]

        # Catching pixels that would be marked as UNKNOWN due to NaNs using chi2 ratios
        if any(not np.isfinite(v) for v in vals):
            if np.isfinite(chi2_rc) and np.isfinite(chi2_tg) and chi2_tg > 0:
                ratio = chi2_rc / chi2_tg

                if ratio <= rc_thresh_ratio_tiebreaker:
                    return "RC_new", 0, 0
                elif ratio >= tg_thresh_ratio_tiebreaker:
                    return "Tele_new", 0, 0
                else:
                    return "Ambig", 0, 0

            # Truly unknown
            return "UNKNOWN", 0, 0

        # Beginning voting
        rc_votes = 0
        tg_votes = 0

        # Votes based on chi2_tg
        if chi2_tg < 5.0:
            tg_votes += 2
        elif chi2_tg > 200.0:
            rc_votes += 2

        # Votes based on AD statistic
        if ad_dr > 10.0:
            tg_votes += 1
        elif ad_dr < 3.0:
            rc_votes += 1

        # Votes based on level_snr
        if level_snr < 3.0:
            tg_votes += 1
        elif level_snr > 10.0:
            rc_votes += 1

        # Votes based on median jump amplitude
        if jump_med > 120.0:
            tg_votes += 1
        elif jump_med < 40.0:
            rc_votes += 1

        # Votes based on chi2 ratio
        if log10_ratio > -0.5:
            tg_votes += 1
        elif log10_ratio < -1.0:
            rc_votes += 1

        # Evaluating the votes and assigning labels
        if tg_votes > rc_votes:
            label = "Tele_new"
        elif rc_votes > tg_votes:
            label = "RC_new"
        else:
            # Tie-breaker using chi2_rc vs chi2_tg
            if chi2_tg > 0 and chi2_rc > 0:
                ratio = chi2_rc / chi2_tg
                if ratio <= rc_thresh_ratio_tiebreaker:
                    label = "RC_new"
                elif ratio >= tg_thresh_ratio_tiebreaker:
                    label = "Tele_new"
                else:
                    label = "Ambig"
            else:
                label = "Ambig"

        # Try to evaluate UNKNOWN
        if label == "UNKNOWN":
            if np.isfinite(chi2_rc) and np.isfinite(chi2_tg) and chi2_tg > 0:
                ratio = chi2_rc / chi2_tg

                if ratio <= rc_thresh_ratio_tiebreaker:
                    label = "RC_new"
                elif ratio >= tg_thresh_ratio_tiebreaker:
                    label = "Tele_new"
                else:
                    label = "Ambig"

        return label, rc_votes, tg_votes


    def compute_metrics_for_pixel_rc_tel(self, ramp, t, jump_idx, jump_count_pix, noise_sigma, min_per_level):
        """
        Compute per-pixel diagnostics used to classify RC vs TELEGRAPH behavior.

        This function fits both an RC-like exponential model and a simple two-level
        telegraph model to a single pixel ramp, then computes residual-based and
        model-comparison metrics used by the voting classifier.

        Noise Handling
        --------------
        noise_sigma represents the effective per-read noise within a stable
        ramp segment. If not given, it is estimated from the residuals of the 
        best-fit RC model.

        Small numerical floors are added to denominators to prevent division-by-zero
        and NaN propagation.
        """
        try:
            # Try an RC (double exp.) fit on the ramp
            ramp_fit_rc, n_params = self.fit_rc_safe(t, ramp)

        # Every fit failed... returning metrics full of NaNs
        except Exception:
            return dict(
                    chi2_rc=np.nan,
                    ad_dr=np.nan,
                    bc_dr=np.nan,
                    frac_flat=np.nan,
                    mean_diff=np.nan,
                    chi2_tg=np.nan,
                    jump_count=jump_count_pix,
                    level_snr=np.nan,
                    jump_amp_med=np.nan,
                )

        resid = ramp - ramp_fit_rc

        # Noise estimate from RC fit residuals if None
        if noise_sigma is None:
            noise_sigma = np.std(resid)

        # Add 1e-12 to avoid dvision by zero below
        sigma2 = noise_sigma**2 + 1e-12

        # Compute reduced chi2 of D.E (RC) model
        dof_rc = max(len(ramp) - n_params, 1)
        chi2_rc = np.sum(resid**2 / sigma2) / dof_rc

        # Extract the residual differences of the ramp
        dr = np.diff(resid)

        # Compute Anderson-Darling metric
        ad_stat, _, _ = anderson(dr, dist="norm")

        # Compute bimodality coefficient metric
        bc = self.bimodality_coefficient(dr)

        # Calculate the segment slopes
        slopes, slope_sigmas, seg_means = self.segment_slopes(t, ramp, jump_idx)

        if slopes.size > 0:
            frac_flat = float(np.mean(slope_sigmas < 1.0))
            mean_diff = float(np.max(seg_means) - min(seg_means))

        else:
            frac_flat = np.nan

        # Fit the telegraph two-level model
        ramp_tg, low, high = self.simple_two_level_model(ramp, min_per_level)

        # Calculating red. chi2 for the two level model
        dof_tg = max(len(ramp) - 2, 1)
        chi2_tg = np.sum((ramp - ramp_tg)**2 / sigma2) / dof_tg

        # 4) Compute the level separation SNR
        level_snr = np.abs(high - low) / (noise_sigma + 1e-12)

        # 5) Calculate the jump amplitude
        ramp_diffs = np.diff(ramp)

        if jump_idx.size > 0:
            jump_amp_med = float(np.median(np.abs(ramp_diffs[jump_idx - 1])))

        else:
            jump_amp_med = np.nan

        # 6) Linear slope estimate for non-jumpy pixels (consistent with slope_map)
        t0 = t - t.mean()
        try:
            lr = linregress(t0, ramp)
            slope_linear = lr.slope

        except Exception:
            slope_linear = np.nan

        return dict(
            chi2_rc=chi2_rc,
            ad_dr=ad_stat,
            bc_dr=bc,
            frac_flat=frac_flat,
            mean_diff=mean_diff,
            chi2_tg=chi2_tg,
            jump_count=jump_count_pix,
            level_snr=level_snr,
            jump_amp_med=jump_amp_med,
            slope_linear=slope_linear,
        )


    def simple_two_level_model(self, ramp, min_per_level):
        """
        Construct a two-level (telegraph) approximation to a ramp.

        This model is intentionally simple and is used only as a comparative
        diagnostic against RC-like fits.

        Strategy
        --------
        1. Split the ramp about its median value.
        2. Require a minimum number of samples (min_per_level) in each level
           to robustly estimate medians and prevent single outliers or smooth
           RC ramps from producing artificially large level separations.
        3. If the minimum requirement is not met, fall back to percentile-based
           levels (25th/75th).
        4. Build a two-level model using a fixed threshold between the levels.

        Note: Telegraph pixels can exhibit more than two states in their ramp,
        but the two-level model is still a much better fit for multi-state
        telegraphs when compared to the RC model fits.
        """
        # Find the split in the ramp (median)
        m = np.median(ramp)

        # The low state is all reads below the median
        low = ramp[ramp <= m]

        # The high state is all reads above the median
        high = ramp[ramp > m]

        # Getting the median of the low and high ramp levels
        if low.size >= min_per_level and high.size >= min_per_level:
            low_med = np.median(low)
            high_med = np.median(high)

        else:
            # If num reads per level isn't met, then look at percentiles
            p25 = np.percentile(ramp, 25)
            p75 = np.percentile(ramp, 75)

            low_med, high_med = p25, p75

        # If levels are still too close (bad telegraph fit), try a looser model
        if np.isclose(low_med, high_med, rtol=0, atol=1e-6):
            low_med = np.percentile(ramp, 10)
            high_med = np.percentile(ramp, 90)

        # Failed telegraph fit, ramp is essentially flat
        if np.isclose(low_med, high_med, rtol=0, atol=1e-6):
            low_med = high_med = np.median(ramp)

        # Build two-level model
        thr = 0.5 * (low_med + high_med)
        ramp_tg = np.where(ramp <= thr, low_med, high_med)

        # Return the telegraph fit and the L/H medians
        return ramp_tg, low_med, high_med

    def segment_slopes(self, t, ramp, jump_idx):
        """
        Get the slopes of the level segments.
        Used for telegraph identification since segments' slopes
        should be flat, whereas RC should have slope up the ramp.
        If slope_sigmas >> 1, then significantly sloped (RC-leaning)
        If slope_sigmas << 1, then essentially flat (telegraph-leaning)
        """
        jump_idx = np.asarray(jump_idx, int)
        idxs = np.concatenate(([0], jump_idx, [len(t)]))

        slopes = []
        slope_sigmas = []
        seg_means = []

        for i in range(len(idxs)-1):
            low, hi = idxs[i], idxs[i+1]

            # Skipping tiny segments
            if hi - low < 4:
                continue

            lr = linregress(t[low:hi], ramp[low:hi])

            slopes.append(lr.slope)
            slope_sigmas.append(abs(lr.slope) / (lr.stderr + 1e-8))
            seg_means.append(np.mean(ramp[low:hi]))

        return np.array(slopes), np.array(slope_sigmas), np.array(seg_means)

    def bimodality_coefficient(self, dr):
        """
        Calculate the bimodality coefficient for the residuals of ramp.
        """
        if dr.size < 5:
            return np.nan

        g1 = skew(dr, bias=False)
        g2 = kurtosis(dr, fisher=False, bias=False)
        n = dr.size

        return (g1**2 + 1) / (g2 + 3 * ((n-1)**2) / ((n-2)*(n-3)))

    def fit_rc_safe(self, t, y):
        """
        Stable RC-like fit with fallbacks:
        1) Double exponential in log-tau space (first attempt)
        2) Single exponential (fallback)
        3) Linear model (last resort, these are likely telegraph)

        Always returns (model_values, n_params).
        Only returns (None, None) if everything explodes.
        Note: tau is in log space to prevent curve_fit from labelling
        linear ramps as RC if choosing a very large time constant
        """
        # Initial guesses for double exp
        a1_0 = float(max(y[0] - y[-1], 1.0))
        a2_0 = 0.5 * a1_0

        t_quarter = max(t[len(t)//4], 1e-3)
        t_half = max(t[len(t)//2], 1e-3)

        p0 = [a1_0, np.log(t_quarter), a2_0, np.log(t_half), float(y[-1])]

        lower = [-1e6, np.log(0.5), -1e6, np.log(0.5), np.min(y) - 500]
        upper = [1e6, np.log(1e6), 1e6, np.log(1e6), np.max(y) + 500]

        # 1) Try a double exponential fit
        def rc_model_transformed(t, a1, log_tau1, a2, log_tau2, c):
            tau1 = np.exp(log_tau1)
            tau2 = np.exp(log_tau2)

            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c

        try:
            # Fitting D.E.
            popt, _ = curve_fit(
                rc_model_transformed,
                t, y,
                p0=p0,
                bounds=(lower, upper),
                method="trf",
                maxfev=600,
            )
            y_rc = rc_model_transformed(t, *popt)

            # Return the model D.E. and number of parameters
            return y_rc, len(popt)

        except Exception:
            pass

        # 2) Try a single exponential fit if double exponential fit fails
        def single_exp(t, a, tau, c):
            return a * np.exp(-t / tau) + c

        try:
            # Initial guesses for single exponential model
            a0 = float(max(y[0] - y[-1], 1.0))
            tau0 = max(t[len(t)//3], 1.0)
            c0 = float(y[-1])

            p0_single = [a0, tau0, c0]
            lower_s = [-1e6, 1.0, np.min(y) - 500]
            upper_s = [1e6, 1e6, np.max(y) + 500]

            # Fitting single exp.
            popt_s, _ = curve_fit(
                single_exp,
                t, y,
                p0=p0_single,
                bounds=(lower_s, upper_s),
                method="trf",
                maxfev=600,
            )
            y_rc = single_exp(t, *popt_s)

            # Return the model single exp. and number of parameters
            return y_rc, len(popt_s)

        except Exception:
            pass

        # 3) If all else fails, try a linear fit (these are likely telegraph)
        try:
            lr = linregress(t, y)
            y_rc = lr.intercept + lr.slope * t

            # Return linear fit and number of parameters
            return y_rc, 2

        except Exception:

            # Everything failed... return None for fit and len(params)
            return None, None


    def mad_based_jump_counter_cube(self, sigma_thresh_jump, eps=1e-8):
        """
        Compute a robust, MAD-based jump mask and per-pixel jump count
        for a full ramp superdark cube.

        This routine identifies statistically significant discontinuities
        ("jumps") between successive reads of a ramp using the Median Absolute
        Deviation (MAD) as a robust estimator of the underlying noise.
        Jumps may arise from telegraph pixels, or the beginning of an RC pixel's ramp.

        Two attributes are set; `jump_mask_cube` is a boolean array of shape (nreads-1, ny, nx)
        with detected jumps flagged at each time step. `jump_count_img` is an array of shape
        (ny, nx) with the number of detected jumps per pixel ramp. These attributes are used
        in `set_rc_tel_pixels` to identify Telegraph / RC+IRC pixels.

        Parameters
        ----------
        sigma_thresh_jump : float
            Threshold in robust sigma units for detecting jumps.
        eps : float, optional
            Epsilon; the minimum sigma value to prevent division by zero.

        Returns
        -------
        jump_count : ndarray
            (ny, nx) array with the number of detected jumps per pixel.
        jump_mask : ndarray
            Boolean array (nreads-1, ny, nx) flagging jumps at each time step.

        Notes
        -----
        The algorithm:
        1. Compute read-to-read differences.
        2. Estimate per-pixel median and MAD of differences.
        3. Convert MAD to sigma (MAD/0.6745) and apply a minimum floor eps.
        4. Flag differences exceeding sigma_thresh x sigma.

        This method is robust to non-Gaussian outliers and effective for
        identifying jump behavior in long superdark integrations. Jump detection
        is used only as a screening step; classification is performed using full-ramp diagnostics.
        """
        logging.info("Creating jump products")
        diffs = np.diff(self.superdark, axis=0)
        med = np.median(diffs, axis=0)
        mad = np.median(np.abs(diffs - med), axis=0)
        sigma = np.maximum(mad / 0.6745, eps)

        # Broadcast med/sigma across reads for per-read residual evaluation
        resid = np.abs(diffs - med[None, :, :])
        thr = sigma_thresh_jump * sigma[None, :, :]
        self.jump_mask_cube = resid > thr
        self.jump_count_img = np.count_nonzero(self.jump_mask_cube, axis=0).astype(np.int16)


    def update_mask_ref_pixels(self):
        """
        Create array to flag the 4 px reference pixel border around detector.
        The reference pixels are static and need no algorithm for identification.
        """
        refpix_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT),
                               dtype=np.uint32)

        refpix_mask[:4, :] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[-4:, :] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[:, :4] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[:, -4:] = dqflags.REFERENCE_PIXEL.value

        self.dq_mask += refpix_mask


    def calculate_error(self):
        """
        Abstract method not applicable to Mask.
        """
        pass


    def update_data_quality_array(self):
        """
        Abstract method not utilized by Mask().

        NOTE - Would be redundant to make_mask_image(). The attribute mask is reserved
        specifically setting the data quality arrays of other reference file types.
        """
        pass


    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """
        # Construct the mask object from the data model.
        mask_datamodel_tree = MaskRefModel()
        mask_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        mask_datamodel_tree['dq'] = self.mask_image

        return mask_datamodel_tree
