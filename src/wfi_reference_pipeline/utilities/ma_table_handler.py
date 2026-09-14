from pathlib import Path

import asdf
import crds
from crds.client import api

from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
)


class MATableHandler:
    '''
    Utility class to handle the MA table reference file.

    The class stores the MA table reference file as an ASDF object. 
    Any class in RFP that needs it has access to it to get a specific MA table via MATableHandler.

    Usage:
        from wfi_reference_pipeline.utilities.ma_table_handler import MATableHandler
        matab_ref_file = MATableHandler()

        # ASDF object of MA table reference file
        matab_ref_file.matab

        # Get the read_pattern  and effective_exposure_time 
        read_pattern, effective_exposure_time = matab_ref_file._get_table_specific_info()       # Diagnostic table
        read_pattern, effective_exposure_time = matab_ref_file._get_table_specific_info(1010)   # IMG_135_8 
    '''

    def __init__(self):

        # Set the context and parameters to grab the MA table reference file on CRDS
        context = api.get_default_context('roman')
        parameters = {'ROMAN.META.INSTRUMENT.NAME': 'WFI',
                      'ROMAN.META.EXPOSURE.START_TIME': '2000-01-01 00:00:00'}

        print('Retrieving the MA table reference file from CRDS')
        ref = crds.getreferences(parameters, 
                                reftypes=['matable'], 
                                context=context, 
                                observatory='roman')
        matab_file = ref['matable']

        # Read the retrieved MA table reference file
        matab = asdf.open(matab_file)

        # Saving the MA tables to be used in other modules
        self.matab = matab  

        # Check if the frame times in constants.py need to be updated
        # If no change is necessary, this function will not do anything
        self._update_constants_dot_py()


    def _get_table_specific_info(self, ma_table_id=9010):
        '''
        A function to return the read pattern, and effective exposure time arrays for a requested the MA Table
        Requires a unique table number of ID. If no table ID is provided, the function returns the diagnostic table information

        Parameters
        ----------
        ma_table_id: int
            Unique MA table ID number to look up.
            IM_135_8: ID 1010
            Diagnostic: ID 9010
        '''

        read_pattern = self.matab['roman']['science_tables'][f'SCI{ma_table_id:04}']['science_read_pattern']
        effective_exposure_times = self.matab['roman']['science_tables'][f'SCI{ma_table_id:04}']['effective_exposure_time']

        return read_pattern, effective_exposure_times


    def _update_constants_dot_py(self):
        '''
        Update the frame times for both imaging and spectroscopy modes in constants.py 
        if they have been changed and are different from the latest version of the MA table values

        Use IM_135_8 (ID 1010) for the imaging mode and SP_300_16 (ID 1038) for the spectroscopy mode
        to grab the frame times for each mode
        '''

        # Latest frame times from the MA table reference file
        wim_frame_time = self.matab['roman']['science_tables']['SCI1010']['frame_time']
        wsm_frame_time = self.matab['roman']['science_tables']['SCI1038']['frame_time']

        if (WFI_FRAME_TIME['WIM'] != wim_frame_time) or (WFI_FRAME_TIME['WSM'] != wsm_frame_time):
            print('The frame times have been changed. Updating constants.py...')

            # Path to constants.py
            constants_path = str((Path(__file__).parent).resolve()) + '/../'
            constants_file = 'constants.py'

            # Open constants.py and read the script line-by-line
            with open(constants_path + '/' + constants_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()  # Returns a list of strings

            modified_lines = []
            for line in lines:
                # Find the line with the frame time and update them to the latest values from the MA table reference file
                if 'WFI_FRAME_TIME' in line:
                    line = f'WFI_FRAME_TIME = {{WFI_MODE_WIM: {wim_frame_time}, WFI_MODE_WSM: {wsm_frame_time}}}\n'
                else:
                # Keep the rest of the lines as-is
                    line = line
                modified_lines.append(line)

            with open(constants_path + '/' + constants_file, 'w') as file:
                file.writelines(modified_lines)

            print(f'Frame times have been updated to {wim_frame_time}sec for WFI_MODE_WIM and {wsm_frame_time}sec for WFI_MODE_WSM')

        else:
            pass
