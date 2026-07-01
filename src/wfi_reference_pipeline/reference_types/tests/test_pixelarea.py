import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_PIXELAREA,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.pixel_area.pixel_area import (
    PixelArea,
)
from wfi_reference_pipeline.resources.make_test_meta import (
    MakeTestMeta,
)


@pytest.fixture
def valid_meta_data():
    """
    Fixture for generating valid WFIMetaPixelArea metadata.
    """
    test_meta = MakeTestMeta(ref_type=REF_TYPE_PIXELAREA)
    return test_meta.meta_pixelarea


@pytest.fixture
def valid_ref_type_data_array():
    """
    Fixture for generating valid pixel area map data.
    """
    return np.random.random(
        (
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )
    ).astype(np.float32)


@pytest.fixture
def pixelarea_object_with_data_array(
    valid_meta_data,
    valid_ref_type_data_array,
):
    """
    Fixture for initializing a PixelArea object with valid data.
    """
    pixelarea_object = PixelArea(
        meta_data=valid_meta_data,
        ref_type_data=valid_ref_type_data_array,
    )

    yield pixelarea_object


class TestPixelArea:

    def test_pixelarea_instantiation_with_valid_ref_type_data(
        self,
        pixelarea_object_with_data_array,
    ):
        """
        Test that PixelArea object is created successfully
        with valid input data.
        """

        assert isinstance(
            pixelarea_object_with_data_array,
            PixelArea,
        )

        assert (
            pixelarea_object_with_data_array.pixel_area.shape
            ==
            (
                DETECTOR_PIXEL_X_COUNT,
                DETECTOR_PIXEL_Y_COUNT,
            )
        )

    def test_pixelarea_instantiation_with_invalid_metadata(
        self,
        valid_ref_type_data_array,
    ):
        """
        Test that PixelArea raises TypeError
        with invalid metadata type.
        """

        bad_test_meta = MakeTestMeta(
            ref_type=REF_TYPE_READNOISE
        )

        with pytest.raises(TypeError):
            PixelArea(
                meta_data=bad_test_meta.meta_readnoise,
                ref_type_data=valid_ref_type_data_array,
            )

    def test_pixelarea_instantiation_with_invalid_ref_type_data(
        self,
        valid_meta_data,
    ):
        """
        Test that PixelArea raises TypeError
        with invalid reference type data.
        """

        with pytest.raises(TypeError):
            PixelArea(
                meta_data=valid_meta_data,
                ref_type_data="invalid_ref_data",
            )

    def test_populate_datamodel_tree(self,
                                     pixelarea_object_with_data_array,
                                     ):
        """
        Test that the PixelArea datamodel tree
        is correctly populated.
        """

        data_model_tree = (pixelarea_object_with_data_array.populate_datamodel_tree()
        )

        assert "meta" in data_model_tree
        assert "data" in data_model_tree

        assert (
            data_model_tree["data"].shape
            ==
            (
                DETECTOR_PIXEL_X_COUNT,
                DETECTOR_PIXEL_Y_COUNT,
            )
        )

        assert (
            data_model_tree["data"].dtype
            ==
            np.float32
        )

    def test_pixelarea_outfile_default(self,
                                       pixelarea_object_with_data_array,
                                       ):
        """
        Test default outfile name.
        """

        assert (
            pixelarea_object_with_data_array.outfile
            ==
            "roman_pixelarea.asdf"
        )

    def test_pixelarea_auto_generation(
            self,
            valid_meta_data,
            ):
        """
        Verify PAM generation path.
        """

        pam = PixelArea(
            meta_data=valid_meta_data,
        )

        assert pam.pixel_area is not None
        assert pam.pixel_area.dtype == np.float32
        assert pam.pixel_area.ndim == 2
        assert np.isfinite(pam.pixel_area).all()