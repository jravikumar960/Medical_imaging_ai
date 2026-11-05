import numpy as np

from xray_gradcam_localization import border_mask, mass_fraction, occlude_image


def test_mass_fraction_empty_cam():
    cam = np.zeros((4, 4))
    mask = np.ones((4, 4), dtype=bool)
    assert mass_fraction(cam, mask) == 0.0


def test_mass_fraction_all_inside():
    cam = np.ones((4, 4))
    mask = np.ones((4, 4), dtype=bool)
    assert mass_fraction(cam, mask) == 1.0


def test_border_mask_shape_and_margin():
    mask = border_mask((8, 8), 2)
    assert mask.shape == (8, 8)
    assert mask[0, 0]
    assert not mask[3, 3]


def test_occlude_image_sets_masked_pixels():
    from PIL import Image

    image = Image.new("L", (4, 4), color=128)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    result = occlude_image(image, mask, fill_value=0)
    assert np.asarray(result)[1, 1] == 0
    assert np.asarray(result)[0, 0] == 128
def test_mass_fraction_shape_mismatch():
    cam = np.ones((4, 4))
    mask = np.ones((3, 3), dtype=bool)
    assert mass_fraction(cam, mask) == 0.0


def test_occlude_image_rejects_shape_mismatch():
    from PIL import Image

    image = Image.new("L", (4, 4), color=128)
    mask = np.zeros((3, 3), dtype=bool)
    try:
        occlude_image(image, mask, fill_value=0)
    except ValueError:
        return
    raise AssertionError("ValueError was not raised")
