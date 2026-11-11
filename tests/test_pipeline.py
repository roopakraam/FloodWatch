"""Tests for src/preprocess.py and src/pipeline.py (synthetic data only, no dataset needed)."""
import numpy as np

from src.pipeline import augment, make_eval_dataset, make_train_dataset, steps_per_epoch
from src.preprocess import CLIP_DB, compute_channels, finite_mask, preprocess

NORM = (np.array([-10.0, -17.0, 7.0], dtype=np.float32), np.array([4.0, 4.5, 3.3], dtype=np.float32))


def test_compute_channels_clips_then_subtracts():
    img = np.array([[[-50.0, 0.0]], [[-20.0, 30.0]]], dtype=np.float32)  # (2, 1, 2): VV, VH
    vv, vh, ratio = compute_channels(img)
    assert vv.tolist() == [[-35.0, 0.0]]            # VV -50 clipped to -35
    assert vh.tolist() == [[-20.0, 5.0]]            # VH 30 clipped to 5
    assert ratio.tolist() == [[-15.0, -5.0]]        # from the CLIPPED values: -35-(-20), 0-5


def test_preprocess_standardises_and_zeroes_nan():
    img = np.full((2, 4, 4), np.nan, dtype=np.float32)
    img[:, 0, 0] = (-10.0, -17.0)  # exactly the mean of VV and VH -> 0 after standardising
    img[:, 1, 1] = (-6.0, -21.5)   # one std above / below
    x = preprocess(img, "vv_vh_ratio", NORM)
    assert x.shape == (4, 4, 3) and x.dtype == np.float32
    assert not np.isnan(x).any()
    np.testing.assert_allclose(x[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(x[1, 1], [1.0, -1.0, (15.5 - 7.0) / 3.3], rtol=1e-5)
    assert (x[2:, 2:] == 0).all()  # NaN pixels become 0 = the average training pixel


def test_channel_sets_are_consistent_subsets():
    rng = np.random.default_rng(0)
    img = rng.uniform(CLIP_DB[0] - 5, CLIP_DB[1] + 5, size=(3, 2, 8, 8)).astype(np.float32)
    full = preprocess(img, "vv_vh_ratio", NORM)
    np.testing.assert_array_equal(preprocess(img, "vh", NORM), full[..., 1:2])
    np.testing.assert_array_equal(preprocess(img, "vv_vh", NORM), full[..., 0:2])
    assert full.shape == (3, 8, 8, 3)


def test_finite_mask():
    img = np.zeros((2, 2, 2), dtype=np.float32)
    img[1, 0, 1] = np.nan  # VH missing at one pixel
    assert finite_mask(img).tolist() == [[True, False], [True, True]]


def _indexed(h=48, w=40):
    """Image and mask that both hold the pixel index, so any misalignment is visible."""
    idx = np.arange(h * w, dtype=np.float32).reshape(h, w, 1)
    return idx, idx.copy()


def test_augment_keeps_image_and_mask_aligned():
    x, y = _indexed()
    rng = np.random.default_rng(0)
    for _ in range(200):
        xa, ya = augment(x, y, rng, crop=16)
        assert xa.shape == (16, 16, 1) and ya.shape == (16, 16, 1)
        np.testing.assert_array_equal(xa, ya)  # the same crop / flip / rotation hit both


def test_augment_returns_a_square_crop_of_the_original():
    x, y = _indexed(48, 40)
    rng = np.random.default_rng(1)
    for _ in range(50):
        xa, _ = augment(x, y, rng, crop=16)
        rows, cols = np.divmod(xa[..., 0].astype(int), 40)
        assert rows.max() - rows.min() == 15 and cols.max() - cols.min() == 15
        assert len(np.unique(xa)) == 256  # no pixel duplicated or lost


def test_augment_uses_all_eight_orientations():
    x, y = _indexed(16, 16)
    rng = np.random.default_rng(2)
    seen = {augment(x, y, rng, crop=16)[0].tobytes() for _ in range(400)}
    assert len(seen) == 8  # 4 rotations x mirror = the 8 symmetries of a square


def test_augment_is_reproducible():
    x, y = _indexed()
    a = [augment(x, y, np.random.default_rng(7), crop=16)[0] for _ in range(2)]
    np.testing.assert_array_equal(a[0], a[1])


def test_train_dataset_shapes_and_epoch_length():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 64, 64, 3)).astype(np.float32)
    Y = rng.integers(-1, 2, size=(6, 64, 64, 1)).astype(np.int8)
    ds = make_train_dataset(X, Y, batch=8, crops_per_chip=4, crop=32, seed=0)
    batches = list(ds)
    assert len(batches) == steps_per_epoch(6, 8, 4) == 3
    for xb, yb in batches:
        assert xb.shape == (8, 32, 32, 3) and yb.shape == (8, 32, 32, 1)
        assert set(np.unique(yb.numpy())) <= {-1.0, 0.0, 1.0}
    first_again = next(iter(make_train_dataset(X, Y, batch=8, crops_per_chip=4, crop=32, seed=0)))
    np.testing.assert_array_equal(batches[0][0].numpy(), first_again[0].numpy())  # same seed, same crops
    second_epoch = next(iter(ds))
    assert not np.array_equal(batches[0][0].numpy(), second_epoch[0].numpy())     # next epoch, new crops


def test_eval_dataset_keeps_order_and_full_size():
    X = np.arange(5, dtype=np.float32)[:, None, None, None] * np.ones((5, 64, 64, 2), dtype=np.float32)
    Y = np.zeros((5, 64, 64, 1), dtype=np.int8)
    out = np.concatenate([xb.numpy() for xb, _ in make_eval_dataset(X, Y, batch=2)])
    assert out.shape == (5, 64, 64, 2)
    np.testing.assert_array_equal(out[:, 0, 0, 0], np.arange(5))
