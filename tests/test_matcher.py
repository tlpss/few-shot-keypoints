"""Regression tests for the matcher.

Plain asserts, so this runs both under pytest and as `python tests/test_matcher.py` (pytest is not a dependency
of this project yet).
"""

import torch

from few_shot_keypoints.matcher import COMPARISON_DTYPE, KeypointFeatureMatcher, custom_cos_sim

DEVICE = "cpu"


def _matcher(reference_vectors, **kwargs):
    return KeypointFeatureMatcher(reference_vectors, device=DEVICE, **kwargs)


def test_topk_suppression_is_clamped_at_the_top_left_border():
    """A negative slice start counts from the end of the tensor, which used to make the suppression window empty
    for any match within `min_distance_between_topk_matches` of the top/left border."""
    padding = 10
    matcher = _matcher(torch.ones(1, 4), top_k_matches=[2], min_distance_between_topk_matches=padding)

    similarities = torch.zeros(1, 64, 64, dtype=COMPARISON_DTYPE)
    similarities[0, 1, 1] = 1.0  # best match, right against the top-left border
    similarities[0, 2, 2] = 0.9  # inside the suppression window, must NOT be returned
    similarities[0, 40, 40] = 0.8  # outside it, must be returned as second best

    best, second = matcher.get_best_matches_from_similarities(similarities)[0]
    assert (best.v, best.u) == (1, 1)
    assert (second.v, second.u) == (40, 40), f"second match {second} was not suppressed near the border"


def test_topk_suppression_away_from_the_border():
    padding = 10
    matcher = _matcher(torch.ones(1, 4), top_k_matches=[2], min_distance_between_topk_matches=padding)

    similarities = torch.zeros(1, 64, 64, dtype=COMPARISON_DTYPE)
    similarities[0, 30, 30] = 1.0
    similarities[0, 31, 31] = 0.9  # inside the window
    similarities[0, 50, 50] = 0.8

    best, second = matcher.get_best_matches_from_similarities(similarities)[0]
    assert (best.v, best.u) == (30, 30)
    assert (second.v, second.u) == (50, 50)


def _featurizer_like_features(patch_grid: int = 32, dim: int = 128, size: int = 512, seed: int = 0):
    """A bf16 feature map with the same structure the ViT featurizers produce: a coarse patch grid, bilinearly
    upsampled to pixel resolution. Neighbouring pixels are therefore highly correlated, which is what creates the
    tie plateaus that bf16 cannot resolve (random per-pixel features would not reproduce this)."""
    torch.manual_seed(seed)
    patches = torch.randn(1, dim, patch_grid, patch_grid)
    features = torch.nn.functional.interpolate(patches, size=(size, size), mode="bilinear", align_corners=False)
    return features.to(torch.bfloat16)


def test_similarities_are_compared_in_fp32():
    assert COMPARISON_DTYPE == torch.float32
    features = _featurizer_like_features()
    reference = features[0, :, 300, 200][None].clone()
    assert custom_cos_sim(features, reference).dtype == torch.float32


def test_bf16_features_do_not_create_a_tie_plateau_at_the_peak():
    """bf16 similarities put dozens of pixels at the exact same max, and argmax then breaks the tie by flat index,
    biasing every match towards the top-left of the plateau. Comparing in fp32 keeps the peak unique."""
    features = _featurizer_like_features()
    v, u = 300, 200
    reference = features[0, :, v, u][None].clone()

    similarities = custom_cos_sim(features, reference)[0]
    n_tied = int((similarities == similarities.max()).sum())
    assert n_tied == 1, f"{n_tied} pixels tie at the similarity peak; argmax between them is arbitrary"


def test_matching_a_pixel_against_its_own_feature_map_is_exact():
    """The strongest available end-to-end check: the best match for a pixel's own feature vector is that pixel.

    Only interior pixels are checked. `align_corners=False` upsampling makes the features in the outer half-patch
    band of the map exactly constant, so pixels there tie regardless of precision and cannot be localised more
    precisely than that band - a property of upsampling patch features, not of the matcher."""
    features = _featurizer_like_features()
    coords = [(300, 200), (120, 400), (57, 43), (480, 60), (256, 256)]
    references = torch.stack([features[0, :, v, u] for v, u in coords])

    matcher = _matcher(references)
    matches = matcher.get_best_matches_from_image_features(features)
    for (v, u), match in zip(coords, matches):
        assert (match[0].v, match[0].u) == (v, u), f"self-match landed at {(match[0].v, match[0].u)} instead of {(v, u)}"


def test_mask_restricts_the_search_region():
    torch.manual_seed(0)
    features = _featurizer_like_features(patch_grid=8, dim=32, size=32)
    v, u = 4, 4
    reference = features[0, :, v, u][None].clone()

    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[16:, 16:] = True

    matcher = _matcher(reference)
    match = matcher.get_best_matches_from_image_features(features, mask=mask)[0][0]
    assert mask[match.v, match.u], f"match {(match.v, match.u)} fell outside the mask"


def test_similarities_are_not_modified_in_place():
    matcher = _matcher(torch.ones(1, 4), top_k_matches=[2], min_distance_between_topk_matches=5)
    similarities = torch.zeros(1, 32, 32, dtype=COMPARISON_DTYPE)
    similarities[0, 10, 10] = 1.0
    original = similarities.clone()

    matcher.get_best_matches_from_similarities(similarities)
    assert torch.equal(similarities, original), "the caller's similarity map was modified"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
