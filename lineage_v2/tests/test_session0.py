"""Session 0 unit tests — no Kaggle, no tracksdata required."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lineage_v2.checkpoint import (  # noqa: E402
    build_checkpoint,
    is_full_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from lineage_v2.constants import FORBIDDEN_TEST_STEMS, METRIC_PIN_SHA256  # noqa: E402
from lineage_v2.denylist import ForbiddenStemError, assert_path_allowed_for_gt  # noqa: E402
from lineage_v2.eval.contract import official_score, verify_metric_pin  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402
from lineage_v2.splits import build_manifest, load_manifest, save_manifest  # noqa: E402
from lineage_v2.targets import decode_peaks, make_targets  # noqa: E402


def test_metric_pin():
    pins = verify_metric_pin()
    assert set(pins) == set(METRIC_PIN_SHA256)


def test_official_score():
    assert abs(official_score(0.90, 0.20) - 0.92) < 1e-12


def test_denylist():
    for s in FORBIDDEN_TEST_STEMS:
        try:
            assert_path_allowed_for_gt(f"/data/train/{s}/x.geff")
            raise AssertionError("should have raised")
        except ForbiddenStemError:
            pass
    assert_path_allowed_for_gt("/data/train/6bba_deadbeef/x.geff")


def test_manifest_excludes_forbidden_and_confirmation_counts():
    stems = []
    for i in range(30):
        stems.append(f"44b6_{i:08x}")
    for i in range(50):
        stems.append(f"6bba_{i:08x}")
    # sneak forbidden
    stems += list(FORBIDDEN_TEST_STEMS)
    try:
        build_manifest(stems)
        raise AssertionError("must reject forbidden")
    except ForbiddenStemError:
        pass
    stems = [s for s in stems if s not in FORBIDDEN_TEST_STEMS]
    m = build_manifest(stems)
    assert m.meta["n_44b6_conf"] == 8
    assert m.meta["n_6bba_conf"] == 16
    assert len(m.confirmation) == 24
    assert len(m.folds) == 3
    conf = set(m.confirmation)
    for f in m.folds:
        assert conf.isdisjoint(f)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.json"
        h = save_manifest(m, p)
        m2 = load_manifest(p)
        assert m2.sha256() == m.sha256() == h


def test_gaussian_two_close_centers_resolvable():
    # two centers ~11 µm apart in xy (spacing 0.40625 → ~27 voxels)
    shape = (16, 64, 64)
    c0 = (8.0, 20.0, 20.0)
    c1 = (8.0, 20.0, 47.0)  # ~11 µm in x
    heat, offset, flow, ov, fv = make_targets(shape, [c0, c1], [-1, -1], sigma_um=2.4)
    assert heat.max() > 0.9
    peaks = decode_peaks(heat, offset, threshold=0.3, nms_um=3.0)
    assert peaks.shape[0] >= 2, f"expected 2 peaks, got {peaks.shape[0]}"


def test_offset_and_flow():
    shape = (12, 32, 32)
    parent = (6.3, 10.2, 10.4)
    child = (6.1, 12.5, 11.0)
    heat, offset, flow, ov, fv = make_targets(shape, [parent, child], [-1, 0])
    az, ay, ax = 6, 12, 11  # floor(child)
    assert ov[az, ay, ax]
    assert abs(float(offset[0, az, ay, ax]) - (6.1 - 6)) < 1e-5
    assert fv[az, ay, ax]
    # backward parent-child in µm: (parent-child)*spacing
    assert flow[:, az, ay, ax].abs().sum() > 0


def test_checkpoint_full_resume_equivalence():
    torch.manual_seed(0)
    model = CellectLite(time_frames=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # one step
    x = torch.randn(1, 1, 8, 32, 32)
    out = model(x)
    loss = out["center_logits"].float().pow(2).mean()
    loss.backward()
    opt.step()
    state = build_checkpoint(
        model=model,
        optimizer=opt,
        stage="A",
        global_step=1,
        best_exact=0.1,
        split_sha256="abc",
        inference_policy_sha256="def",
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "last.pt"
        save_checkpoint(state, path)
        assert is_full_resume_checkpoint(path)
        model2 = CellectLite(time_frames=1)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        load_checkpoint(
            path,
            model=model2,
            optimizer=opt2,
            expected_split_sha256="abc",
            expected_inference_sha256="def",
        )
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), n1


def test_cellect_forward_shapes():
    m = CellectLite(time_frames=2)
    x = torch.randn(1, 2, 1, 16, 64, 64)
    o = m(x)
    assert o["center_logits"].shape[0] == 1
    assert o["center_logits"].shape[1] == 1
    assert o["offset_vox"].shape[1] == 3
    assert o["embedding_map"].shape[1] == 64


def test_legacy_weights_not_full_resume():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "w.pth"
        torch.save({"state_dict": CellectLite(time_frames=1).state_dict()}, path)
        assert not is_full_resume_checkpoint(path)


if __name__ == "__main__":
    test_metric_pin()
    test_official_score()
    test_denylist()
    test_manifest_excludes_forbidden_and_confirmation_counts()
    test_gaussian_two_close_centers_resolvable()
    test_offset_and_flow()
    test_checkpoint_full_resume_equivalence()
    test_cellect_forward_shapes()
    test_legacy_weights_not_full_resume()
    print("ALL_PASS session0")
