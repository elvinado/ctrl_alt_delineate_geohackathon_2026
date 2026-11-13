from types import SimpleNamespace

from scripts.geogg_fresh.contracts import StageResult, StageStatus
from scripts.geogg_fresh.inference_runner import _inference_stage_status
from scripts.geogg_fresh.volume_inference import _xy_mapping_quality


def test_inference_limitations_do_not_change_success_status():
    assert _inference_stage_status("09_inference", {}) is StageStatus.COMPLETED
    assert _inference_stage_status("09_inference", {
        "stage03_patch_parity": {"passed": True, "detail": "exact"},
    }) is StageStatus.COMPLETED


def test_failed_scientific_check_keeps_conditional_status():
    assert _inference_stage_status("09_inference", {
        "stage03_patch_parity": {"passed": False, "detail": "mismatch"},
    }) is StageStatus.CONDITIONAL


def test_blockers_take_precedence_over_checks_and_limitations():
    assert _inference_stage_status("09_inference", {}, blockers=["required input missing"]) is StageStatus.BLOCKED


def test_stage_result_serializes_quality_flags_separately_from_limitations():
    result = StageResult(
        stage_id="09_inference",
        status=StageStatus.COMPLETED,
        limitations=["XY mapping requires trace-coordinate QC"],
        quality_flags=[{
            "category": "xy_geometry",
            "detail": "XY mapping requires trace-coordinate QC",
        }],
    )
    payload = result.to_dict()
    assert payload["status"] == "COMPLETED"
    assert payload["limitations"] == ["XY mapping requires trace-coordinate QC"]
    assert payload["quality_flags"][0]["category"] == "xy_geometry"


def test_xy_mapping_quality_flags_non_affine_geometry_without_invalidating_native_grid():
    geometry = SimpleNamespace(
        affine={"x": [0, 1, 0], "y": [0, 0, 1]},
        affine_residual_max=17.68,
        affine_residual_rms=0.437,
        flags=[],
        bin_spacing=lambda: (20.28, 20.28),
    )
    quality = _xy_mapping_quality(geometry)
    assert quality["native_grid"] == "IL/XL"
    assert quality["xy_mapping_supported"] is False
    assert quality["xy_mapping_status"] == "requires_trace_coordinate_qc"
    assert quality["affine_residual_threshold_m"] == 5.07


def test_xy_mapping_quality_accepts_affine_geometry():
    geometry = SimpleNamespace(
        affine={"x": [0, 1, 0], "y": [0, 0, 1]},
        affine_residual_max=0.2,
        affine_residual_rms=0.1,
        flags=[],
        bin_spacing=lambda: (20.0, 20.0),
    )
    assert _xy_mapping_quality(geometry)["xy_mapping_supported"] is True



def test_stage09_geometry_limitation_does_not_downgrade_success():
    assert _inference_stage_status(
        "09_inference", {}, limitations=["XY mapping requires trace-coordinate QC"]
    ) is StageStatus.COMPLETED


def test_stage08_limitations_still_mark_evaluation_conditional():
    assert _inference_stage_status(
        "08_evaluation", {}, limitations=["required evaluation output unavailable"]
    ) is StageStatus.CONDITIONAL
