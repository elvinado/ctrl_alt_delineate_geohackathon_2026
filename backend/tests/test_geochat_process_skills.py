"""Ask Geo integration of the Process A-F skills: discovery, routing and the fixed answer layout.

The result dictionaries below are SOFTWARE-TEST fixtures shaped like the handlers' output. Their
numbers are illustrative and must never be quoted as field results.
"""
from __future__ import annotations

import importlib
import json
import re

import pytest

LETTERS = {
    "a": "process_a_foundations", "b": "process_b_inference", "c": "process_c_volume_products",
    "d": "process_d_geobody", "e": "process_e_well_geobody_analysis", "f": "process_f_prospect",
}
JOB = "8243c6e5-3cc3-4d42-ae62-64f609c7dd4c"


def _finding(fid, level, observed, meaning, files):
    return {"id": fid, "level": level, "observed": observed, "plain_meaning": meaning, "evidence": files}


def _base(letter, **extra):
    takeaway = _finding("no_depth_baseline", "critical", "The 'beyond_depth.csv' table is empty.",
                        "No model was compared with a depth-only baseline.", ["stages/08_evaluation/tables/beyond_depth.csv"])
    other = _finding("status_note", "info", "Run status is COMPLETED_WITH_GRV_BLOCKERS | see summary.",
                     "The run finished; some outputs were withheld by its own gates.", ["geogg_prospect_summary.json"])
    return {
        "skill": f"{LETTERS[letter]}_output_analysis", "job_id": JOB, "python_script": None, "status": "succeeded",
        "applicable": True, "focus": "overview",
        "verdict": {"level": "usable_with_caveats", "plain": "The run completed and passed all 49 of its self-checks.",
                    "checks_passed": 49, "checks_total": 49, "blockers": 0, "limitations": 12},
        "summary": "The run completed.", "takeaways": [takeaway], "key_findings": [takeaway, other],
        "findings": [takeaway["observed"], other["observed"]],
        "next_steps": ["Compare every model with a depth-only (TWT) baseline.", "Obtain measured checkshots."],
        "details": {}, "findings_not_shown": 3, "suggested_questions": [],
        "artifacts": [{"name": "geogg_foundations_summary.json", "size_bytes": 1200, "role": "evidence read by this analysis"}],
        "skipped": [], "limitations": ["Numbers are read from the run's own tables; nothing is recomputed."],
        **extra,
    }


RESULTS = {
    "a": _base(
        "a",
        scorecard=[{"target": "sand versus shale", "task": "lithology", "input": "stacks@matched", "model": "hist_gbm",
                    "score": "ROC-AUC 0.71", "rating": "modest", "basis": "matched rows",
                    "inversion_volumes": "no clear difference", "deep_learning": "hurts"}],
        stage_cards=[{"stage_id": "01_well_ties", "title": "Depth-to-time conversion", "what_it_does": "Places logs in time.",
                      "status": "CONDITIONAL", "light": "amber", "checks_passed": 6, "checks_total": 6, "n_limitations": 2,
                      "n_warnings": 1, "failed_checks": [], "blockers": []}],
        well_card={"well": "M-1", "found": True, "tie_status": "NOT_VALIDATED", "test_group": 2, "training_rows": 1840,
                   "seismic_time_window_ms": [1204.0, 2380.0], "thickness_m_by_class": {"sand": 212.5, "shale": 640.0}},
        details={"ties": {"wells_processed": 13, "wells_with_validated_tie": 0, "time_uncertainty_rms_ms": 14.2,
                          "tie_status_counts": {"NOT_VALIDATED": 13},
                          "selection_rule": "The regional model with the lowest mean leave-one-well-out error is used, "
                                            "provided it is physically valid over the logged interval of every well.",
                          "per_well": [{"well": "M-1", "time_depth_source": "checkshot", "tie_status": "NOT_VALIDATED",
                                        "extrapolation_m": 0.0, "strong_extrapolation": False}]}},
    ),
    "b": _base(
        "b",
        scorecard=[{"target": "total porosity (PHIT)", "task": "phit", "input": "patch_stacks", "model": "cnn3d_multitask",
                    "score": "R2 0.31, RMSE 0.041", "rating": "moderate", "wells": 13, "rows": 22140, "basis": "all rows"}],
        cube_guide=[{"property": "sand versus shale", "task": "lithology", "file": "prediction_lithology.npy",
                     "file_present": True, "size_gb": 8.1, "made_by": "cnn3d_multitask", "held_out_score": "ROC-AUC 0.66",
                     "rating": "modest", "advice": "Use as a relative indicator. Not for quantitative work."}],
        stage_cards=[{"stage_id": "09_inference", "title": "Prediction cubes for the whole survey", "status": "CONDITIONAL",
                      "light": "amber", "checks_passed": 9, "checks_total": 9, "limitations": ["No blind well."],
                      "warnings": [], "failed_checks": [], "blockers": []}],
    ),
    "c": _base(
        "c",
        verdict={"level": "usable_with_caveats", "plain": "The package is complete.", "audit_status": "conditional",
                 "audit_checks": {"PASS": 16, "WARN": 2, "NOT_TESTABLE": 1}, "validation_status": "in_progress",
                 "note": "File integrity and scientific quality are separate."},
        cube_guide=[{"property": "shale volume (VSH)", "task": "vsh", "made_by": "cnn3d_multitask",
                     "held_out_score": "R2 0.01", "rating": "none", "advice": "Do not interpret this cube."}],
        figures_guide={"dashboard": {"file": "stages/10_summary/dashboard.html", "present": True, "size_mb": 4.2,
                                     "shows": "Offline one-page overview of the package."}, "sections": []},
    ),
    "d": _base(
        "d",
        verdict={"level": "usable_with_caveats", "plain": "248 bodies were kept; one holds 97% of all body voxels.",
                 "segmentation": "one_dominant_body", "bodies_kept": 248, "preflight_status": "PASS"},
        at_a_glance={"bodies_kept": 248, "one_dominant_body": True, "share_of_predicted_voxels_in_bodies": 0.412,
                     "low_threshold": 0.35, "high_threshold": 0.65, "interpretation_status": "PROVISIONAL", "units": "voxels"},
        body_card={"geobody_id": 17, "found": True, "voxels": 21044, "shape": "sheet", "twt_range_ms": [1820.0, 1904.0],
                   "centre": {"inline": 1420, "crossline": 3310, "twt_ms": 1860.0}, "reading": "A thin, laterally extensive body."},
    ),
    "e": _base(
        "e",
        verdict={"level": "usable_with_caveats", "plain": "All 13 wells meet the same survey-wide body.",
                 "assessment": "same_background_body", "wells_analysed": 13},
        at_a_glance={"wells_analysed": 13, "different_bodies_met": 1, "all_wells_same_body": True,
                     "share_of_path_inside_a_body": 0.44, "share_of_cube_inside_a_body": 0.41,
                     "wells_with_a_depth_for_their_body": 0, "bodies_with_porosity_or_saturation_statistics": 0},
        details={"wells": {"wells_analysed": 13, "per_well": [
            {"well": "M-1", "status": "INTERSECTS", "path_samples": 512, "samples_inside_a_body": 230,
             "share_of_path_inside_a_body": 0.449, "bodies_met": 1, "selected_body": 1, "separate_stretches": 14,
             "typical_stretch_m": 18.5, "inside_from_md_m": 1310.0, "inside_to_md_m": 2890.5}]}},
    ),
    "f": _base(
        "f",
        verdict={"level": "usable_with_caveats", "plain": "244 prospects were defined, all cut from one body.",
                 "assessment": "background_slices", "run_status": "COMPLETED_WITH_GRV_BLOCKERS", "prospects": 244,
                 "volumes_reported": False},
        at_a_glance={"prospects": 244, "survey_scale_prospects": 236, "compact_prospects": 8, "prospects_with_rock_volume": 0,
                     "prospects_with_hcpv": 0, "typical_net_to_gross_from_logs": 0.30, "sand_share_of_all_logged_rock": 0.20,
                     "typical_water_saturation_of_net": 0.81, "run_status": "COMPLETED_WITH_GRV_BLOCKERS"},
        prospect_card={"prospect": "P042", "found": False, "example_ids": ["P064", "P101"], "prospects": 244},
        glossary={"GRV": "Gross rock volume: the bulk volume of rock inside the prospect outline.",
                  "survey_scale_object": "The run's own flag for a prospect as wide as the survey."},
    ),
}


def _formatting(letter):
    return importlib.import_module(f"app.geochat.skills.{LETTERS[letter]}_output_analysis.formatting")


def _outside_code(markdown: str) -> str:
    return re.sub(r"`[^`]*`", "", markdown)


@pytest.mark.parametrize("letter", sorted(LETTERS))
def test_result_layout_is_fixed_and_markdown_safe(letter):
    module = _formatting(letter)
    markdown = module.render_markdown(json.loads(json.dumps(RESULTS[letter])))
    lines = markdown.splitlines()

    assert lines[0] == "# " + module.RESULT_MARKER
    assert f"`{JOB}`" in lines[2]
    assert "**Verdict: Usable with caveats.**" in markdown
    # Fixed order: verdict, what matters most, next steps, evidence footer.
    order = [markdown.index(marker) for marker in ("> ", "## What matters most", "## Suggested next steps", "**Evidence read:**")]
    assert order == sorted(order)
    # Identifiers are code, so the chat renderer cannot turn their underscores into italics.
    assert "_" not in _outside_code(markdown)
    # Every table row has as many cells as its header (a stray pipe would shift columns).
    for index, line in enumerate(lines):
        if line.startswith("|---"):
            width = lines[index - 1].count("|")
            row = index + 1
            while row < len(lines) and lines[row].startswith("|"):
                assert lines[row].count("|") == width, lines[row]
                row += 1


def test_process_specific_tables_and_cards_are_rendered():
    a = _formatting("a").render_markdown(RESULTS["a"])
    assert "## Scorecard: practice scores, not final accuracy" in a
    assert "| sand versus shale | ROC-AUC 0.71 | 🟡 modest | `hist_gbm` | stacks only, matched rows |" in a
    assert "| `01_well_ties` | Depth-to-time conversion | 🟡 CONDITIONAL | 6 of 6 | 2 | 1 |" in a
    assert "## Well card: M-1" in a and "| Seismic time window (ms) | 1,204 to 2,380 |" in a
    assert "## Details: ties" in a and "### Per well" in a

    b = _formatting("b").render_markdown(RESULTS["b"])
    assert b.index("## Which prediction cubes to trust") < b.index("## Scorecard: held-out wells")
    assert "- 🟡 `09_inference` limitation: No blind well." in b

    d = _formatting("d").render_markdown(RESULTS["d"])
    assert "| Share of predicted cube inside bodies | 41.2% |" in d and "| One body dominates | Yes |" in d
    assert "## Geobody card: 17" in d

    f = _formatting("f").render_markdown(RESULTS["f"])
    assert "| Prospects defined | 244 |" in f and "| Run status | `COMPLETED_WITH_GRV_BLOCKERS` |" in f
    assert "## Prospect card: P042" in f and "Available: P064, P101" in f
    assert "## Glossary" in f and "3 more finding(s) are held back" in f


def test_plain_language_reading_sits_between_verdict_and_tables():
    result = {**RESULTS["e"], "narrative": "The run shows every well meets the same body."}
    markdown = _formatting("e").render_markdown(result)
    assert markdown.index("**Verdict") < markdown.index("**In plain words**") < markdown.index("## At a glance")


def test_inapplicable_folder_is_reported_not_invented():
    result = {"job_id": JOB, "status": "failed", "applicable": False,
              "summary": "This folder was not produced by Process D (expected script 'process_D_geobody.py').",
              "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}
    markdown = _formatting("d").render_markdown(result)
    assert "**Not analysed.**" in markdown and "## " not in markdown


@pytest.mark.parametrize("letter", sorted(LETTERS))
def test_preflight_layout(letter):
    module = importlib.import_module(f"app.geochat.skills.{LETTERS[letter]}.formatting")
    handler = importlib.import_module(f"app.geochat.skills.{LETTERS[letter]}.handler")
    result = handler.execute(project_id="p1", parameters={}, db=None)  # no inputs named: requirements only
    markdown = module.render_markdown(result)
    assert markdown.splitlines()[0] == "# " + module.PREFLIGHT_MARKER
    assert "**Inputs needed.**" in markdown and "## Inputs this process needs" in markdown
    assert "## What the run will do" in markdown and "Nothing was submitted." in markdown
    assert "_" not in _outside_code(markdown)

    blocked = {**result, "status": "not_ready", "summary": "Not ready: 1 blocking issue(s), 1 warning(s).", "issues": [
        {"level": "warning", "code": "few_wells", "message": "Only 5 wells have logs.", "how_to_fix": "Add wells."},
        {"level": "error", "code": "no_time_depth", "message": "No time-depth table was selected.", "how_to_fix": "Add one."}]}
    markdown = module.render_markdown(blocked)
    assert "**Not ready.**" in markdown
    assert markdown.index("🔴 Blocks the run") < markdown.index("🟡 Warning")  # blocking issues first


# ------------------------------------------------------------------------------------------------
# Routing: each skill's own intent.py decides which questions it answers
# ------------------------------------------------------------------------------------------------
def _route(message, after=None):
    from app.geochat import agent

    history = []
    if after:
        history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "# " + _formatting(after).RESULT_MARKER}]
    for skill in agent._process_skills():
        parameters = skill.intent(message, history)
        if parameters is not None:
            return skill.name, parameters
    return None, None


@pytest.mark.parametrize("message, skill, expected", [
    ("Explain the results of Process A", "process_a_foundations_output_analysis", {"focus": "overview"}),
    ("How do I explain the results of process B?", "process_b_inference_output_analysis", {"focus": "overview"}),
    ("Interpret the volume products output", "process_c_volume_products_output_analysis", {"focus": "overview"}),
    ("Summarise the geobody extraction results", "process_d_geobody_output_analysis", {"focus": "overview"}),
    ("explain the well-geobody analysis", "process_e_well_geobody_analysis_output_analysis", {"focus": "overview"}),
    ("What did process F find?", "process_f_prospect_output_analysis", {"focus": "overview"}),
    ("explain process a results in full", "process_a_foundations_output_analysis", {"focus": "all"}),
    ("Explain prospect P042 from process F", "process_f_prospect_output_analysis", {"focus": "prospects", "prospect": "P042"}),
    ("explain body 17 in process D results", "process_d_geobody_output_analysis", {"focus": "bodies", "geobody_id": 17}),
    (f"Explain the results of process D job {JOB}", "process_d_geobody_output_analysis", {"focus": "overview", "job_id": JOB}),
    ("What does Process A need?", "process_a_foundations", {"_project_datasets": True}),
    ("Run process B", "process_b_inference",
     {"_source_job": {"parameter": "foundation_job_id", "python_script": "process_A_foundations.py"}}),
    ("Can I run process E for wells M-1 and M-3?", "process_e_well_geobody_analysis",
     {"_source_job": {"parameter": "source_geobody_job_id", "python_script": "process_D_geobody.py"}, "wells": ["M-1", "M-3"]}),
    ("prepare process d with threshold 0.4", "process_d_geobody",
     {"_source_job": {"parameter": "source_volume_job_id", "python_script": "process_C_volume_products.py"}, "low_threshold": 0.4}),
])
def test_questions_that_name_a_process(message, skill, expected):
    assert _route(message) == (skill, expected)


@pytest.mark.parametrize("message", [
    "What processing workflows can I run?", "List my completed processing jobs", "What datasets are available?",
    "can you process a new seismic volume?", "What is a geobody?", "List the wells in this project",
])
def test_other_questions_are_left_to_the_rest_of_ask_geo(message):
    assert _route(message) == (None, None)


@pytest.mark.parametrize("after, message, focus", [
    ("a", "Why are there no oil or gas labels?", "labels"),
    ("a", "How reliable is the depth-to-time conversion?", "ties"),
    ("b", "Over what time range is the cube reliable?", "by_time"),
    ("c", "What did the audit find?", "audit"),
    ("d", "Why is one body so much larger than all the others?", "method"),
    ("e", "Did any well confirm a geobody?", "wells"),
    ("f", "Why are there no volumes?", "volumes"),
])
def test_follow_ups_stay_with_the_report_they_follow(after, message, focus):
    skill, parameters = _route(message, after=after)
    assert skill == f"{LETTERS[after]}_output_analysis" and parameters["focus"] == focus


def test_follow_ups_do_not_capture_unrelated_requests():
    assert _route("What datasets are available?", after="a") == (None, None)
    assert _route("which wells have logs?", after="e") == (None, None)
    assert _route("thanks!", after="a") == (None, None)
    assert _route("now explain process B", after="a")[0] == "process_b_inference_output_analysis"


def test_every_handler_suggested_question_routes_back_to_its_own_skill():
    for letter, package in LETTERS.items():
        handler = importlib.import_module(f"app.geochat.skills.{package}_output_analysis.handler")
        for question in handler.SUGGESTED_QUESTIONS:
            skill, parameters = _route(question, after=letter)
            assert skill == f"{package}_output_analysis", question
            assert parameters["focus"] in set(handler.FOCUS_ALIASES.values()), question


def test_registry_discovers_renderers_parsers_and_job_matchers():
    from app.geochat.skills.registry import SKILL_REGISTRY

    for package in LETTERS.values():
        run, analysis = SKILL_REGISTRY[package], SKILL_REGISTRY[f"{package}_output_analysis"]
        assert run.handler and run.renderer and run.intent and run.scope == "project"
        assert analysis.handler and analysis.renderer and analysis.intent and analysis.scope == "global"
        handler = importlib.import_module(f"app.geochat.skills.{package}_output_analysis.handler")
        assert handler.supports_job({"python_script": handler.EXPECTED_SCRIPT})
        assert not handler.supports_job({"python_script": "example_worker.py"})


# ------------------------------------------------------------------------------------------------
# Agent: latest-run lookup, grounded reading, deterministic layout
# ------------------------------------------------------------------------------------------------
class _Completions:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        message = type("Message", (), {"content": self.reply})()
        return type("Completion", (), {"choices": [type("Choice", (), {"message": message})()], "usage": None, "model": "test"})()


class _Client:
    def __init__(self, reply):
        self.chat = type("Chat", (), {"completions": _Completions(reply)})()


def _ask(monkeypatch, message, *, reply="The run shows one dominant body.", records=None, result=None, history=None):
    from app.geochat import agent

    records = records if records is not None else [
        {"job_id": JOB, "metadata": {"python_script": "process_D_geobody.py"}, "status": "succeeded", "order": ("2", 2.0)},
        {"job_id": "11111111-2222-4333-8444-555555555555", "metadata": {"python_script": "process_D_geobody.py"},
         "status": "succeeded", "order": ("1", 1.0)},
    ]
    calls = []

    def fake_execute(name, *, db, project_id, parameters):
        calls.append((name, dict(parameters)))
        return json.loads(json.dumps(result if result is not None else RESULTS["d"]))

    monkeypatch.setattr(agent, "_job_records", lambda db: records)
    monkeypatch.setattr(agent, "execute_skill", fake_execute)
    client = _Client(reply)
    messages = [*(history or []), {"role": "user", "content": message}]
    answer = agent.run_agent(client, model="m", system_prompt="s", messages=messages, project_id=None, db=None,
                             generation={"temperature": 0.7, "max_tokens": 2000, "top_p": 1.0})
    return answer, calls, client.chat.completions.calls


def test_agent_explains_the_latest_run_in_the_fixed_layout(monkeypatch):
    (content, skill, ui_action, _, _), calls, llm = _ask(monkeypatch, "Explain the results of Process D")
    assert skill == "process_d_geobody_output_analysis" and ui_action is None
    assert calls == [(skill, {"focus": "overview", "job_id": JOB})]  # newest finished run, no project needed
    assert content.startswith("# Process D · Geobody extraction (stage 12) — results")
    assert "**In plain words**\n\nThe run shows one dominant body." in content
    assert "most recent of 2 runs" in content
    # One model call, for the short reading only, capped and grounded in the handler's evidence.
    assert len(llm) == 1 and llm[0]["max_tokens"] <= 320 and llm[0]["temperature"] <= 0.3
    assert "EVIDENCE JSON" in llm[0]["messages"][1]["content"] and "Expert reading rules" in llm[0]["messages"][0]["content"]


def test_agent_answer_survives_an_unavailable_language_model(monkeypatch):
    from openai import OpenAIError

    (content, skill, *_), _, _ = _ask(monkeypatch, "Explain the results of Process D", reply=OpenAIError("down"))
    assert skill == "process_d_geobody_output_analysis"
    assert "**Verdict: Usable with caveats.**" in content and "In plain words" not in content


def test_agent_uses_the_job_named_in_the_conversation(monkeypatch):
    older = "11111111-2222-4333-8444-555555555555"
    history = [{"role": "user", "content": f"Explain the results of process D job {older}"},
               {"role": "assistant", "content": "# " + _formatting("d").RESULT_MARKER + f"\n\nJob `{older}`"}]
    (_, skill, *_), calls, _ = _ask(monkeypatch, "Which bodies are worth looking at first?", history=history)
    assert skill == "process_d_geobody_output_analysis"
    assert calls == [(skill, {"focus": "bodies", "job_id": older})]


def test_agent_says_so_when_a_process_has_no_run(monkeypatch):
    (content, skill, *_), calls, llm = _ask(monkeypatch, "Explain the results of Process D", records=[])
    assert skill == "process_d_geobody_output_analysis" and not calls and not llm
    assert "could not find a run" in content


def test_agent_summarises_every_process_without_a_model_call(monkeypatch):
    (content, skill, *_), calls, llm = _ask(monkeypatch, "Explain the results of each process")
    assert skill is None and not llm
    assert content.startswith("# Results of every process")
    assert content.count("⚪ No run found") == 5 and "| Process D · Geobody extraction (stage 12) | `" + JOB in content
    assert calls == [("process_d_geobody_output_analysis", {"job_id": JOB})]


def test_agent_preflight_fills_in_the_upstream_job(monkeypatch):
    records = [{"job_id": JOB, "metadata": {"python_script": "process_D_geobody.py"}, "status": "succeeded", "order": ("1", 1.0)}]
    preflight = {"skill": "process_f_prospect", "python_script": "process_F_prospect.py", "status": "ready_for_confirmation",
                 "summary": "Ready to run.", "issues": [], "requirements": [], "what_will_happen": ["14 Prospect definition."]}
    from app.geochat import agent

    monkeypatch.setattr(agent, "_job_records", lambda db: records)
    seen = []
    monkeypatch.setattr(agent, "execute_skill", lambda name, *, db, project_id, parameters: seen.append((name, parameters)) or preflight)
    content, skill, ui_action, _, _ = agent.run_agent(
        _Client("unused"), model="m", system_prompt="s", messages=[{"role": "user", "content": "Run process F"}],
        project_id="p1", db=None, generation={})
    assert skill == "process_f_prospect" and ui_action is None
    assert seen == [("process_f_prospect", {"source_geobody_job_id": JOB})]
    assert "**Ready to run.**" in content and "submit `process_F_prospect.py` from the Processing tab" in content


def test_suggested_questions_come_from_the_skill_and_skip_what_was_asked():
    from app.geochat.api import _suggested_questions

    handler = importlib.import_module("app.geochat.skills.process_f_prospect_output_analysis.handler")
    first = _suggested_questions("process_f_prospect_output_analysis", None, "", asked=[])
    assert first == handler.SUGGESTED_QUESTIONS[:4]
    later = _suggested_questions("process_f_prospect_output_analysis", None, "", asked=[first[0].upper()])
    assert first[0] not in later and len(later) == 4
