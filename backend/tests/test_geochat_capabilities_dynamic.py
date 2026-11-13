from app.geochat.skills.ask_geo_capabilities.handler import execute


def test_capabilities_follow_live_skill_registry(monkeypatch):
    from app.geochat.skills import registry

    class Skill:
        def __init__(self, name, description):
            self.name = name
            self.description = description

    monkeypatch.setattr(registry, "SKILL_REGISTRY", {
        "ask_geo_capabilities": Skill("ask_geo_capabilities", "Help"),
        "process_example": Skill("process_example", "Run example workflow."),
        "process_example_output_analysis": Skill("process_example_output_analysis", "Analyze example output."),
    })

    result = execute(db=None, project_id="", parameters={})
    names = [item["name"] for item in result["capabilities"]]
    assert names == ["process_example", "process_example_output_analysis"]
    assert any("Run <process workflow>" in example for example in result["examples"])


def test_capabilities_omit_processing_example_when_no_process_skills(monkeypatch):
    from app.geochat.skills import registry

    class Skill:
        name = "ask_geo_capabilities"
        description = "Help"

    monkeypatch.setattr(registry, "SKILL_REGISTRY", {"ask_geo_capabilities": Skill()})
    result = execute(db=None, project_id="", parameters={})
    assert not any("Run <process workflow>" in example for example in result["examples"])
