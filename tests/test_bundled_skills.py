from services.memory.skills import SkillsManager


def test_bundled_law_study_skill_is_seeded_and_available_to_users(tmp_path):
    manager = SkillsManager(str(tmp_path))
    skills = manager.load(owner="test-user")
    skill = next(item for item in skills if item.get("name") == "law-study-assistant")

    assert skill["source"] == "builtin"
    assert skill["status"] == "published"
    assert "England and Wales" in skill["body_extra"]
    assert "Never answer the questions" in skill["body_extra"]
