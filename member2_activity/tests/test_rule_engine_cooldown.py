
from member2_activity.rule_engine import RuleEngine


def test_cooldown_suppresses_repeated_alerts():
    engine = RuleEngine(buffer=None)
    engine.cooldown_sec = 5.0

    assert engine._should_emit("running", "person-1", 1.0)
    assert not engine._should_emit("running", "person-1", 2.0)
    assert not engine._should_emit("running", "person-1", 5.9)
    assert engine._should_emit("running", "person-1", 6.0)


def test_different_people_have_independent_cooldowns():
    engine = RuleEngine(buffer=None)
    engine.cooldown_sec = 5.0

    assert engine._should_emit("running", "person-1", 1.0)
    assert engine._should_emit("running", "person-2", 1.0)


def test_different_activities_have_independent_cooldowns():
    engine = RuleEngine(buffer=None)
    engine.cooldown_sec = 5.0

    assert engine._should_emit("running", "person-1", 1.0)
    assert engine._should_emit("loitering", "person-1", 1.0)


def test_zero_cooldown_allows_each_observation():
    engine = RuleEngine(buffer=None)
    engine.cooldown_sec = 0.0

    assert engine._should_emit("running", "person-1", 1.0)
    assert engine._should_emit("running", "person-1", 1.0)
