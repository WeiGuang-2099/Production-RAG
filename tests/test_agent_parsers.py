from app.agent.parsers import parse_grade, parse_route


def test_parse_route_each_label():
    assert parse_route("retrieve") == "retrieve"
    assert parse_route("answer") == "answer"
    assert parse_route("clarify") == "clarify"


def test_parse_route_is_case_and_noise_insensitive():
    assert parse_route("RETRIEVE.") == "retrieve"
    assert parse_route("I think we should answer this") == "answer"


def test_parse_route_defaults_to_retrieve():
    assert parse_route("???") == "retrieve"
    assert parse_route("") == "retrieve"


def test_parse_grade_yes_no():
    assert parse_grade("yes") is True
    assert parse_grade("Yes, sufficient") is True
    assert parse_grade("no") is False
    assert parse_grade("No, missing the key fact") is False


def test_parse_grade_defaults_to_true():
    assert parse_grade("???") is True
    assert parse_grade("") is True
