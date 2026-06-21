from app.guardrails.toxicity import detect_toxicity


def test_flags_listed_term():
    assert "idiot" in detect_toxicity("you are an idiot")


def test_case_insensitive():
    assert "moron" in detect_toxicity("What a MORON")


def test_benign_is_clean():
    assert detect_toxicity("The paper compares BERT and GPT-3.") == []
