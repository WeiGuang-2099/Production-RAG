from app.guardrails.pii import redact_pii


def test_redacts_email():
    out, types = redact_pii("contact me at john.doe@example.com please")
    assert "[REDACTED_EMAIL]" in out and "john.doe@example.com" not in out
    assert "email" in types


def test_redacts_ssn_and_phone():
    out, types = redact_pii("SSN 123-45-6789 call 555-123-4567")
    assert "[REDACTED_SSN]" in out and "[REDACTED_PHONE]" in out
    assert {"ssn", "phone"} <= set(types)


def test_redacts_credit_card():
    out, types = redact_pii("card 4111 1111 1111 1111")
    assert "[REDACTED_CC]" in out and "credit_card" in types


def test_clean_text_unchanged():
    text = "The Transformer uses 8 attention heads."
    out, types = redact_pii(text)
    assert out == text and types == []
