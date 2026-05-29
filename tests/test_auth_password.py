"""bcrypt password hashing round-trips and rejects bad input safely."""

from src.auth import hash_password, verify_password


def test_hash_is_not_plaintext():
    h = hash_password("correct horse")
    assert h != "correct horse"
    assert isinstance(h, str)
    assert len(h) > 20


def test_verify_correct_password():
    h = hash_password("s3cret!")
    assert verify_password("s3cret!", h) is True


def test_verify_wrong_password():
    h = hash_password("s3cret!")
    assert verify_password("nope", h) is False


def test_verify_malformed_hash_is_false_not_error():
    # A corrupted/empty hash must return False, never raise.
    assert verify_password("anything", "") is False
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_distinct_salts_per_hash():
    assert hash_password("same") != hash_password("same")
