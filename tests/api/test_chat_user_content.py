"""_build_user_content wraps tagged text like the live Streamlit app and builds
multimodal content for image attachments. DB-free."""

from types import SimpleNamespace

from api.services.chat_service import _build_user_content


def test_no_attached_text_no_attachments_returns_bare_message():
    # byte-identical to today's user turn (the live-app-safe invariant)
    assert _build_user_content("who is hot?", None, None) == "who is hot?"


def test_attached_text_wraps_like_streamlit():
    out = _build_user_content("is this good?", "Mike Trout .312 AVG", None)
    assert out == "[Context the user selected on the page]\nMike Trout .312 AVG\n\n[Question]\nis this good?"


def test_image_attachment_builds_multimodal_list():
    att = SimpleNamespace(kind="image", data_url="data:image/png;base64,AAA")
    out = _build_user_content("what's wrong here?", None, [att])
    assert isinstance(out, list)
    assert out[0] == {"type": "text", "text": "what's wrong here?"}
    assert out[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


def test_text_plus_image_wraps_text_then_image():
    att = SimpleNamespace(kind="image", data_url="data:image/png;base64,BBB")
    out = _build_user_content("explain", "ERA 5.40", [att])
    assert out[0]["text"].startswith("[Context the user selected on the page]\nERA 5.40")
    assert out[1]["image_url"]["url"] == "data:image/png;base64,BBB"


def test_non_image_attachment_is_skipped():
    bad = SimpleNamespace(kind="audio", data_url="data:audio/x")
    # no usable image -> falls back to the plain text string
    assert _build_user_content("hi", None, [bad]) == "hi"
