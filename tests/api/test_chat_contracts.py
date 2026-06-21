from api.contracts.chat import ChatSendRequest, ChatSendResponse


def test_send_request_defaults():
    req = ChatSendRequest(message="hi", model="gpt-5")
    assert req.conversation_id is None and req.web_search is False and req.deep_research is False


def test_send_response_carries_error_optional():
    r = ChatSendResponse(content="ok", conversation_id=1, tokens_in=1, tokens_out=2, cost_usd=0.0)
    assert r.error is None


def test_send_request_reasoning_effort_defaults_none_and_accepts_levels():
    assert ChatSendRequest(message="hi", model="m").reasoning_effort is None
    assert ChatSendRequest(message="hi", model="m", reasoning_effort="high").reasoning_effort == "high"
