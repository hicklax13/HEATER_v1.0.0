from api.services.ai_allowance import managed_cap_for_tier


def test_pro_cap_exceeds_free_and_both_positive():
    assert managed_cap_for_tier("pro") > managed_cap_for_tier("free") > 0


def test_unknown_tier_gets_free_cap():
    assert managed_cap_for_tier("mystery") == managed_cap_for_tier("free")
