import pytest

from api.gateways.stripe_gateway import BillingSignatureError, NullStripeGateway


def test_null_gateway_is_unconfigured_and_refuses():
    g = NullStripeGateway()
    assert g.configured is False
    with pytest.raises(RuntimeError):
        g.create_customer("user_1", None)
    with pytest.raises(RuntimeError):
        g.create_checkout_session(
            customer_id="c", price_id="p", success_url="s", cancel_url="x", trial_days=7, clerk_user_id="user_1"
        )
    with pytest.raises(BillingSignatureError):
        g.parse_webhook_event(b"{}", "sig", "secret")
    with pytest.raises(RuntimeError):
        g.create_portal_session(customer_id="c", return_url="r")


def test_billing_signature_error_is_exception():
    assert issubclass(BillingSignatureError, Exception)
