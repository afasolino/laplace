from request import PolicyRequest


def test_valid_policy_request_is_preserved() -> None:
    request = PolicyRequest(retries=3)
    assert request.retries == 3
