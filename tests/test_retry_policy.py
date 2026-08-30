from tools.error_classifier import classify_error
from tools.retry_policy import RetryPolicy


def test_timeout_is_retryable():
    result = classify_error(
        "TimeoutError: request timed out"
    )

    assert result.kind == "retryable"


def test_rate_limit_is_retryable():
    result = classify_error(
        "HTTP 429 Too Many Requests"
    )

    assert result.kind == "retryable"


def test_invalid_url_is_permanent():
    result = classify_error(
        "ValueError: invalid url"
    )

    assert result.kind == "permanent"


def test_unknown_error_is_unknown():
    result = classify_error(
        "Something completely unexpected happened"
    )

    assert result.kind == "unknown"


def test_retry_policy_allows_first_retry():
    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=2,
    )

    assert policy.can_retry(1) is True
    assert policy.can_retry(2) is True
    assert policy.can_retry(3) is False


def test_retry_policy_exponential_delay():
    policy = RetryPolicy(
        max_attempts=5,
        base_delay_seconds=2,
        max_delay_seconds=30,
    )

    assert policy.delay_for(1) == 2
    assert policy.delay_for(2) == 4
    assert policy.delay_for(3) == 8


def test_retry_policy_caps_delay():
    policy = RetryPolicy(
        max_attempts=10,
        base_delay_seconds=10,
        max_delay_seconds=30,
    )

    assert policy.delay_for(1) == 10
    assert policy.delay_for(2) == 20
    assert policy.delay_for(3) == 30
    assert policy.delay_for(4) == 30
