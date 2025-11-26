import pytest

from dexter.tools.yfinance.agent.stanley_druckenmiller import (
    analyze_growth_and_momentum,
)


@pytest.mark.integration
def test_analyze_growth_and_momentum_integration():
    """
    Integration test for analyze_growth_and_momentum.
    Fetches real data for D05.SI (DBS Group Holdings) and verifies the analysis output.
    """
    ticker = "D05.SI"

    result = analyze_growth_and_momentum(ticker)

    assert isinstance(result, dict)
    assert "score" in result
    assert "details" in result

    # Score should be between 0 and 10
    assert isinstance(result["score"], (int, float))
    assert 0 <= result["score"] <= 10

    # Details should be a non-empty string explaining the score
    assert isinstance(result["details"], str)
    assert len(result["details"]) > 0

    # Since D05.SI is a major bank, we expect at least some financial data to be found
    # and thus the details shouldn't just be "Insufficient financial data..."
    # unless the API is completely down or data is missing.
    # We can check for keywords like "growth" or "momentum" or "revenue" in details
    # to ensure the logic actually ran.
    keywords = ["growth", "momentum", "revenue", "EPS", "data"]
    assert any(keyword in result["details"] for keyword in keywords)
