import pytest

from dexter.tools.yfinance.agent.stanley_druckenmiller import (
    stanley_druckenmiller_agent,
)


@pytest.mark.integration
def test_stanley_druckenmiller_agent_integration():
    """
    Integration test for stanley_druckenmiller_agent.
    Fetches real data for D05.SI (DBS Group Holdings) and verifies the analysis output.
    """
    ticker_to_check = "D05.SI"
    tickers = ["D05.SI", "C09.SI"]  # DBS Group Holdings and City Developments Limited

    results = stanley_druckenmiller_agent(tickers)

    assert isinstance(results, dict)
    assert ticker_to_check in results

    analysis = results[ticker_to_check]

    # Check structure
    expected_keys = [
        "signal",
        "score",
        "max_score",
        "growth_momentum_analysis",
        "risk_reward_analysis",
    ]
    for key in expected_keys:
        assert key in analysis

    # Check values
    assert analysis["max_score"] == 20
    assert 0 <= analysis["score"] <= 20
    assert analysis["signal"] in ["Strong Buy", "Buy", "Hold", "Sell"]

    # Check sub-analyses
    growth = analysis["growth_momentum_analysis"]
    assert "score" in growth
    assert "details" in growth

    risk = analysis["risk_reward_analysis"]
    assert "score" in risk
    assert "details" in risk
