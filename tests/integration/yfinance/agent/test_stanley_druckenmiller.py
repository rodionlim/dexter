import pytest
import json

from dexter.tools.yfinance.agent.stanley_druckenmiller import (
    stanley_druckenmiller_agent,
)


@pytest.mark.integration
def test_stanley_druckenmiller_agent_integration():
    """
    Integration test for stanley_druckenmiller_agent.
    Fetches real data for D05.SI (DBS Group Holdings) and verifies the analysis output.
    """
    ticker = "D05.SI"
    tickers = ["QCOM", "D05.SI"]

    results = stanley_druckenmiller_agent.invoke({"tickers": tickers})

    print("\n" + "=" * 50)
    print(json.dumps(results, indent=2))
    print("=" * 50 + "\n")

    assert isinstance(results, dict)
    assert ticker in results

    analysis = results[ticker]

    # Check structure
    expected_keys = [
        "signal",
        "score",
        "max_score",
        "growth_momentum_analysis",
        "risk_reward_analysis",
        "valuation_analysis",
    ]
    for key in expected_keys:
        assert key in analysis

    # Check values
    assert analysis["max_score"] == 10
    assert 0 <= analysis["score"] <= 10
    assert analysis["signal"] in ["Strong Buy", "Buy", "Hold", "Sell"]

    # Check sub-analyses
    growth = analysis["growth_momentum_analysis"]
    assert "score" in growth
    assert "details" in growth

    risk = analysis["risk_reward_analysis"]
    assert "score" in risk
    assert "details" in risk

    valuation = analysis["valuation_analysis"]
    assert "score" in valuation
    assert "details" in valuation

    insider = analysis["insider_activity"]
    assert "score" in insider
    assert "details" in insider
