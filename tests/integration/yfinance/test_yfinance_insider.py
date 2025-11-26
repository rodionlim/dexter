import pytest

from datetime import datetime, timedelta
from dexter.tools.yfinance.insider import yf_get_insider_trades


@pytest.mark.integration
def test_yf_get_insider_trades_integration():
    """
    Integration test for yf_get_insider_trades.
    Fetches real insider trade data for a major ticker and verifies structure.
    """
    ticker = "MSFT"  # MSFT usually has regular insider activity

    # Set a wide enough range to ensure we get some data
    end_date = datetime.now().date().isoformat()
    start_date = (datetime.now() - timedelta(days=365)).date().isoformat()
    limit = 5

    results = yf_get_insider_trades(
        ticker=ticker, start_date=start_date, end_date=end_date, limit=limit
    )

    assert isinstance(results, list)

    # Note: It's possible (though unlikely for MSFT in a year) to have 0 trades.
    # If we get results, verify their structure.
    if len(results) > 0:
        assert len(results) <= limit

        trade = results[0]
        expected_keys = {
            "ticker",
            "issuer",
            "name",
            "title",
            "is_board_director",
            "transaction_date",
            "transaction_shares",
            "transaction_price_per_share",
            "transaction_value",
            "shares_owned_before_transaction",
            "shares_owned_after_transaction",
            "security_title",
            "filing_date",
        }
        assert expected_keys.issubset(trade.keys())

        # Verify date filtering
        if trade["transaction_date"]:
            trade_date = datetime.fromisoformat(trade["transaction_date"]).date()
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
            assert start_dt <= trade_date <= end_dt

        # Verify types
        if trade["transaction_shares"] is not None:
            assert isinstance(trade["transaction_shares"], (int, float))
        if trade["transaction_value"] is not None:
            assert isinstance(trade["transaction_value"], (int, float))

        assert trade["ticker"] == ticker
