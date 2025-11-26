import pytest

from dexter.tools.yfinance.fundamentals import yf_search_line_items


@pytest.mark.integration
def test_yf_search_line_items_integration():
    """
    Integration test for yf_search_line_items.
    Fetches real data for AAPL and verifies the structure and content.
    """
    ticker = "AAPL"
    line_items = [
        "revenue",
        "net_income",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
    ]
    limit = 2

    results = yf_search_line_items(
        ticker=ticker, line_items=line_items, period="annual", limit=limit
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= limit

    first_record = results[0]
    assert "period" in first_record
    assert isinstance(first_record["period"], (int, str))

    # Check that at least some of the requested fields are present
    # Note: Not all fields might be available for all years, but for AAPL revenue usually is.
    assert "revenue" in first_record
    assert first_record["revenue"] > 0

    # Check margins if they were calculated
    if "gross_margin" in first_record:
        assert (
            0 < first_record["gross_margin"] < 1
        )  # Margins are usually percentages (0.xx)
