"""Financial statement tools backed by the yfinance API."""

from __future__ import annotations

from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from dexter.tools.finance.fundamentals import FinancialStatementsInput
from dexter.tools.yfinance.shared import (
    apply_period_filters,
    frame_to_records,
    get_ticker,
    limit_records,
    load_statement_frame,
)


def _prepare_response(
    ticker: str,
    period: Literal["annual", "quarterly", "ttm"],
    frame,
    limit: int,
    report_period_gt: Optional[str],
    report_period_gte: Optional[str],
    report_period_lt: Optional[str],
    report_period_lte: Optional[str],
    statement_label: str,
) -> dict:
    records = frame_to_records(frame)
    records = apply_period_filters(
        records,
        report_period_gt,
        report_period_gte,
        report_period_lt,
        report_period_lte,
    )
    records = limit_records(records, limit)
    return {
        "data_source": "yfinance",
        "ticker": ticker.upper(),
        "statement": statement_label,
        "period": period,
        "results": records,
    }


@tool(args_schema=FinancialStatementsInput)
def yf_get_income_statements(
    ticker: str,
    period: Literal["annual", "quarterly", "ttm"],
    limit: int = 10,
    report_period_gt: Optional[str] = None,
    report_period_gte: Optional[str] = None,
    report_period_lt: Optional[str] = None,
    report_period_lte: Optional[str] = None,
) -> dict:
    """Return structured income statement records from Yahoo Finance via yfinance.

    Use this when the agent needs revenue, profit, or expense line items without
    hitting the FinancialDatasets API. The `period` flag selects `annual`,
    `quarterly`, or trailing-twelve-month (`ttm`) data and respects optional ISO
    date filters plus a maximum number of periods (`limit`).
    Yahoo Finance may not have extensive historical annual/quarterly/ttm data for all tickers.
    """
    ticker_obj = get_ticker(ticker)
    frame = load_statement_frame(ticker_obj, "income_stmt", period)
    return _prepare_response(
        ticker,
        period,
        frame,
        limit,
        report_period_gt,
        report_period_gte,
        report_period_lt,
        report_period_lte,
        "income_statement",
    )


@tool(args_schema=FinancialStatementsInput)
def yf_get_balance_sheets(
    ticker: str,
    period: Literal["annual", "quarterly", "ttm"],
    limit: int = 10,
    report_period_gt: Optional[str] = None,
    report_period_gte: Optional[str] = None,
    report_period_lt: Optional[str] = None,
    report_period_lte: Optional[str] = None,
) -> dict:
    """Return balance sheet snapshots from Yahoo Finance via yfinance.

    Includes assets, liabilities, and equity lines so the agent can analyse
    capital structure when the yfinance provider is selected. Supports the same
    period selection and ISO date filtering options as the income statement
    tool.
    Yahoo Finance may not have extensive historical annual/quarterly/ttm data for all tickers.
    """
    ticker_obj = get_ticker(ticker)
    frame = load_statement_frame(ticker_obj, "balance_sheet", period)
    return _prepare_response(
        ticker,
        period,
        frame,
        limit,
        report_period_gt,
        report_period_gte,
        report_period_lt,
        report_period_lte,
        "balance_sheet",
    )


@tool(args_schema=FinancialStatementsInput)
def yf_get_cash_flow_statements(
    ticker: str,
    period: Literal["annual", "quarterly", "ttm"],
    limit: int = 10,
    report_period_gt: Optional[str] = None,
    report_period_gte: Optional[str] = None,
    report_period_lt: Optional[str] = None,
    report_period_lte: Optional[str] = None,
) -> dict:
    """Return cash flow statement entries (operating, investing, financing) using yfinance.

    Ideal for questions about liquidity, free cash flow, or capital allocation
    when the agent works in `yfinance` mode. Mirrors the other financial
    statement tools in supported arguments and response shape.
    Yahoo Finance may not have extensive historical annual/quarterly/ttm data for all tickers.
    """
    ticker_obj = get_ticker(ticker)
    frame = load_statement_frame(ticker_obj, "cashflow", period)
    return _prepare_response(
        ticker,
        period,
        frame,
        limit,
        report_period_gt,
        report_period_gte,
        report_period_lt,
        report_period_lte,
        "cash_flow_statement",
    )


class ComprehensiveFinancialsInput(BaseModel):
    """Input schema for comprehensive financial statements retrieval."""

    ticker: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')")
    include_quarterly: bool = Field(
        default=True,
        description="Whether to include quarterly statements (typically last 8 quarters)",
    )
    include_annual: bool = Field(
        default=True,
        description="Whether to include annual statements (typically last 5 years)",
    )
    include_ttm: bool = Field(
        default=True, description="Whether to include trailing-twelve-month data"
    )
    quarterly_limit: int = Field(
        default=8, description="Number of quarterly periods to retrieve"
    )
    annual_limit: int = Field(
        default=5, description="Number of annual periods to retrieve"
    )


@tool(args_schema=ComprehensiveFinancialsInput)
def yf_get_comprehensive_financials(
    ticker: str,
    include_quarterly: bool = True,
    include_annual: bool = True,
    include_ttm: bool = True,
    quarterly_limit: int = 8,
    annual_limit: int = 5,
) -> dict:
    """Retrieve ALL financial statements (income, balance sheet, cash flow) for multiple periods in ONE efficient call.

    **CRITICAL**: This is the MOST TOKEN-EFFICIENT way to get comprehensive financial data.
    Always prefer this tool over making multiple separate calls to yf_get_income_statements,
    yf_get_balance_sheets, and yf_get_cash_flow_statements when you need:
    - Multiple statement types (income + balance sheet + cash flow)
    - Multiple periods (quarterly + annual + TTM)
    - Complete financial picture for analysis

    This single call replaces what would otherwise be 6-9 separate tool calls:
    - 3 calls for quarterly data (income, balance, cash flow)
    - 3 calls for annual data
    - 3 calls for TTM data

    Returns a comprehensive dataset containing:
    - Income statements (revenue, expenses, margins)
    - Balance sheets (assets, liabilities, equity)
    - Cash flow statements (operating, investing, financing activities)

    Use the individual statement tools (yf_get_income_statements, etc.) ONLY when:
    - User asks specifically for ONE statement type
    - You need very specific date filtering
    - Query is narrowly focused on a single metric

    Token savings: ~70-80% compared to multiple individual calls.
    """
    ticker_obj = get_ticker(ticker)
    result = {
        "data_source": "yfinance",
        "ticker": ticker.upper(),
        "statements": {},
    }

    periods_to_fetch = []
    if include_quarterly:
        periods_to_fetch.append(("quarterly", quarterly_limit))
    if include_annual:
        periods_to_fetch.append(("annual", annual_limit))
    if include_ttm:
        periods_to_fetch.append(("ttm", 1))

    statement_types = [
        ("income_stmt", "income_statement"),
        ("balance_sheet", "balance_sheet"),
        ("cashflow", "cash_flow_statement"),
    ]

    for period, limit in periods_to_fetch:
        result["statements"][period] = {}
        for attr_name, label in statement_types:
            frame = load_statement_frame(ticker_obj, attr_name, period)
            records = frame_to_records(frame, limit=limit if period != "ttm" else None)
            result["statements"][period][label] = records

    return result
