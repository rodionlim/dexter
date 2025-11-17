"""Web search tools backed by the Tavily API."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from langchain.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


def _get_tavily_tool() -> TavilySearch:
    """Initialize Tavily search tool with API key from environment."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Please add your Tavily API key to the .env file."
        )
    return TavilySearch(max_results=3)


# Initialize the Tavily search tool once for reuse
_tavily_tool = None


def _ensure_tavily_initialized() -> TavilySearch:
    """Lazy initialization of Tavily tool."""
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = _get_tavily_tool()
    return _tavily_tool


class SocialMediaSentimentInput(BaseModel):
    """Input schema for social media sentiment search."""

    ticker: str = Field(
        description="Stock ticker symbol or description (e.g., 'AAPL', 'MSFT', 'UOL u14.si')"
    )
    trade_date: str = Field(
        description="Date to search around in ISO format (YYYY-MM-DD)"
    )


class MacroeconomicNewsInput(BaseModel):
    """Input schema for macroeconomic news search."""

    trade_date: str = Field(description="Date to search for in ISO format (YYYY-MM-DD)")


class CompanyNewsInput(BaseModel):
    """Input schema for company-specific news search."""

    ticker: str = Field(
        description="Stock ticker symbol or description (e.g., 'AAPL', 'MSFT', 'UOL u14.si')"
    )
    query_context: Optional[str] = Field(
        default=None,
        description="Optional context to refine the search (e.g., 'earnings', 'acquisition', 'product launch')",
    )


@tool(args_schema=SocialMediaSentimentInput)
def tavily_get_social_media_sentiment(ticker: str, trade_date: str) -> dict:
    """Perform a live web search for social media sentiment regarding a stock.

    Searches for discussions, sentiment, and opinions about a specific stock
    from social media platforms, forums, and online communities around a given date.
    Use this when you need to gauge market sentiment or retail investor opinions.

    Returns up to 3 relevant search results with titles, URLs, and content snippets.
    """
    tool = _ensure_tavily_initialized()

    # Determine if trade_date is recent (within 2 weeks)
    try:
        trade_dt = datetime.fromisoformat(trade_date)
        days_ago = (datetime.now() - trade_dt).days
        time_context = "recent" if days_ago <= 14 else f"around {trade_date}"
    except (ValueError, TypeError):
        time_context = "recent"

    query = (
        f"{ticker} stock sentiment investor opinion reddit stocktwits {time_context}"
    )
    result = tool.invoke({"query": query})

    return {
        "data_source": "tavily",
        "ticker": ticker.upper(),
        "trade_date": trade_date,
        "query": query,
        "results": result,
    }


@tool(args_schema=MacroeconomicNewsInput)
def tavily_get_macroeconomic_news(trade_date: str) -> dict:
    """Perform a live web search for macroeconomic news relevant to the stock market.

    Searches for broader market news, economic indicators, policy changes, and
    macroeconomic events that could affect the overall stock market on a specific date.
    Use this when you need context about market conditions or economic environment.

    Returns up to 3 relevant search results with titles, URLs, and content snippets.
    """
    tool = _ensure_tavily_initialized()

    # Determine if trade_date is recent (within 2 weeks)
    try:
        trade_dt = datetime.fromisoformat(trade_date)
        days_ago = (datetime.now() - trade_dt).days
        time_context = "recent" if days_ago <= 14 else f"around {trade_date}"
    except (ValueError, TypeError):
        time_context = "recent"

    query = f"stock market news economic trends Fed interest rates {time_context}"
    result = tool.invoke({"query": query})

    return {
        "data_source": "tavily",
        "trade_date": trade_date,
        "query": query,
        "results": result,
    }


@tool(args_schema=CompanyNewsInput)
def tavily_get_company_news(ticker: str, query_context: Optional[str] = None) -> dict:
    """Perform a live web search for recent news and developments about a specific company.

    Searches for company-specific news, announcements, press releases, and developments.
    Optionally provide context to refine the search (e.g., 'earnings', 'merger', 'lawsuit'). Keep optional context concise and strictly less than 400 characters.
    Use this when you need current information about a company that may not be in financial statements.

    Returns up to 3 relevant search results with titles, URLs, and content snippets.
    """
    tool = _ensure_tavily_initialized()

    if query_context:
        # More action-oriented query when context is provided
        query = f"{ticker} {query_context} latest news updates"
    else:
        # Broader query for general company news and catalysts
        query = f"{ticker} stock news earnings guidance analyst upgrades recent"

    result = tool.invoke({"query": query})

    return {
        "data_source": "tavily",
        "ticker": ticker.upper(),
        "query_context": query_context,
        "query": query,
        "results": result,
    }
