from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dexter.tools.yfinance.fundamentals import yf_search_line_items
from dexter.tools.yfinance.news import yf_get_news
from dexter.tools.yfinance.prices import yf_get_prices, yf_get_price_snapshot
from dexter.tools.yfinance.insider import yf_get_insider_trades


class BaseFinancialAgent(ABC):
    """
    Abstract base class for financial analysis agents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the agent/persona."""
        pass

    @property
    @abstractmethod
    def required_line_items(self) -> List[str]:
        """List of financial line items required by this agent."""
        pass

    @abstractmethod
    def analyze(
        self,
        ticker: str,
        financials: List[Dict[str, Any]],
        prices: List[Dict[str, Any]],
        market_cap: Optional[float],
        insider_trades: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Perform the analysis for a single ticker given the data.
        """
        pass


def run_financial_analysis(
    tickers: List[str], agents: List[BaseFinancialAgent]
) -> Dict[str, Dict[str, Any]]:
    """
    Orchestrates data fetching and analysis for multiple agents.
    Optimizes data fetching by aggregating requirements.
    """
    # Aggregate required line items
    all_line_items = set()
    for agent in agents:
        all_line_items.update(agent.required_line_items)

    # Ensure we have basic items if not requested
    # (Though yf_search_line_items handles this, explicit is better)

    results = {}

    for ticker in tickers:
        ticker_results = {}

        # 1. Fetch Financials
        financial_line_items = yf_search_line_items(
            ticker=ticker,
            line_items=list(all_line_items),
            period="annual",
            limit=4,
        )

        # 2. Fetch Prices
        end_date = datetime.now().date().isoformat()
        start_date = (datetime.now() - timedelta(days=365)).date().isoformat()
        prices_data = yf_get_prices.invoke(
            {
                "ticker": ticker,
                "interval": "day",
                "interval_multiplier": 1,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        prices = prices_data.get("prices", [])

        # 3. Fetch Market Cap
        snapshot_data = yf_get_price_snapshot.invoke({"tickers": [ticker]})
        market_cap = snapshot_data.get(ticker, {}).get("snapshot", {}).get("market_cap")

        # 4. Fetch Insider Trades
        insider_trades = yf_get_insider_trades(
            ticker=ticker, end_date=end_date, start_date=start_date, limit=50
        )

        # 5. Fetch News
        company_news = yf_get_news.invoke(
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "limit": 50,
            }
        )
        news_items = company_news.get("news", [])

        # 6. Run Agents
        for agent in agents:
            ticker_results[agent.name] = agent.analyze(
                ticker=ticker,
                financials=financial_line_items,
                prices=prices,
                market_cap=market_cap,
                insider_trades=insider_trades,
                news=news_items,
            )

        results[ticker] = ticker_results

    return results
