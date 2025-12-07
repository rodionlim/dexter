# This file makes the directory a Python package
import os

from langchain_core.tools import BaseTool

from dexter.tools.finance.filings import get_filings
from dexter.tools.finance.filings import get_10K_filing_items
from dexter.tools.finance.filings import get_10Q_filing_items
from dexter.tools.finance.filings import get_8K_filing_items
from dexter.tools.finance.fundamentals import get_income_statements
from dexter.tools.finance.fundamentals import get_balance_sheets
from dexter.tools.finance.fundamentals import get_cash_flow_statements
from dexter.tools.finance.fundamentals import get_all_financial_statements
from dexter.tools.finance.metrics import get_financial_metrics_snapshot
from dexter.tools.finance.metrics import get_financial_metrics
from dexter.tools.finance.prices import get_price_snapshot
from dexter.tools.finance.prices import get_prices
from dexter.tools.finance.news import get_news
from dexter.tools.finance.estimates import get_analyst_estimates
from dexter.tools.finance.segments import get_segmented_revenues
from dexter.tools.search.google import search_google_news
from dexter.tools.search.tavily import tavily_get_social_media_sentiment
from dexter.tools.search.tavily import tavily_get_macroeconomic_news
from dexter.tools.search.tavily import tavily_get_company_news

from dexter.tools.yfinance.filings import yf_get_filings
from dexter.tools.yfinance.filings import yf_get_10K_filing_items
from dexter.tools.yfinance.filings import yf_get_10Q_filing_items
from dexter.tools.yfinance.filings import yf_get_8K_filing_items
from dexter.tools.yfinance.fundamentals import yf_get_income_statements
from dexter.tools.yfinance.fundamentals import yf_get_balance_sheets
from dexter.tools.yfinance.fundamentals import yf_get_cash_flow_statements
from dexter.tools.yfinance.fundamentals import yf_get_comprehensive_financials
from dexter.tools.yfinance.metrics import yf_get_financial_metrics_snapshot
from dexter.tools.yfinance.metrics import yf_get_financial_metrics
from dexter.tools.yfinance.prices import yf_get_price_snapshot
from dexter.tools.yfinance.prices import yf_get_prices
from dexter.tools.yfinance.prices import yf_get_price_performance
from dexter.tools.yfinance.news import yf_get_news
from dexter.tools.yfinance.estimates import yf_get_analyst_estimates
from dexter.tools.yfinance.agent.stanley_druckenmiller import stanley_druckenmiller_agent

AVAILABLE_DATA_PROVIDERS = ["financialdatasets", "yfinance"]

tavily_tools = []
if os.environ.get("TAVILY_API_KEY"):
    tavily_tools = [
        tavily_get_social_media_sentiment,
        tavily_get_macroeconomic_news,
        tavily_get_company_news,
    ]

TOOLS: dict[str, list[BaseTool]] = {
    "financialdatasets": [
        get_income_statements,
        get_balance_sheets,
        get_cash_flow_statements,
        get_all_financial_statements,
        get_10K_filing_items,
        get_10Q_filing_items,
        get_8K_filing_items,
        get_filings,
        get_price_snapshot,
        get_prices,
        get_financial_metrics_snapshot,
        get_financial_metrics,
        get_news,
        get_analyst_estimates,
        get_segmented_revenues,
        search_google_news,
    ]
    + tavily_tools,
    "yfinance": [
        yf_get_comprehensive_financials,
        yf_get_income_statements,
        yf_get_balance_sheets,
        yf_get_cash_flow_statements,
        yf_get_10K_filing_items,
        yf_get_10Q_filing_items,
        yf_get_8K_filing_items,
        yf_get_filings,
        yf_get_price_snapshot,
        yf_get_prices,
        yf_get_price_performance,
        yf_get_financial_metrics_snapshot,
        yf_get_financial_metrics,
        yf_get_news,
        yf_get_analyst_estimates,
        stanley_druckenmiller_agent,
    ]
    + tavily_tools,
}
