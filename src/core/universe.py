"""
universe.py
Builds and validates the trading universe for both tracks (Europe + Global),
and saves a dated snapshot for reproducible backtesting. Sector data comes
from the Wikipedia tables used for the Global ticker lists (which carry
their own classification columns -- GICS for S&P 500, ICB for Nasdaq-100)
and from a manual mapping for the hand-curated Europe list. Everything is
normalized onto GICS sector names so the portfolio's sector cap compares
like with like regardless of where a ticker's data came from.
"""

import json
import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Track A: Europe (EURO STOXX 50 + AEX), Yahoo Finance ticker format.
# The last 5 are AEX members not fully confirmed for 2026 -- validation
# below will drop any that don't actually resolve.
EUROPE_TICKERS = [
    "ASML.AS", "MC.PA", "TTE.PA", "SAP.DE", "SIE.DE", "SAN.PA", "OR.PA",
    "SU.PA", "AI.PA", "ALV.DE", "AIR.PA", "RMS.PA", "IBE.MC", "DTE.DE",
    "DG.PA", "BNP.PA", "MBG.DE", "SAN.MC", "SAF.PA", "EL.PA", "CS.PA",
    "BAYN.DE", "IFX.DE", "PRX.AS", "ABI.BR", "ENEL.MI", "MUV2.DE",
    "INGA.AS", "ADYEN.AS", "DHL.DE", "BBVA.MC", "BAS.DE", "RI.PA",
    "ISP.MI", "ITX.MC", "KER.PA", "UCG.MI", "STLAM.MI", "CRH",
    "NDA-FI.HE", "BMW.DE", "BN.PA", "FLUT", "DB1.DE", "ENI.MI",
    "AD.AS", "ADS.DE", "VOW3.DE", "NOKIA.HE", "VNA.DE",
    "SHELL.AS", "UNA.AS", "REN.AS", "ASM.AS", "ABN.AS", "MT.AS", "NN.AS",
    "HEIA.AS", "DSFIR.AS", "PHIA.AS", "WKL.AS", "UMG.AS", "KPN.AS",
    "BESI.AS", "ASRNL.AS",
    "AGN.AS", "AKZA.AS", "EXO.AS", "IMCD.AS", "RAND.AS",
]

# Manual sector mapping for the Europe list, in the same Yahoo-style
# wording used elsewhere -- normalize_sector() below converts these
# (and anything else) onto GICS names before they're saved.
EUROPE_SECTORS = {
    "ASML.AS": "Technology", "MC.PA": "Consumer Cyclical", "TTE.PA": "Energy",
    "SAP.DE": "Technology", "SIE.DE": "Industrials", "SAN.PA": "Healthcare",
    "OR.PA": "Consumer Defensive", "SU.PA": "Industrials", "AI.PA": "Basic Materials",
    "ALV.DE": "Financial Services", "AIR.PA": "Industrials", "RMS.PA": "Consumer Cyclical",
    "IBE.MC": "Utilities", "DTE.DE": "Communication Services", "DG.PA": "Industrials",
    "BNP.PA": "Financial Services", "MBG.DE": "Consumer Cyclical", "SAN.MC": "Financial Services",
    "SAF.PA": "Industrials", "EL.PA": "Healthcare", "CS.PA": "Financial Services",
    "BAYN.DE": "Healthcare", "IFX.DE": "Technology", "PRX.AS": "Consumer Cyclical",
    "ABI.BR": "Consumer Defensive", "ENEL.MI": "Utilities", "MUV2.DE": "Financial Services",
    "INGA.AS": "Financial Services", "ADYEN.AS": "Technology", "DHL.DE": "Industrials",
    "BBVA.MC": "Financial Services", "BAS.DE": "Basic Materials", "RI.PA": "Consumer Defensive",
    "ISP.MI": "Financial Services", "ITX.MC": "Consumer Cyclical", "KER.PA": "Consumer Cyclical",
    "UCG.MI": "Financial Services", "STLAM.MI": "Consumer Cyclical", "CRH": "Basic Materials",
    "NDA-FI.HE": "Financial Services", "BMW.DE": "Consumer Cyclical", "BN.PA": "Consumer Defensive",
    "FLUT": "Consumer Cyclical", "DB1.DE": "Financial Services", "ENI.MI": "Energy",
    "AD.AS": "Consumer Defensive", "ADS.DE": "Consumer Cyclical", "VOW3.DE": "Consumer Cyclical",
    "NOKIA.HE": "Technology", "VNA.DE": "Real Estate", "SHELL.AS": "Energy",
    "UNA.AS": "Consumer Defensive", "REN.AS": "Industrials", "ASM.AS": "Technology",
    "ABN.AS": "Financial Services", "MT.AS": "Basic Materials", "NN.AS": "Financial Services",
    "HEIA.AS": "Consumer Defensive", "DSFIR.AS": "Basic Materials", "PHIA.AS": "Healthcare",
    "WKL.AS": "Industrials", "UMG.AS": "Communication Services", "KPN.AS": "Communication Services",
    "BESI.AS": "Technology", "ASRNL.AS": "Financial Services", "AGN.AS": "Financial Services",
    "AKZA.AS": "Basic Materials", "EXO.AS": "Financial Services", "IMCD.AS": "Basic Materials",
    "RAND.AS": "Industrials",
}

# Maps whatever a source calls a sector (Yahoo-style, or ICB's "Industry"
# wording from the Nasdaq-100 table) onto GICS's official sector names.
# GICS names map to themselves, so this is safe to apply to every source.
SECTOR_NORMALIZATION = {
    "Energy": "Energy",
    "Materials": "Materials",
    "Industrials": "Industrials",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Health Care": "Health Care",
    "Financials": "Financials",
    "Information Technology": "Information Technology",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Technology": "Information Technology",
    "Telecommunications": "Communication Services",
    "Basic Materials": "Materials",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
}


def normalize_sector(raw: str) -> str:
    return SECTOR_NORMALIZATION.get(raw, raw)


# Manual additions to the Global track beyond S&P 500 + Nasdaq-100
# (e.g. very recent IPOs not yet in either index).
# Note: a stock with only days/weeks of price history won't have enough
# data yet for the 50-day trend filter or ATR(14) stop to compute --
# it'll sit inactive in the universe until enough trading days accumulate.
MANUAL_GLOBAL_ADDITIONS = [
    # "ANT",  # Anthropic -- uncomment to include
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_sp500_tickers() -> tuple:
    """Scrape the current S&P 500 constituents and their GICS sectors from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0]
    tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
    sectors = dict(zip(tickers, sp500["GICS Sector"]))
    return tickers, sectors


def fetch_nasdaq100_tickers() -> tuple:
    """Scrape the current Nasdaq-100 constituents and their sector classification from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        cols = list(table.columns)
        ticker_col = next((c for c in cols if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
        if ticker_col is not None:
            tickers = table[ticker_col].astype(str).str.replace(".", "-", regex=False).tolist()
            # Prefer a broad "industry"/"sector" column over a more granular
            # "sub-sector"/"sub-industry" one -- e.g. the Nasdaq-100 table's
            # 'ICB Subsector[2]' contains "sector" as a substring and would
            # otherwise get matched ahead of the correct 'ICB Industry[2]'.
            sector_col = next(
                (c for c in cols
                 if ("sector" in str(c).lower() or "industry" in str(c).lower())
                 and "sub" not in str(c).lower()),
                None
            )
            sectors = dict(zip(tickers, table[sector_col])) if sector_col is not None else {}
            return tickers, sectors
    print("Could not auto-detect the ticker column. Tables found:")
    for i, table in enumerate(tables):
        print(f"  Table {i}: columns = {list(table.columns)}, rows = {len(table)}")
    raise ValueError("Could not find Nasdaq-100 constituents table")


def validate_tickers(tickers: list) -> tuple:
    """Try pulling a bit of data for each ticker; split into valid/invalid."""
    valid, invalid = [], []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d")
            (valid if not hist.empty else invalid).append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid


def build_universe_snapshot():
    print("Fetching S&P 500 list from Wikipedia...")
    sp500, sp500_sectors = fetch_sp500_tickers()

    print("Fetching Nasdaq-100 list from Wikipedia...")
    nasdaq100, nasdaq100_sectors = fetch_nasdaq100_tickers()

    global_tickers = sorted(set(sp500) | set(nasdaq100) | set(MANUAL_GLOBAL_ADDITIONS))
    print(f"Combined Global universe: {len(global_tickers)} unique tickers")
    global_sectors = {**sp500_sectors, **nasdaq100_sectors}

    print(f"Validating {len(EUROPE_TICKERS)} Europe tickers...")
    europe_valid, europe_invalid = validate_tickers(EUROPE_TICKERS)

    print(f"Validating {len(global_tickers)} Global tickers (this will take a few minutes)...")
    global_valid, global_invalid = validate_tickers(global_tickers)

    sectors = {}
    for t in europe_valid:
        sectors[t] = normalize_sector(EUROPE_SECTORS.get(t, "Unknown"))
    for t in global_valid:
        sectors[t] = normalize_sector(global_sectors.get(t, "Unknown"))

    snapshot = {
        "date": datetime.date.today().isoformat(),
        "europe": sorted(europe_valid),
        "global": sorted(global_valid),
        "sectors": sectors,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = DATA_DIR / f"universe_{snapshot['date']}.json"
    with open(filename, "w") as f:
        json.dump(snapshot, f, indent=2)

    unknown_count = sum(1 for s in sectors.values() if s == "Unknown")
    print(f"\nSaved snapshot to {filename}")
    print(f"Europe: {len(europe_valid)} valid, {len(europe_invalid)} invalid -> {europe_invalid}")
    print(f"Global: {len(global_valid)} valid, {len(global_invalid)} invalid -> {global_invalid}")
    print(f"Sectors resolved: {len(sectors) - unknown_count}/{len(sectors)}, {unknown_count} Unknown")


if __name__ == "__main__":
    build_universe_snapshot()