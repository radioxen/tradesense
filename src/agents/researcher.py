"""Researcher Agent for market scanning.

Scans the market for day-trading candidates likely to move in 0-3 days.
Uses a multi-stage approach to minimize token usage:
1. Dynamic universe discovery (Perplexity + index components)
2. Lightweight price/volume scoring for intraday momentum
3. LLM + Perplexity for catalyst/news ranking on top candidates
"""

from datetime import datetime, timedelta
from typing import Any
from dataclasses import dataclass, field
import json
import re

import pandas as pd
import numpy as np

from src.data.market_data import get_provider
from src.data.feature_store import FeatureBuilder
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ScanResult:
    """Result of market scan."""
    symbol: str
    score: float  # 0-100 composite score
    momentum_score: float
    volume_score: float
    technical_score: float
    catalyst_score: float  # From news/LLM
    sector: str
    market_cap: str
    price: float
    avg_volume: float
    rsi: float
    macd_signal: str
    volume_surge: float  # % above average
    short_term_trend: str  # "bullish", "bearish", "neutral"
    gap_pct: float = 0.0
    atr_percent: float = 0.0
    volatility_score: float = 0.0
    catalysts: list[str] = field(default_factory=list)
    reasoning: str = ""


class ResearcherAgent:
    """Market Researcher Agent.
    
    Scans the market to find stocks with high probability of
    price increase in 2-5 day horizon. Uses tiered approach:
    
    Stage 1: Universe filtering via free screeners
    Stage 2: Technical momentum scoring (no API cost)
    Stage 3: LLM + Perplexity for catalyst detection (token-efficient)
    """

    # Stock universes to scan
    UNIVERSES = {
        "sp500": "S&P 500 large caps",
        "nasdaq100": "NASDAQ 100 tech leaders",
        "russell2000": "Small cap growth",
        "momentum": "High momentum stocks",
        "volume_leaders": "Unusual volume",
    }

    # Day-trading filters (tunable via config)
    DAY_TRADING_FILTERS = {
        "min_price": 3.0,
        "max_price": 1000.0,
        "min_avg_volume": 1_000_000,
        "min_dollar_volume": 20_000_000,
        "min_atr_pct": 1.0,
    }

    DEFAULT_UNIVERSE_SIZE = 300
    DEFAULT_CANDIDATE_POOL = 250

    # Technical thresholds for momentum
    MOMENTUM_THRESHOLDS = {
        "rsi_oversold": 35,  # Bounce candidates
        "rsi_momentum": 55,  # Trending up
        "volume_surge": 1.5,  # 50% above average
        "macd_cross_days": 3,  # Recent MACD crossover
    }

    def __init__(
        self,
        openai_api_key: str | None = None,
        perplexity_api_key: str | None = None,
        max_llm_calls: int = 50,  # Limit LLM usage
        config: dict | None = None,
    ):
        """Initialize Researcher Agent.
        
        Args:
            openai_api_key: OpenAI API key.
            perplexity_api_key: Perplexity API key.
            max_llm_calls: Maximum LLM API calls per scan.
            config: Additional configuration.
        """
        self.name = "researcher"
        self.openai_api_key = openai_api_key
        self.perplexity_api_key = perplexity_api_key
        self.max_llm_calls = max_llm_calls
        self.config = config or {}
        self.feature_builder = FeatureBuilder()
        self.universe_size = self.config.get("scanner_universe_size", self.DEFAULT_UNIVERSE_SIZE)
        self.candidate_pool = self.config.get("scanner_candidate_pool", self.DEFAULT_CANDIDATE_POOL)
        custom_filters = self.config.get("scanner_day_trading_filters", {})
        self.day_trading_filters = {**self.DAY_TRADING_FILTERS, **custom_filters}

    def _get_system_prompt(self) -> str:
        return """You are a market research analyst specializing in day trading (0-3 day horizon).
Your job is to identify US stocks likely to move 5-20% within the next 1-3 sessions.

Prioritize:
- Premarket gappers and unusual volume
- Earnings within 7 days, guidance, FDA/SEC events
- Short-interest/option-flow driven squeeze risk
- Social media buzz and sentiment shifts
- Sector rotation and fresh news catalysts

Be concise, skeptical, and output JSON only."""

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace(".", "-").upper().strip()

    def _extract_symbols_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        candidates = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", text)
        stopwords = {
            "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WILL", "BEAR",
            "BULL", "BUY", "SELL", "HOLD", "OVER", "UNDER", "MORE", "LESS", "NEXT",
            "DAYS", "WEEK", "NEWS", "UP", "DOWN",
        }
        symbols = []
        seen = set()
        for raw in candidates:
            normalized = self._normalize_symbol(raw)
            if normalized in stopwords or len(normalized) < 1:
                continue
            if normalized not in seen:
                seen.add(normalized)
                symbols.append(normalized)
        return symbols

    async def _perplexity_chat(self, user_prompt: str, max_tokens: int = 1000) -> str | None:
        if not self.perplexity_api_key:
            return None
        import aiohttp

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Perplexity request failed: {e}")
            return None

    async def scan_market(
        self,
        target_count: int = 500,
        use_llm: bool = True,
        universe_size: int | None = None,
        candidate_pool: int | None = None,
    ) -> list[ScanResult]:
        """Scan market for momentum candidates.
        
        Args:
            target_count: Target number of stocks to return.
            use_llm: Whether to use LLM for catalyst detection.
            
        Returns:
            List of scan results, sorted by score.
        """
        logger.info("Starting market scan...")
        
        # Stage 1: Get broad universe (200-300 tickers)
        universe_size = universe_size or self.universe_size
        candidate_pool = candidate_pool or self.candidate_pool
        candidate_pool = max(candidate_pool, target_count)

        universe = await self._get_universe(
            max_symbols=universe_size,
            use_llm=use_llm,
        )
        logger.info(f"Stage 1: Universe of {len(universe)} symbols")
        
        # Stage 2: Technical filtering
        scored = await self._score_technical(universe)
        scored = sorted(scored, key=lambda x: x.score, reverse=True)
        
        # Keep top candidates for LLM analysis
        top_candidates = scored[:min(len(scored), candidate_pool)]
        logger.info(f"Stage 2: {len(top_candidates)} technical candidates")
        
        # Stage 3: LLM catalyst detection (only on top stocks)
        if use_llm and self.perplexity_api_key:
            top_candidates = await self._add_catalyst_scores(top_candidates)
            top_candidates = sorted(top_candidates, key=lambda x: x.score, reverse=True)
        
        # Return top N
        results = top_candidates[:target_count]
        logger.info(f"Stage 3: Returning {len(results)} final candidates")
        
        return results

    async def _get_universe(
        self,
        max_symbols: int = 300,
        use_llm: bool = True,
    ) -> list[str]:
        """Get universe of stocks to scan.
        
        Uses free data sources to get comprehensive list.
        """
        symbols: list[str] = []
        seen: set[str] = set()

        def add(items: list[str]):
            for item in items:
                normalized = self._normalize_symbol(item)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                symbols.append(normalized)

        if use_llm and self.perplexity_api_key:
            trending = await self._discover_trending_symbols(max_symbols=min(200, max_symbols))
            add(trending)
        
        # S&P 500
        try:
            sp500 = await self._fetch_index_components("SPY")
            add(sp500)
        except Exception as e:
            logger.warning(f"Failed to get S&P 500: {e}")
        
        # NASDAQ 100
        try:
            nasdaq = await self._fetch_index_components("QQQ")
            add(nasdaq)
        except Exception as e:
            logger.warning(f"Failed to get NASDAQ 100: {e}")
        
        # Add popular/liquid stocks
        popular = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX",
            "CRM", "ADBE", "INTC", "PYPL", "SQ", "SHOP", "ROKU", "SNAP", "UBER",
            "LYFT", "ABNB", "COIN", "HOOD", "PLTR", "SNOW", "DDOG", "NET", "ZS",
            "CRWD", "OKTA", "MDB", "PANW", "NOW", "TEAM", "WDAY", "SPLK", "DOCU",
            "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GM", "F", "TM", "HMC",
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "USB", "PNC",
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD", "BIIB",
            "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "HES",
        ]
        add(popular)

        extended = [
            "BA", "RTX", "LMT", "NOC", "GD", "GE", "MMM", "CAT", "DE", "HON",
            "T", "VZ", "TMUS", "CMCSA", "DIS", "PARA", "WBD", "SIRI",
            "KO", "PEP", "WMT", "COST", "TGT", "LOW", "HD", "DG", "KR", "SBUX",
            "MCD", "CMG", "YUM", "NKE", "LULU", "EL", "CL", "PG",
            "ACN", "ORCL", "IBM", "CSCO", "AVGO", "QCOM", "INTU", "TXN",
            "MRVL", "AMAT", "LRCX", "KLAC", "ADI", "MU", "ASML", "TSM",
            "SHOP", "SE", "MELI", "BABA", "JD", "PDD", "BIDU",
            "UAL", "DAL", "AAL", "LUV", "JBLU", "SAVE", "ALK",
            "CCL", "RCL", "NCLH",
            "CVS", "CI", "HUM", "UNH", "MRNA", "REGN", "VRTX", "ISRG",
            "SQ", "AFRM", "UPST", "SOFI", "LC", "PYPL", "COIN", "MARA", "RIOT",
            "X", "CLF", "FCX", "AA", "NUE", "STLD",
            "ET", "KMI", "PXD", "DVN", "MRO",
            "FSLR", "ENPH", "SEDG", "PLUG", "RUN",
            "DKNG", "PENN", "MGM", "WYNN", "CZR",
            "GME", "AMC", "BBBY", "BB", "NKLA", "SPCE",
        ]
        add(extended)
        
        # Filter to valid, liquid stocks
        return symbols[:max_symbols]

    async def _discover_trending_symbols(self, max_symbols: int = 150) -> list[str]:
        """Discover trending tickers via Perplexity."""
        prompt = f"""Return a JSON object with a single key "symbols" that is a list of
US stock tickers (max {max_symbols}) suitable for day trading in the next 1-3 days.

Sources to consider: premarket gainers/losers, unusual volume, heavy options flow,
short-interest squeeze risk, earnings in the next 7 days, FDA/SEC events, and
social-media buzz. Avoid ETFs and macro indexes. Output JSON only."""

        content = await self._perplexity_chat(prompt, max_tokens=800)
        if not content:
            return []

        # Try JSON parse
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON block if present
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = {}
            else:
                payload = {}

        symbols = payload.get("symbols") if isinstance(payload, dict) else None
        if isinstance(symbols, list):
            normalized = [self._normalize_symbol(s) for s in symbols if s]
            return normalized[:max_symbols]

        return self._extract_symbols_from_text(content)[:max_symbols]

    async def _fetch_index_components(self, etf: str) -> list[str]:
        """Fetch index components from ETF holdings."""
        # In production, would use API. For now, return common holdings.
        if etf == "SPY":
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK.B", "TSLA",
                "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
                "LLY", "ABBV", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
                "ACN", "TMO", "DHR", "ABT", "VZ", "ADBE", "CRM", "NEE", "CMCSA",
                "NKE", "PM", "INTC", "AMD", "TXN", "UPS", "RTX", "HON", "QCOM",
            ]
        elif etf == "QQQ":
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
                "COST", "ADBE", "PEP", "CSCO", "CMCSA", "NFLX", "AMD", "INTC",
                "TXN", "QCOM", "AMGN", "INTU", "ISRG", "HON", "AMAT", "BKNG",
                "SBUX", "GILD", "ADI", "ADP", "MDLZ", "LRCX", "PYPL", "REGN",
            ]
        return []

    async def _score_technical(
        self,
        symbols: list[str],
    ) -> list[ScanResult]:
        """Score stocks based on technical indicators.
        
        This stage uses only price/volume data (no API cost).
        """
        results = []
        provider = get_provider("yfinance")
        
        # Batch process for efficiency
        end = datetime.now()
        start = end - timedelta(days=60)
        
        for symbol in symbols:
            try:
                data = await provider.fetch_ohlcv(symbol, start, end, "1d")
                if data.empty or len(data) < 20:
                    continue
                
                # Build features
                features = self.feature_builder.build_features(data)
                if features.empty:
                    continue
                
                latest = features.iloc[-1]
                
                # Extract scores
                rsi = latest.get("rsi", 50)
                macd = latest.get("macd", 0)
                macd_signal = latest.get("macd_signal", 0)
                volume_ratio = latest.get("volume_ratio", 1.0)
                price = float(data.iloc[-1]["close"])
                avg_volume = float(data["volume"].tail(20).mean())
                atr_percent = float(latest.get("atr_percent", 0) or 0)

                filters = self.day_trading_filters
                if price < filters.get("min_price", 0) or price > filters.get("max_price", float("inf")):
                    continue
                if avg_volume < filters.get("min_avg_volume", 0):
                    continue
                if avg_volume * price < filters.get("min_dollar_volume", 0):
                    continue
                if atr_percent < filters.get("min_atr_pct", 0):
                    continue

                return_1 = float(latest.get("return_1", 0) or 0)
                return_5 = float(latest.get("return_5", 0) or 0)
                gap_pct = 0.0
                if len(data) >= 2:
                    prev_close = float(data.iloc[-2]["close"])
                    open_price = float(data.iloc[-1]["open"])
                    if prev_close:
                        gap_pct = (open_price - prev_close) / prev_close
                
                # Calculate component scores (0-100)
                # Momentum: recent returns + RSI sweet spot
                momentum_score = 50.0
                momentum_score += np.clip(return_1 * 1000, -20, 20)
                momentum_score += np.clip(return_5 * 500, -20, 20)
                if 45 <= rsi <= 70:
                    momentum_score += 20
                elif rsi < 30:
                    momentum_score += 10
                elif rsi > 75:
                    momentum_score -= 10
                momentum_score = float(np.clip(momentum_score, 0, 100))

                # Volume: relative volume + liquidity
                volume_score = float(np.clip((volume_ratio - 1) * 60 + 50, 0, 100))

                # Volatility: ATR percent (higher = more tradable)
                volatility_score = float(np.clip(atr_percent * 15, 0, 100))

                # Gap: premarket/overnight gap (magnitude)
                gap_score = float(np.clip(abs(gap_pct) * 20, 0, 100))

                # Breakout/mean reversion: 20-day range
                high_20 = float(data["high"].tail(20).max())
                low_20 = float(data["low"].tail(20).min())
                if price >= high_20 * 0.995:
                    breakout_score = 90
                elif price <= low_20 * 1.005:
                    breakout_score = 80
                else:
                    breakout_score = 50

                technical_score = breakout_score
                if macd > macd_signal:
                    technical_score += 10
                elif macd < macd_signal:
                    technical_score -= 10
                technical_score = float(np.clip(technical_score, 0, 100))

                # Short-term trend
                if return_5 > 0.05 and return_1 > 0:
                    trend = "bullish"
                elif return_5 < -0.05 and return_1 < 0:
                    trend = "bearish"
                else:
                    trend = "neutral"

                # Composite score (day-trading focus)
                score = (
                    momentum_score * 0.25 +
                    volume_score * 0.25 +
                    volatility_score * 0.20 +
                    gap_score * 0.15 +
                    technical_score * 0.15
                )
                
                results.append(ScanResult(
                    symbol=symbol,
                    score=score,
                    momentum_score=momentum_score,
                    volume_score=volume_score,
                    technical_score=technical_score,
                    catalyst_score=0,  # Added in Stage 3
                    sector="",  # Would need separate lookup
                    market_cap="",
                    price=price,
                    avg_volume=avg_volume,
                    rsi=rsi,
                    macd_signal="bullish" if macd > macd_signal else "bearish",
                    volume_surge=(volume_ratio - 1) * 100,
                    short_term_trend=trend,
                    gap_pct=gap_pct * 100,
                    atr_percent=atr_percent,
                    volatility_score=volatility_score,
                    reasoning=(
                        f"gap={gap_pct:+.1%}, rel_vol={volume_ratio:.2f}x, "
                        f"rsi={rsi:.1f}, atr%={atr_percent:.1f}"
                    ),
                ))
                
            except Exception as e:
                logger.debug(f"Failed to process {symbol}: {e}")
                continue
        
        return results

    async def _add_catalyst_scores(
        self,
        candidates: list[ScanResult],
    ) -> list[ScanResult]:
        """Add catalyst scores using Perplexity + GPT.
        
        This is the token-expensive stage, so we batch efficiently.
        """
        # Group into batches for efficient API usage
        batch_size = 20
        batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
        
        llm_calls = 0
        
        for batch in batches:
            if llm_calls >= self.max_llm_calls:
                logger.info(f"Reached LLM call limit ({self.max_llm_calls})")
                break
            
            # Query Perplexity for batch news
            symbols = [c.symbol for c in batch]
            try:
                catalysts = await self._query_catalysts_batch(symbols)
                llm_calls += 1
                
                # Update scores
                for candidate in batch:
                    cat = catalysts.get(candidate.symbol, {})
                    candidate.catalyst_score = cat.get("score", 0)
                    candidate.catalysts = cat.get("catalysts", [])
                    candidate.reasoning = cat.get("reasoning", "")
                    
                    # Update composite score with catalyst weight
                    candidate.score = min(
                        100.0,
                        candidate.score * 0.55 +  # Technical weight
                        candidate.catalyst_score * 0.45,  # Catalyst weight
                    )
                    
            except Exception as e:
                logger.warning(f"Catalyst query failed: {e}")
                continue
        
        return candidates

    async def _query_catalysts_batch(
        self,
        symbols: list[str],
    ) -> dict[str, dict]:
        """Query Perplexity for catalysts on batch of symbols."""
        import aiohttp
        
        prompt = f"""Analyze these stocks for near-term (0-3 day) catalysts: {', '.join(symbols)}

For each stock, identify:
1. Earnings within 7 days, guidance, FDA/SEC decisions, product launches
2. Premarket gappers or unusual volume
3. Social-media buzz or meme/short-squeeze risk
4. Options flow or analyst upgrades/downgrades

Return JSON format:
{{
  "SYMBOL": {{
    "score": 0-100,
    "catalysts": ["catalyst1", "catalyst2"],
    "reasoning": "brief explanation (1-2 sentences)"
  }}
}}

Only include stocks with score > 50. Output JSON only."""

        content = await self._perplexity_chat(prompt, max_tokens=1000)
        if not content:
            return {}

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            return {}

        results: dict[str, dict] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                symbol = self._normalize_symbol(key)
                if not symbol or not isinstance(value, dict):
                    continue
                score = value.get("score", 0)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0
                results[symbol] = {
                    "score": max(0, min(score, 100)),
                    "catalysts": value.get("catalysts", []) or [],
                    "reasoning": value.get("reasoning", "") or "",
                }

        return results

    async def get_daily_picks(
        self,
        count: int = 10,
    ) -> list[ScanResult]:
        """Get top daily picks.
        
        Convenience method for getting actionable picks.
        """
        all_results = await self.scan_market(
            target_count=max(count * 10, 200),
            use_llm=True,
        )
        
        # Filter for high-conviction picks
        picks = [
            r for r in all_results
            if r.score >= 70 and r.catalyst_score >= 50
        ]
        
        return picks[:count]


async def run_daily_scan(
    openai_api_key: str | None = None,
    perplexity_api_key: str | None = None,
    output_file: str | None = None,
) -> list[ScanResult]:
    """Run daily market scan.
    
    Args:
        openai_api_key: OpenAI API key.
        perplexity_api_key: Perplexity API key.
        output_file: Optional file to save results.
        
    Returns:
        List of scan results.
    """
    researcher = ResearcherAgent(
        openai_api_key=openai_api_key,
        perplexity_api_key=perplexity_api_key,
    )
    
    results = await researcher.scan_market(target_count=500)
    
    # Save to file
    if output_file:
        df = pd.DataFrame([
            {
                "symbol": r.symbol,
                "score": r.score,
                "momentum": r.momentum_score,
                "volume": r.volume_score,
                "technical": r.technical_score,
                "catalyst": r.catalyst_score,
                "price": r.price,
                "rsi": r.rsi,
                "trend": r.short_term_trend,
                "catalysts": ", ".join(r.catalysts),
            }
            for r in results
        ])
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    return results
