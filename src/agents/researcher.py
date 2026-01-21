"""Researcher Agent for market scanning.

Scans the entire market to find stocks likely to jump in 2-5 days.
Uses a multi-stage approach to minimize token usage:
1. Free screeners for initial filtering (5000+ → 1000)
2. Technical indicators for momentum signals (1000 → 500)
3. LLM + Perplexity for final ranking (only on top candidates)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from dataclasses import dataclass, field
import json

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

    def _get_system_prompt(self) -> str:
        return """You are a market research analyst specializing in short-term momentum trading.
Your job is to identify stocks likely to increase 5-20% within 2-5 trading days.

Focus on:
- Upcoming catalysts (earnings, FDA approvals, product launches)
- Sector momentum and rotation
- Technical breakout patterns
- Unusual institutional activity
- News sentiment shifts

Be concise. Output JSON only."""

    async def scan_market(
        self,
        target_count: int = 500,
        use_llm: bool = True,
    ) -> list[ScanResult]:
        """Scan market for momentum candidates.
        
        Args:
            target_count: Target number of stocks to return.
            use_llm: Whether to use LLM for catalyst detection.
            
        Returns:
            List of scan results, sorted by score.
        """
        logger.info("Starting market scan...")
        
        # Stage 1: Get broad universe
        universe = await self._get_universe()
        logger.info(f"Stage 1: Universe of {len(universe)} symbols")
        
        # Stage 2: Technical filtering
        scored = await self._score_technical(universe)
        scored = sorted(scored, key=lambda x: x.score, reverse=True)
        
        # Keep top candidates for LLM analysis
        top_candidates = scored[:min(len(scored), target_count + 100)]
        logger.info(f"Stage 2: {len(top_candidates)} technical candidates")
        
        # Stage 3: LLM catalyst detection (only on top stocks)
        if use_llm and self.perplexity_api_key:
            top_candidates = await self._add_catalyst_scores(top_candidates)
            top_candidates = sorted(top_candidates, key=lambda x: x.score, reverse=True)
        
        # Return top N
        results = top_candidates[:target_count]
        logger.info(f"Stage 3: Returning {len(results)} final candidates")
        
        return results

    async def _get_universe(self) -> list[str]:
        """Get universe of stocks to scan.
        
        Uses free data sources to get comprehensive list.
        """
        symbols = set()
        
        # S&P 500
        try:
            sp500 = await self._fetch_index_components("SPY")
            symbols.update(sp500)
        except Exception as e:
            logger.warning(f"Failed to get S&P 500: {e}")
        
        # NASDAQ 100
        try:
            nasdaq = await self._fetch_index_components("QQQ")
            symbols.update(nasdaq)
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
        symbols.update(popular)
        
        # Filter to valid, liquid stocks
        return list(symbols)

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
                
                # Calculate component scores (0-100)
                
                # Momentum: RSI in sweet spot (40-70)
                if 35 <= rsi <= 55:
                    momentum_score = 80  # Bounce potential
                elif 55 < rsi <= 70:
                    momentum_score = 70  # Trending
                elif rsi > 70:
                    momentum_score = 30  # Overbought
                else:
                    momentum_score = 50
                
                # Volume: Higher is better
                volume_score = min(100, volume_ratio * 50)
                
                # Technical: MACD crossover
                if macd > macd_signal and macd > 0:
                    technical_score = 85  # Bullish crossover
                elif macd > macd_signal:
                    technical_score = 70  # Crossover below zero
                elif macd < macd_signal:
                    technical_score = 40  # Bearish
                else:
                    technical_score = 50
                
                # Short-term trend
                returns_5d = latest.get("return_5", 0)
                if returns_5d > 0.03:
                    trend = "bullish"
                elif returns_5d < -0.03:
                    trend = "bearish"
                else:
                    trend = "neutral"
                
                # Composite score (technical only at this stage)
                score = (momentum_score * 0.3 + volume_score * 0.3 + technical_score * 0.4)
                
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
                    candidate.score = (
                        candidate.score * 0.6 +  # Technical weight
                        candidate.catalyst_score * 0.4  # Catalyst weight
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
        
        prompt = f"""Analyze these stocks for potential 2-5 day catalysts: {', '.join(symbols)}

For each stock, identify:
1. Upcoming earnings, FDA decisions, product launches
2. Recent insider buying or institutional activity
3. Sector tailwinds or news momentum
4. Technical breakout potential

Return JSON format:
{{
  "SYMBOL": {{
    "score": 0-100,
    "catalysts": ["catalyst1", "catalyst2"],
    "reasoning": "brief explanation"
  }}
}}

Only include stocks with score > 50. Be concise."""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                    },
                ) as resp:
                    if resp.status != 200:
                        return {}
                    
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Parse JSON from response
                    # Handle markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    return json.loads(content)
                    
        except Exception as e:
            logger.error(f"Perplexity query failed: {e}")
            return {}

    async def get_daily_picks(
        self,
        count: int = 10,
    ) -> list[ScanResult]:
        """Get top daily picks.
        
        Convenience method for getting actionable picks.
        """
        all_results = await self.scan_market(target_count=500, use_llm=True)
        
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
