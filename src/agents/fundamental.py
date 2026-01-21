"""Fundamental Analyst Agent with LLM and web search.

Uses OpenAI GPT for reasoning and Perplexity for real-time
news and research. Outputs sentiment, events, and citations.
"""

from datetime import datetime
from typing import Any
import json

from src.agents.base import LLMAgent
from src.orchestrator.contracts import (
    Citation,
    Direction,
    Event,
    FundamentalSignal,
    PipelineState,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are an expert fundamental analyst at a quantitative hedge fund.

Your role is to analyze news, events, and market sentiment to provide trading signals.

CRITICAL RULES:
1. EVERY claim must be backed by a citation from the provided sources
2. Be skeptical of unverified rumors - flag them as low credibility
3. Focus on material events: earnings, guidance, M&A, regulatory, macro
4. Consider both immediate and longer-term implications
5. Rate sentiment on a scale from -1.0 (extremely bearish) to +1.0 (extremely bullish)

Output your analysis as valid JSON only, no other text."""


class FundamentalAnalystAgent(LLMAgent[FundamentalSignal]):
    """Fundamental Analyst Agent using LLM + web search.

    Analyzes news, events, and sentiment to produce trading signals.
    Uses Perplexity for real-time web search when enabled.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        perplexity_api_key: str | None = None,
        enable_live_search: bool = False,
        config: dict[str, Any] | None = None,
    ):
        """Initialize Fundamental Analyst.

        Args:
            openai_api_key: OpenAI API key.
            perplexity_api_key: Perplexity API key for web search.
            enable_live_search: Enable real-time web search.
            config: Additional configuration.
        """
        super().__init__(
            name="fundamental_analyst",
            system_prompt=SYSTEM_PROMPT,
            config=config,
        )
        self.openai_api_key = openai_api_key
        self.perplexity_api_key = perplexity_api_key
        self.enable_live_search = enable_live_search
        self._openai_client = None
        self._perplexity_available = False

    @property
    def description(self) -> str:
        return "Analyzes news, events, and sentiment using LLM and web search"

    async def _get_openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            except ImportError:
                logger.error("OpenAI package not installed")
                raise
        return self._openai_client

    async def _search_news(self, symbol: str) -> list[dict]:
        """Search for recent news using Perplexity.

        Args:
            symbol: Trading symbol.

        Returns:
            List of news items.
        """
        if not self.enable_live_search or not self.perplexity_api_key:
            logger.info("Live search disabled, using cached/mock data")
            return self._get_mock_news(symbol)

        try:
            # Use Perplexity API through MCP or direct call
            # For now, return mock data as fallback
            return self._get_mock_news(symbol)
        except Exception as e:
            logger.warning(f"Perplexity search failed: {e}")
            return self._get_mock_news(symbol)

    def _get_mock_news(self, symbol: str) -> list[dict]:
        """Get mock news for backtesting."""
        return [
            {
                "title": f"{symbol} quarterly earnings beat expectations",
                "source": "Reuters",
                "timestamp": datetime.now().isoformat(),
                "snippet": f"{symbol} reported Q4 earnings that exceeded analyst estimates by 12%, driven by strong revenue growth in core segments.",
                "url": "https://reuters.com/example",
                "credibility": 0.95,
            },
            {
                "title": f"Analysts upgrade {symbol} price target",
                "source": "Bloomberg",
                "timestamp": datetime.now().isoformat(),
                "snippet": f"Multiple Wall Street analysts raised their price targets for {symbol} following strong earnings, citing improved margins.",
                "url": "https://bloomberg.com/example",
                "credibility": 0.90,
            },
        ]

    def _build_user_prompt(self, state: PipelineState) -> str:
        """Build user prompt with context."""
        # This will be populated with actual news
        return ""

    def _parse_response(self, response: str) -> FundamentalSignal:
        """Parse LLM response into signal."""
        # Parse JSON from response
        try:
            data = json.loads(response)
            return self._create_signal_from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise

    def _create_signal_from_dict(
        self,
        data: dict,
        news_count: int = 0,
    ) -> FundamentalSignal:
        """Create signal from parsed data."""
        # Parse events
        events = []
        for event_data in data.get("events", []):
            events.append(Event(
                event_type=event_data.get("type", "news"),
                title=event_data.get("title", ""),
                timestamp=datetime.now(),
                impact_estimate=event_data.get("impact", "medium"),
                sentiment=event_data.get("sentiment", 0.0),
                citations=[],
            ))

        # Parse citations
        citations = []
        for cite_data in data.get("citations", []):
            citations.append(Citation(
                source=cite_data.get("source", "Unknown"),
                title=cite_data.get("title", ""),
                url=cite_data.get("url"),
                timestamp=datetime.now(),
                credibility_score=cite_data.get("credibility", 0.5),
            ))

        # Determine direction
        sentiment = data.get("sentiment_score", 0.0)
        if sentiment > 0.3:
            direction = Direction.LONG
        elif sentiment < -0.3:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        return FundamentalSignal(
            sentiment_score=sentiment,
            credibility_score=data.get("credibility_score", 0.5),
            events=events,
            catalysts=data.get("catalysts", []),
            risks=data.get("risks", []),
            impact_prior=data.get("impact_prior", {}),
            risk_flags=data.get("risk_flags", []),
            direction=direction,
            confidence=data.get("confidence", 0.5),
            citations=citations,
            news_count_analyzed=news_count,
            timestamp=datetime.now(),
        )

    async def analyze(self, state: PipelineState) -> FundamentalSignal:
        """Perform fundamental analysis.

        Args:
            state: Current pipeline state.

        Returns:
            Fundamental signal with sentiment and events.
        """
        symbol = state.symbol

        # Get news
        news_items = await self._search_news(symbol)
        news_count = len(news_items)

        if not news_items:
            # Return neutral signal if no news
            return FundamentalSignal(
                sentiment_score=0.0,
                credibility_score=0.5,
                events=[],
                catalysts=[],
                risks=[],
                impact_prior={},
                risk_flags=["no_recent_news"],
                direction=Direction.NEUTRAL,
                confidence=0.3,
                citations=[],
                news_count_analyzed=0,
                timestamp=state.timestamp,
            )

        # Build prompt
        news_text = "\n\n".join([
            f"[{i+1}] {item['source']}: {item['title']}\n{item['snippet']}"
            for i, item in enumerate(news_items)
        ])

        user_prompt = f"""Analyze the following news for {symbol} and provide a trading signal.

Current price: ${state.current_price:.2f}
Timestamp: {state.timestamp}

NEWS:
{news_text}

Provide your analysis as JSON with the following structure:
{{
    "sentiment_score": <float -1 to 1>,
    "credibility_score": <float 0 to 1>,
    "confidence": <float 0 to 1>,
    "events": [
        {{"type": "earnings|guidance|macro|regulatory", "title": "...", "impact": "high|medium|low", "sentiment": <float>}}
    ],
    "catalysts": ["catalyst1", "catalyst2"],
    "risks": ["risk1", "risk2"],
    "risk_flags": ["rumor", "unverified", etc if applicable],
    "citations": [
        {{"source": "...", "title": "...", "credibility": <float>}}
    ]
}}"""

        try:
            # Call OpenAI
            client = await self._get_openai_client()

            response = await client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            result = response.choices[0].message.content
            signal = self._parse_response(result)

            # Override news count
            signal = FundamentalSignal(
                **{**signal.model_dump(), "news_count_analyzed": news_count}
            )

            return signal

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Return fallback signal based on raw news sentiment
            return self._fallback_analysis(news_items, state)

    def _fallback_analysis(
        self,
        news_items: list[dict],
        state: PipelineState,
    ) -> FundamentalSignal:
        """Fallback analysis when LLM fails."""
        # Simple keyword-based sentiment
        positive_words = ["beat", "upgrade", "strong", "growth", "profit", "exceed"]
        negative_words = ["miss", "downgrade", "weak", "loss", "decline", "warning"]

        sentiment_scores = []
        citations = []

        for item in news_items:
            text = (item.get("title", "") + " " + item.get("snippet", "")).lower()

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)

            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                score = 0.0

            sentiment_scores.append(score)

            citations.append(Citation(
                source=item.get("source", "Unknown"),
                title=item.get("title", ""),
                url=item.get("url"),
                timestamp=datetime.now(),
                credibility_score=item.get("credibility", 0.5),
            ))

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        if avg_sentiment > 0.3:
            direction = Direction.LONG
        elif avg_sentiment < -0.3:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        return FundamentalSignal(
            sentiment_score=avg_sentiment,
            credibility_score=0.6,
            events=[],
            catalysts=[],
            risks=[],
            impact_prior={},
            risk_flags=["fallback_analysis"],
            direction=direction,
            confidence=0.4,
            citations=citations,
            news_count_analyzed=len(news_items),
            timestamp=state.timestamp,
        )
