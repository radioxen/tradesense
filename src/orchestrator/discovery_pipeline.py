"""Discovery Pipeline - Integrated Scanner + Agent Pipeline.

Connects the Researcher Agent with the full analysis pipeline:
Scanner → Technical → Fundamental → Hybrid → Executive → Risk Guardian

This is the main end-to-end flow for autonomous stock discovery and analysis.
"""

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any
import json

from src.agents.researcher import ResearcherAgent, ScanResult
from src.data.market_data import get_provider
from src.data.feature_store import FeatureBuilder
from src.orchestrator.contracts import (
    Action,
    PipelineState,
    TechnicalSignal,
    FundamentalSignal,
    HybridSignal,
    ExecutiveDecision,
    RiskGuardianOutput,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DiscoveryResult:
    """Result from the discovery pipeline for a single stock."""
    symbol: str
    scan_result: ScanResult
    technical_signal: TechnicalSignal | None = None
    fundamental_signal: FundamentalSignal | None = None
    hybrid_signal: HybridSignal | None = None
    executive_decision: ExecutiveDecision | None = None
    risk_output: RiskGuardianOutput | None = None
    final_action: Action | None = None
    approved: bool = False
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "scan_score": self.scan_result.score,
            "technical": self.technical_signal.direction.value if self.technical_signal else None,
            "fundamental": self.fundamental_signal.direction.value if self.fundamental_signal else None,
            "hybrid": self.hybrid_signal.net_direction.value if self.hybrid_signal else None,
            "action": self.final_action.value if self.final_action else None,
            "approved": self.approved,
            "reasoning": self.reasoning,
        }


class DiscoveryPipeline:
    """Integrated Scanner + Agent Pipeline.
    
    Flow:
    1. Scanner discovers top momentum stocks
    2. Each stock goes through Technical → Fundamental → Hybrid
    3. Executive makes decision
    4. Risk Guardian validates
    5. Returns actionable recommendations
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        perplexity_api_key: str | None = None,
        max_stocks_to_analyze: int = 10,
        use_llm: bool = True,
        config: dict | None = None,
    ):
        """Initialize discovery pipeline.
        
        Args:
            openai_api_key: OpenAI API key.
            perplexity_api_key: Perplexity API key.
            max_stocks_to_analyze: Max stocks to run full analysis on.
            use_llm: Whether to use LLM for agents.
            config: Additional configuration.
        """
        self.openai_api_key = openai_api_key
        self.perplexity_api_key = perplexity_api_key
        self.max_stocks = max_stocks_to_analyze
        self.use_llm = use_llm
        self.config = config or {}

        # Initialize components
        self.researcher = ResearcherAgent(
            openai_api_key=openai_api_key,
            perplexity_api_key=perplexity_api_key,
        )
        self.feature_builder = FeatureBuilder()

        # Agents initialized lazily to avoid circular imports
        self._technical_agent = None
        self._fundamental_agent = None
        self._hybrid_agent = None
        self._executive_agent = None
        self._risk_guardian = None

    def _init_agents(self):
        """Initialize agents lazily."""
        if self._technical_agent is not None:
            return

        from src.agents.technical import TechnicalAnalystAgent
        from src.agents.fundamental import FundamentalAnalystAgent
        from src.agents.hybrid import HybridAnalystAgent
        from src.agents.executive import ExecutiveAgent
        from src.agents.risk_guardian import RiskGuardian

        self._technical_agent = TechnicalAnalystAgent(config=self.config)
        self._fundamental_agent = FundamentalAnalystAgent(
            openai_api_key=self.openai_api_key,
            perplexity_api_key=self.perplexity_api_key,
            config=self.config,
        )
        self._hybrid_agent = HybridAnalystAgent(
            openai_api_key=self.openai_api_key,
            config=self.config,
        )
        self._executive_agent = ExecutiveAgent(
            openai_api_key=self.openai_api_key,
        )
        self._risk_guardian = RiskGuardian()  # No config kwarg - uses defaults

    async def run(
        self,
        scan_count: int = 50,
        analyze_top_n: int | None = None,
    ) -> list[DiscoveryResult]:
        """Run the full discovery pipeline.
        
        Args:
            scan_count: Number of stocks to scan.
            analyze_top_n: Number of top stocks to fully analyze.
            
        Returns:
            List of discovery results with recommendations.
        """
        analyze_count = analyze_top_n or self.max_stocks
        
        logger.info(f"Starting discovery pipeline (scan={scan_count}, analyze={analyze_count})")
        
        # Stage 1: Scan market
        logger.info("Stage 1: Running market scanner...")
        scan_results = await self.researcher.scan_market(
            target_count=scan_count,
            use_llm=self.perplexity_api_key is not None,
        )
        
        if not scan_results:
            logger.warning("No stocks found by scanner")
            return []
        
        logger.info(f"Scanner found {len(scan_results)} candidates")
        
        # Stage 2-5: Run full analysis on top N
        top_picks = scan_results[:analyze_count]
        logger.info(f"Stage 2-5: Analyzing top {len(top_picks)} stocks...")
        
        results = []
        for scan_result in top_picks:
            try:
                result = await self._analyze_stock(scan_result)
                results.append(result)
                
                # Log progress
                action = result.final_action.value if result.final_action else "NONE"
                logger.info(f"  {scan_result.symbol}: {action} (approved={result.approved})")
                
            except Exception as e:
                logger.error(f"Failed to analyze {scan_result.symbol}: {e}")
                continue
        
        # Sort by actionable first
        results.sort(key=lambda x: (x.approved, x.scan_result.score), reverse=True)
        
        logger.info(f"Pipeline complete: {len(results)} stocks analyzed")
        return results

    async def _analyze_stock(self, scan_result: ScanResult) -> DiscoveryResult:
        """Run full agent pipeline on a single stock.
        
        Args:
            scan_result: Scan result from researcher.
            
        Returns:
            Discovery result with all agent outputs.
        """
        self._init_agents()
        
        symbol = scan_result.symbol
        result = DiscoveryResult(symbol=symbol, scan_result=scan_result)
        
        # Get market data
        provider = get_provider("yfinance")
        end = datetime.now()
        start = end - timedelta(days=60)
        
        data = await provider.fetch_ohlcv(symbol, start, end, "1h")
        if data.empty:
            result.reasoning = "No market data available"
            return result
        
        # Build features
        features_df = self.feature_builder.build_features(data)
        if features_df.empty:
            result.reasoning = "Failed to compute features"
            return result
        
        latest_features = features_df.iloc[-1].to_dict()
        current_price = float(data.iloc[-1]["close"])
        
        # Build pipeline state
        state = PipelineState(
            symbol=symbol,
            timestamp=datetime.now(),
            interval="1h",
            current_price=current_price,
            last_n_bars=data.tail(20).to_dict("records"),
            features=latest_features,
            current_position=0,
            current_position_pct=0,
            cash=100_000,  # Simulated
            equity=100_000,
            daily_pnl=0,
            trades_today=0,
        )
        
        # Stage 2: Technical Analysis
        try:
            tech_signal = await self._technical_agent.analyze(state)
            result.technical_signal = tech_signal
            state.technical_signal = tech_signal
        except Exception as e:
            logger.warning(f"Technical analysis failed for {symbol}: {e}")
        
        # Stage 3: Fundamental Analysis
        try:
            fund_signal = await self._fundamental_agent.analyze(state)
            result.fundamental_signal = fund_signal
            state.fundamental_signal = fund_signal
        except Exception as e:
            logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
        
        # Stage 4: Hybrid Synthesis
        if result.technical_signal or result.fundamental_signal:
            try:
                hybrid_signal = await self._hybrid_agent.analyze(state)
                result.hybrid_signal = hybrid_signal
                state.hybrid_signal = hybrid_signal
            except Exception as e:
                logger.warning(f"Hybrid analysis failed for {symbol}: {e}")
        
        # Stage 5: Executive Decision
        if result.hybrid_signal:
            try:
                decision = await self._executive_agent.analyze(state)
                result.executive_decision = decision
                state.executive_decision = decision
                result.final_action = decision.action
            except Exception as e:
                logger.warning(f"Executive decision failed for {symbol}: {e}")
        
        # Stage 6: Risk Guardian Check
        if result.executive_decision:
            try:
                # RiskGuardian._apply_rules is sync, not async
                risk_output = self._risk_guardian._apply_rules(state)
                result.risk_output = risk_output
                result.approved = risk_output.approved
                
                if not risk_output.approved:
                    result.reasoning = f"Blocked: {risk_output.veto_reasons}"
                else:
                    result.reasoning = result.executive_decision.rationale
                    
            except Exception as e:
                logger.warning(f"Risk check failed for {symbol}: {e}")
        
        return result

    async def get_buy_recommendations(
        self,
        min_score: float = 70,
        max_results: int = 5,
    ) -> list[DiscoveryResult]:
        """Get top BUY recommendations.
        
        Args:
            min_score: Minimum scan score.
            max_results: Maximum results to return.
            
        Returns:
            List of approved BUY recommendations.
        """
        all_results = await self.run(scan_count=100, analyze_top_n=20)
        
        buys = [
            r for r in all_results
            if r.approved
            and r.final_action == Action.BUY
            and r.scan_result.score >= min_score
        ]
        
        return buys[:max_results]


async def run_discovery_test(
    openai_api_key: str | None = None,
    perplexity_api_key: str | None = None,
    top_n: int = 5,
) -> list[DiscoveryResult]:
    """Run a test of the discovery pipeline.
    
    Args:
        openai_api_key: OpenAI API key.
        perplexity_api_key: Perplexity API key.
        top_n: Number of stocks to analyze.
        
    Returns:
        List of discovery results.
    """
    pipeline = DiscoveryPipeline(
        openai_api_key=openai_api_key,
        perplexity_api_key=perplexity_api_key,
        max_stocks_to_analyze=top_n,
    )
    
    return await pipeline.run(scan_count=50, analyze_top_n=top_n)
