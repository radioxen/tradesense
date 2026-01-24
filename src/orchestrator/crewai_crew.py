"""CrewAI Trading Crew with LLM-powered agents.

This module integrates CrewAI framework for multi-agent trading decisions.
All agents use GPT-5 models for dynamic decision-making.
Enhanced with Evolution Strategy from tradioxen for technical analysis.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, List
import json

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Model configuration
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
EXECUTIVE_MODEL = "gpt-5.2-2025-12-11"


# ============================================
# CrewAI Tools
# ============================================

@tool("Fetch Stock Data")
def fetch_stock_data(symbol: str, days: int = 30) -> str:
    """Fetch historical OHLCV data for a stock.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days of history
        
    Returns:
        Summary of stock data including price, range, and volume
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        end = datetime.now()
        start = end - timedelta(days=days)
        data = ticker.history(start=start, end=end)
        
        if data.empty:
            return f"No data available for {symbol}"
        
        latest = data.iloc[-1]
        prev_close = data.iloc[-2]['Close'] if len(data) > 1 else latest['Close']
        change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
        
        # Calculate momentum indicators
        returns = data['Close'].pct_change().dropna()
        momentum_5d = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0
        volatility = returns.std() * 100
        
        return f"""
Stock: {symbol}
Latest Price: ${latest['Close']:.2f}
Daily Change: {change_pct:+.2f}%
5-Day Momentum: {momentum_5d:+.2f}%
52-week Range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}
Volatility (daily): {volatility:.2f}%
Volume (avg): {data['Volume'].mean():,.0f}
Days of Data: {len(data)}
"""
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"


@tool("Calculate Technical Indicators")
def calculate_technical_indicators(symbol: str) -> str:
    """Calculate key technical indicators for trading decisions.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Technical analysis with RSI, MACD, momentum, and trading signals
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="60d")
        
        if len(data) < 20:
            return f"Insufficient data for {symbol}"
        
        close = data['Close']
        
        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12.iloc[-1] - ema26.iloc[-1]
        signal = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
        macd_hist = macd - signal
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
        
        # Moving Averages
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        price_vs_sma50 = ((close.iloc[-1] / sma50) - 1) * 100
        
        # Determine signal
        signals = []
        if rsi < 30:
            signals.append("RSI OVERSOLD (bullish)")
        elif rsi > 70:
            signals.append("RSI OVERBOUGHT (bearish)")
        
        if macd_hist > 0:
            signals.append("MACD BULLISH")
        else:
            signals.append("MACD BEARISH")
        
        if bb_position < 0.2:
            signals.append("Near BB Lower (potential bounce)")
        elif bb_position > 0.8:
            signals.append("Near BB Upper (potential pullback)")
        
        signal_summary = ", ".join(signals) if signals else "Neutral"
        
        return f"""
Technical Indicators for {symbol}:

Price Action:
- Current Price: ${close.iloc[-1]:.2f}
- vs SMA50: {price_vs_sma50:+.2f}%
- Bollinger Band Position: {bb_position:.2f} (0=lower, 1=upper)

Momentum:
- RSI (14): {rsi:.1f}
- MACD: {macd:.3f}
- MACD Signal: {signal:.3f}
- MACD Histogram: {macd_hist:.3f}

Trading Signals: {signal_summary}
"""
    except Exception as e:
        return f"Error calculating indicators for {symbol}: {str(e)}"


@tool("Get Evolution Strategy Signal")
def get_evolution_strategy_signal(symbol: str) -> str:
    """Get trading signal from Evolution Strategy neural network.
    
    This tool uses a trained neural network (based on tradioxen)
    to generate buy/sell signals from price patterns.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Trading signal with confidence from Evolution Strategy model
    """
    import yfinance as yf
    from src.agents.evolution_strategy import EvolutionStrategyAgent
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="90d")
        
        if len(data) < 60:
            return f"Insufficient data for Evolution Strategy on {symbol}"
        
        prices = data['Close'].values
        
        # Create and train agent on recent data
        agent = EvolutionStrategyAgent(
            window_size=30,
            initial_money=10000.0,
        )
        
        # Quick training (fewer epochs for speed)
        train_prices = prices[:-30]  # Use all but last 30 days for training
        agent.fit(train_prices, epochs=100, print_every=50)
        
        # Get current signal
        signal, confidence = agent.get_current_signal(prices)
        
        # Get simulated performance on test period
        result = agent.predict(prices[-30:])
        
        return f"""
Evolution Strategy Signal for {symbol}:

Current Signal: {signal}
Confidence: {confidence:.1%}

Backtest Results (last 30 days):
- Simulated Return: {result.investment_return:.2f}%
- Number of Trades: {len(result.trades)}
- Total Gains: ${result.total_gains:.2f}

Recommendation: {"STRONG " + signal if confidence > 0.7 else signal}
"""
    except Exception as e:
        return f"Evolution Strategy error for {symbol}: {str(e)}"


@tool("Search Market News")
def search_market_news(query: str) -> str:
    """Search for recent market news and catalysts.
    
    Args:
        query: Search query (e.g., stock symbol or topic)
        
    Returns:
        Summary of recent news and market sentiment
    """
    import os
    import requests
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "Perplexity API key not configured. Using basic info only."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "user",
                    "content": f"What are the latest news and catalysts for {query}? Focus on events that could move the stock price in the next 1-5 days. Be concise.",
                }
            ],
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"News search failed: {response.status_code}"
            
    except Exception as e:
        return f"Error searching news: {str(e)}"


@tool("Get Portfolio Status")
def get_portfolio_status() -> str:
    """Get current portfolio holdings and cash balance.
    
    Returns:
        Summary of current positions and available capital
    """
    import os
    from alpaca.trading.client import TradingClient
    
    try:
        client = TradingClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_API_SECRET"),
            paper=True,
        )
        
        account = client.get_account()
        positions = client.get_all_positions()
        
        result = f"""
Portfolio Status:
- Cash: ${float(account.cash):,.2f}
- Portfolio Value: ${float(account.portfolio_value):,.2f}
- Buying Power: ${float(account.buying_power):,.2f}

Positions:
"""
        if positions:
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                result += f"- {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}, P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
        else:
            result += "- No open positions\n"
        
        return result
        
    except Exception as e:
        return f"Error getting portfolio: {str(e)}"


# ============================================
# CrewAI Trading Crew
# ============================================

class TradingCrew:
    """CrewAI-powered trading crew with LLM agents.
    
    Agents:
    - Market Scanner: Finds high-momentum stocks
    - Technical Analyst: Analyzes charts and indicators
    - Fundamental Analyst: Evaluates news and catalysts
    - Risk Manager: Calculates position sizes
    - Executive Trader: Makes final decisions
    
    All agents use GPT-5 models for dynamic reasoning.
    """

    def __init__(
        self,
        symbols: List[str] | None = None,
        openai_api_key: str | None = None,
        model: str | None = None,
        executive_model: str | None = None,
        verbose: bool = True,
        budget: float = 10000.0,
    ):
        """Initialize the trading crew.
        
        Args:
            symbols: Stock symbols to analyze
            openai_api_key: OpenAI API key
            model: Model for regular agents (default: gpt-5-mini)
            executive_model: Model for executive (default: gpt-5.2)
            verbose: Enable verbose output
            budget: Trading budget
        """
        self.symbols = symbols or ["AAPL", "NVDA", "GOOGL"]
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.executive_model = executive_model or EXECUTIVE_MODEL
        self.verbose = verbose
        self.budget = budget
        self.on_step: Optional[Callable] = None
        
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        logger.info(f"TradingCrew initializing with models: agents={self.model}, executive={self.executive_model}")
        
        self._agents = self._create_agents()
        self._tasks = self._create_tasks()
        self._crew = self._create_crew()

    def _create_agents(self) -> dict[str, Agent]:
        """Create the trading crew agents."""
        
        market_scanner = Agent(
            role="Market Scanner",
            goal="Identify stocks with strong momentum and upcoming catalysts that could move 5-20% in the next 1-5 trading days",
            backstory="""You are an expert market scanner with 20 years of experience 
            identifying high-momentum stocks before they make big moves. You excel at 
            finding technical breakouts, earnings plays, and catalyst-driven opportunities.
            You focus on liquid stocks that are likely to see significant price action.
            
            You use your LLM reasoning to:
            - Analyze volume patterns and price action
            - Interpret news and sentiment
            - Score stocks by momentum potential
            - Explain your reasoning for each pick""",
            tools=[fetch_stock_data, search_market_news],
            llm=self.model,
            verbose=self.verbose,
            allow_delegation=True,
        )
        
        technical_analyst = Agent(
            role="Technical Analyst",
            goal="Provide precise entry and exit points using technical analysis and the Evolution Strategy neural network",
            backstory="""You are a quantitative technical analyst combining classical 
            chart patterns with machine learning. You use RSI, MACD, Bollinger Bands,
            and a proprietary Evolution Strategy neural network trained on historical data.
            
            You use your LLM reasoning to:
            - Interpret technical indicator signals
            - Combine multiple signals into actionable recommendations
            - Identify support/resistance levels
            - Predict short-term price movements with confidence levels
            
            Your Evolution Strategy model was trained on the tradioxen framework
            and has shown strong backtested returns.""",
            tools=[calculate_technical_indicators, get_evolution_strategy_signal, fetch_stock_data],
            llm=self.model,
            verbose=self.verbose,
            allow_delegation=True,
        )
        
        fundamental_analyst = Agent(
            role="Fundamental Analyst",
            goal="Evaluate news, catalysts, and market sentiment to identify event-driven opportunities",
            backstory="""You are a fundamental analyst specializing in catalyst-driven 
            trades. You track earnings, FDA approvals, product launches, and macro events.
            You excel at predicting how news will impact stock prices.
            
            You use your LLM reasoning to:
            - Analyze news sentiment and impact
            - Identify upcoming catalysts
            - Assess market positioning
            - Predict near-term price reactions""",
            tools=[search_market_news, fetch_stock_data],
            llm=self.model,
            verbose=self.verbose,
            allow_delegation=True,
        )
        
        risk_manager = Agent(
            role="Risk Manager",
            goal=f"Protect capital by sizing positions appropriately for ${self.budget:,.0f} budget with max 1% risk per trade",
            backstory=f"""You are a risk manager responsible for capital preservation.
            You calculate position sizes based on volatility and stop-loss levels.
            You never risk more than 1% of portfolio on any single trade.
            
            Current trading budget: ${self.budget:,.0f}
            
            You use your LLM reasoning to:
            - Calculate optimal position sizes
            - Set appropriate stop-loss levels
            - Assess portfolio correlation risk
            - Veto trades that exceed risk parameters""",
            tools=[get_portfolio_status, fetch_stock_data],
            llm=self.model,
            verbose=self.verbose,
            allow_delegation=False,
        )
        
        executive_trader = Agent(
            role="Executive Trader (CEO)",
            goal="Make final trading decisions by synthesizing all agent inputs into actionable trades with detailed rationale",
            backstory=f"""You are the Chief Investment Officer responsible for final execution 
            decisions. You weigh technical, fundamental, and risk inputs to make confident 
            BUY, SELL, or HOLD decisions. You're decisive but disciplined, only taking trades 
            with clear edge and proper risk/reward ratios of at least 2:1.
            
            Your trading budget is ${self.budget:,.0f}.
            
            You use your superior LLM reasoning (GPT-5.2) to:
            - Synthesize all analyst reports
            - Resolve conflicts between signals
            - Make final BUY/SELL/HOLD decisions
            - Determine exact position sizes
            - Set stop-loss and take-profit levels
            - Document detailed rationale for each decision""",
            tools=[get_portfolio_status],
            llm=self.executive_model,
            verbose=self.verbose,
            allow_delegation=False,
        )
        
        return {
            "scanner": market_scanner,
            "technical": technical_analyst,
            "fundamental": fundamental_analyst,
            "risk": risk_manager,
            "executive": executive_trader,
        }

    def _create_tasks(self) -> List[Task]:
        """Create the trading workflow tasks."""
        
        scan_task = Task(
            description=f"""Scan the following stocks for momentum opportunities: {', '.join(self.symbols)}
            
            For each stock:
            1. Check recent price action and volume
            2. Search for any upcoming catalysts or news
            3. Rank stocks by momentum potential
            
            Output a ranked list of the top opportunities with brief reasoning.""",
            expected_output="Ranked list of stocks with momentum scores and catalysts",
            agent=self._agents["scanner"],
        )
        
        technical_task = Task(
            description=f"""Perform technical analysis on the top stocks from the scanner.
            
            For each stock:
            1. Calculate RSI, MACD, and Bollinger Band signals
            2. Run the Evolution Strategy neural network for ML-based signals
            3. Identify key support/resistance levels
            4. Provide entry price, stop-loss, and target
            
            Focus on stocks showing bullish technical setups.""",
            expected_output="Technical analysis with specific entry/exit levels and ML signals",
            agent=self._agents["technical"],
            context=[scan_task],
        )
        
        fundamental_task = Task(
            description=f"""Analyze news and catalysts for the stocks being considered.
            
            For each stock:
            1. Search for recent news that could move the price
            2. Identify any upcoming events (earnings, FDA, etc.)
            3. Assess market sentiment
            4. Provide a fundamental rating (Bullish/Neutral/Bearish)
            
            Focus on near-term catalysts (1-5 days).""",
            expected_output="Fundamental analysis with catalyst timeline and sentiment",
            agent=self._agents["fundamental"],
            context=[scan_task],
        )
        
        risk_task = Task(
            description=f"""Calculate position sizes and risk parameters for ${self.budget:,.0f} budget.
            
            For each potential trade:
            1. Calculate appropriate position size based on portfolio
            2. Verify stop loss provides max 1% portfolio risk
            3. Check if adding position maintains diversification
            4. Approve or reject based on risk parameters
            
            Get current portfolio status first.""",
            expected_output="Risk assessment with approved position sizes and any trades that should be rejected",
            agent=self._agents["risk"],
            context=[technical_task, fundamental_task],
        )
        
        decision_task = Task(
            description="""Make final trading decisions based on all inputs.
            
            Synthesize technical, fundamental, and risk assessments to:
            1. Decide BUY, SELL, or HOLD for each stock
            2. Specify exact order parameters (symbol, action, quantity, price limits)
            3. Provide confidence score (0-1)
            4. State clear rationale for each decision
            
            Only recommend trades with clear edge and proper risk/reward.
            
            Output ONLY valid JSON (no markdown).""",
            expected_output="""Return ONLY valid JSON in this schema:
            
            {
              "decisions": [
                {
                  "symbol": "AAPL",
                  "action": "BUY|SELL|HOLD",
                  "quantity": 100,
                  "entry": 185.25,
                  "stop_loss": 178.0,
                  "target": 198.0,
                  "confidence": 0.72,
                  "rationale": "Brief explanation"
                }
              ]
            }
            
            Do not include markdown or commentary outside the JSON.""",
            agent=self._agents["executive"],
            context=[technical_task, fundamental_task, risk_task],
        )
        
        return [scan_task, technical_task, fundamental_task, risk_task, decision_task]

    def _create_crew(self) -> Crew:
        """Create the CrewAI crew."""
        
        def step_callback(step_output):
            """Callback for step updates."""
            if self.on_step:
                try:
                    agent_name = getattr(step_output, 'agent', 'Agent')
                    if hasattr(agent_name, 'role'):
                        agent_name = agent_name.role
                    output = getattr(step_output, 'output', str(step_output))
                    self.on_step(str(agent_name), "Step completed", str(output)[:500])
                except Exception as e:
                    logger.warning(f"Step callback error: {e}")
        
        def task_callback(task_output):
            """Callback for task updates."""
            if self.on_step:
                try:
                    description = getattr(task_output, 'description', 'Task')[:100]
                    output = getattr(task_output, 'raw', str(task_output))
                    agent = getattr(task_output, 'agent', 'Agent')
                    if hasattr(agent, 'role'):
                        agent = agent.role
                    self.on_step(str(agent), f"Task: {description}...", str(output)[:500])
                except Exception as e:
                    logger.warning(f"Task callback error: {e}")
        
        return Crew(
            agents=list(self._agents.values()),
            tasks=self._tasks,
            process=Process.sequential,
            verbose=self.verbose,
            tracing=True,
            memory=True,
            step_callback=step_callback,
            task_callback=task_callback,
        )

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
        on_step: Callable | None = None,
    ) -> str:
        """Execute the trading crew.
        
        Args:
            inputs: Optional inputs for the crew
            on_step: Callback(agent, action, thinking) for real-time updates
            
        Returns:
            Final trading recommendations
        """
        self.on_step = on_step
        
        logger.info(f"Starting TradingCrew for symbols: {self.symbols}")
        logger.info("Traces available at: https://app.crewai.com")
        
        if on_step:
            on_step("System", "CrewAI Crew starting", f"Analyzing {len(self.symbols)} stocks with 5 LLM agents")
        
        result = self._crew.kickoff(inputs=inputs)
        
        if on_step:
            on_step("System", "CrewAI Crew completed", "All agents finished analysis")
        
        logger.info("TradingCrew execution complete")
        return result.raw if hasattr(result, 'raw') else str(result)


# Convenience function
def run_trading_crew(
    symbols: List[str],
    budget: float = 10000.0,
    callback: Optional[Callable] = None,
) -> dict:
    """Run the trading crew and return results.
    
    Args:
        symbols: Stocks to analyze
        budget: Trading budget
        callback: Optional callback for updates
        
    Returns:
        Dictionary with status and results
    """
    try:
        crew = TradingCrew(symbols=symbols, budget=budget)
        result = crew.kickoff(on_step=callback)
        return {"status": "success", "result": result, "decisions": parse_crew_decisions(result)}
    except Exception as e:
        logger.error(f"Trading crew failed: {e}")
        return {"status": "error", "error": str(e)}


def parse_crew_decisions(output: str) -> list[dict[str, Any]]:
    """Parse CrewAI output into structured decisions."""
    if not output:
        return []

    payload = _load_json_payload(output)
    if payload is not None:
        decisions = _normalize_decisions(payload)
        if decisions:
            return decisions

    return _parse_text_decisions(output)


def _load_json_payload(output: str) -> Any | None:
    """Best-effort JSON extraction from output."""
    try:
        return json.loads(output)
    except Exception:
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, output, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                continue
    return None


def _normalize_decisions(payload: Any) -> list[dict[str, Any]]:
    """Normalize decision payload into a list of decision dicts."""
    if isinstance(payload, dict):
        items = payload.get("decisions") or payload.get("trades") or [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    decisions: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        decision = _normalize_decision(item)
        if decision.get("symbol"):
            decisions.append(decision)
    return decisions


def _normalize_decision(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw decision dict fields."""
    symbol = str(raw.get("symbol") or raw.get("ticker") or "").upper().strip()
    action = _normalize_action(raw.get("action") or raw.get("decision") or raw.get("signal"))
    confidence = _parse_number(raw.get("confidence") or raw.get("confidence_pct") or raw.get("score"))
    if confidence is None:
        confidence = 0.5
    elif confidence > 1:
        confidence = min(confidence / 100, 1.0)

    quantity = _parse_number(raw.get("quantity") or raw.get("qty") or raw.get("shares"))
    entry = _parse_number(raw.get("entry") or raw.get("entry_price") or raw.get("price"))
    stop_loss = _parse_number(raw.get("stop_loss") or raw.get("stop"))
    target = _parse_number(raw.get("target") or raw.get("take_profit"))

    return {
        "symbol": symbol,
        "action": action,
        "quantity": int(quantity) if quantity is not None else 0,
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "confidence": float(confidence),
        "rationale": str(raw.get("rationale") or raw.get("reasoning") or ""),
    }


def _parse_text_decisions(output: str) -> list[dict[str, Any]]:
    """Parse decisions from formatted text output."""
    decisions: list[dict[str, Any]] = []
    blocks = re.split(r"(?i)DECISION\s*:", output)
    for block in blocks[1:]:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        action = _normalize_action(lines[0].split()[0])
        symbol = _search_field(block, r"(?i)Symbol\s*:\s*([A-Z0-9.\-]+)") or ""
        quantity = _parse_number(_search_field(block, r"(?i)Quantity\s*:\s*([0-9,.]+)"))
        entry = _parse_number(_search_field(block, r"(?i)Entry\s*:\s*\$?([0-9,.]+)"))
        stop_loss = _parse_number(_search_field(block, r"(?i)Stop\s*Loss\s*:\s*\$?([0-9,.]+)"))
        target = _parse_number(_search_field(block, r"(?i)Target\s*:\s*\$?([0-9,.]+)"))
        confidence = _parse_number(_search_field(block, r"(?i)Confidence\s*:\s*([0-9.]+)"))
        if confidence is None:
            confidence = 0.5
        elif confidence > 1:
            confidence = min(confidence / 100, 1.0)
        rationale = _search_field(block, r"(?i)Rationale\s*:\s*(.*)") or ""

        if symbol:
            decisions.append({
                "symbol": symbol.upper(),
                "action": action,
                "quantity": int(quantity) if quantity is not None else 0,
                "entry": entry,
                "stop_loss": stop_loss,
                "target": target,
                "confidence": float(confidence),
                "rationale": rationale.strip(),
            })
    return decisions


def _normalize_action(value: Any) -> str:
    action = str(value or "").upper().strip()
    if action not in {"BUY", "SELL", "HOLD"}:
        return "HOLD"
    return action


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def _search_field(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
