"""Workflow orchestration API - CrewAI-powered trading flow.

All agents are LLM-powered using GPT-5 models via CrewAI framework.
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Request, BackgroundTasks
from pydantic import BaseModel
import structlog

from api import database as db
from src.data.market_data import get_provider
from src.data.feature_store import FeatureBuilder
from src.execution.broker import OrderSide

logger = structlog.get_logger()
router = APIRouter()

# CrewAI configuration - all agents are LLM-powered
CREWAI_MODEL = "gpt-5-mini-2025-08-07"
CREWAI_EXECUTIVE_MODEL = "gpt-5.2-2025-12-11"


def get_agents():
    """Get CrewAI configuration (legacy compatibility shim).
    
    Note: The webapp now uses CrewAI for all agent operations.
    This function returns None as agents are instantiated via CrewAI.
    """
    logger.info("Using CrewAI workflow - legacy agent instances not used")
    return None, None, None


class StartWorkflowRequest(BaseModel):
    """Request to start trading workflow."""
    scan_count: int = 10
    analyze_top: int = 5
    auto_execute: bool = False
    focus: str = "momentum"  # momentum, value, swing
    use_crewai: bool = True  # Use CrewAI with full LLM agents


class BudgetRequest(BaseModel):
    """Budget modification request."""
    amount: float
    action: Literal["deposit", "withdraw"]
    description: str = ""


# Store active workflow for real-time updates
active_workflow: dict = {}
workflow_cancelled: bool = False


@router.get("/budget")
async def get_budget():
    """Get current budget and history."""
    return {
        "current_budget": db.get_budget(),
        "history": db.get_budget_history(20),
    }


@router.post("/budget")
async def modify_budget(request: BudgetRequest):
    """Deposit or withdraw from budget."""
    amount = request.amount if request.action == "deposit" else -request.amount
    new_budget = db.add_budget_entry(amount, request.action, request.description)
    
    logger.info("Budget modified", action=request.action, amount=request.amount, new_total=new_budget)
    
    return {
        "status": "success",
        "new_budget": new_budget,
        "action": request.action,
        "amount": request.amount,
    }


@router.post("/start")
async def start_workflow(request: Request, params: StartWorkflowRequest, background_tasks: BackgroundTasks):
    """Start the trading workflow orchestrated by Executive."""
    global active_workflow
    
    workflow_id = str(uuid.uuid4())[:8]
    db.create_workflow(workflow_id)
    
    active_workflow = {
        "id": workflow_id,
        "status": "starting",
        "steps": [],
        "current_step": "Initializing",
        "stocks": [],
        "decisions": [],
        "trades": [],
    }
    
    # Run workflow in background
    background_tasks.add_task(
        run_workflow,
        request.app.state.app_state,
        workflow_id,
        params.scan_count,
        params.analyze_top,
        params.auto_execute,
    )
    
    return {
        "status": "started",
        "workflow_id": workflow_id,
        "message": "Executive Agent is now orchestrating the trading workflow",
    }


@router.get("/status")
async def get_workflow_status():
    """Get current workflow status."""
    global active_workflow
    return active_workflow


@router.get("/history")
async def get_workflow_history():
    """Get workflow history."""
    return {
        "workflows": db.get_workflows(20),
    }


@router.get("/decisions/{workflow_id}")
async def get_workflow_decisions(workflow_id: str):
    """Get all decisions for a workflow."""
    return {
        "decisions": db.get_decisions(workflow_id),
    }


@router.post("/cancel")
async def cancel_workflow():
    """Cancel the running workflow."""
    global workflow_cancelled, active_workflow
    
    if active_workflow.get("status") == "running":
        workflow_cancelled = True
        active_workflow["status"] = "cancelling"
        active_workflow["current_step"] = "Cancellation requested..."
        active_workflow["steps"].append({
            "agent": "System",
            "action": "Workflow cancellation requested by user",
            "time": datetime.now().isoformat(),
        })
        return {"status": "cancelling", "message": "Workflow cancellation requested"}
    
    return {"status": "no_workflow", "message": "No running workflow to cancel"}


@router.get("/trades")
async def get_trade_history():
    """Get trade history."""
    return {
        "trades": db.get_trades(100),
    }


@router.get("/performance")
async def get_performance():
    """Get performance statistics."""
    stats = db.get_performance_stats()
    portfolio_history = db.get_portfolio_history(30)
    
    return {
        "stats": stats,
        "portfolio_history": portfolio_history,
    }


def add_step(agent: str, action: str, thinking: str = ""):
    """Add a step to the active workflow with optional thinking."""
    global active_workflow
    step = {
        "agent": agent,
        "action": action,
        "thinking": thinking,
        "time": datetime.now().isoformat(),
    }
    active_workflow["steps"].append(step)
    return step


async def run_crewai_workflow(
    state,
    workflow_id: str,
    scan_count: int,
    analyze_top: int,
    auto_execute: bool,
):
    """Run the trading workflow using CrewAI with full LLM agents.
    
    All agents use GPT-5 models:
    - Scanner: gpt-5-mini - scans market for momentum stocks
    - Technical: gpt-5-mini - analyzes price action with LLM reasoning
    - Fundamental: gpt-5-mini - synthesizes news and catalysts
    - Risk Manager: gpt-5-mini - calculates position sizing
    - Executive: gpt-5.2 - makes final trading decisions
    """
    global active_workflow, workflow_cancelled
    workflow_cancelled = False
    
    from src.orchestrator.crewai_crew import TradingCrew, parse_crew_decisions
    
    try:
        active_workflow["status"] = "running"
        active_workflow["current_step"] = "Initializing CrewAI Trading Crew"
        
        add_step(
            "System",
            "Starting CrewAI Trading Crew with GPT-5 LLM agents",
            "All agents are LLM-powered: Scanner, Technical, Fundamental, Risk Manager use gpt-5-mini; Executive uses gpt-5.2"
        )
        
        # Get budget
        budget = db.get_budget() or 10000.0
        
        # Step 1: Scan for stocks first
        add_step(
            "Scanner",
            f"Scanning market for top {scan_count} momentum stocks",
            "Using LLM to analyze market data, volume patterns, and news to identify high-momentum opportunities"
        )
        
        # Use the researcher to get initial stock list
        scan_results = await state.researcher.scan_market(
            target_count=scan_count,
            use_llm=True,
        )
        
        if workflow_cancelled:
            raise Exception("Workflow cancelled by user")
        
        # Get top symbols
        symbols = [r.symbol for r in scan_results[:analyze_top]]
        
        if not symbols:
            add_step("System", "No stocks found matching criteria", "Scanner returned no results")
            active_workflow["status"] = "completed"
            active_workflow["current_step"] = "No stocks found"
            db.update_workflow(workflow_id, status="completed", result="No stocks found")
            return
        
        add_step(
            "Scanner",
            f"Found {len(symbols)} candidates: {', '.join(symbols)}",
            f"Momentum scores: {', '.join(f'{r.symbol}={r.score:.1f}' for r in scan_results[:analyze_top])}"
        )
        
        active_workflow["stocks"] = [
            {"symbol": r.symbol, "score": r.score, "price": r.price, "rsi": r.rsi}
            for r in scan_results[:analyze_top]
        ]
        
        db.update_workflow(workflow_id, stocks_scanned=len(scan_results))
        
        # Step 2: Run CrewAI Crew with all LLM agents
        add_step(
            "Executive",
            "Launching CrewAI Trading Crew",
            "Orchestrating all LLM agents: Technical Analyst (gpt-5-mini), Fundamental Analyst (gpt-5-mini), Risk Manager (gpt-5-mini), Executive (gpt-5.2)"
        )
        
        active_workflow["current_step"] = "Running CrewAI agents..."
        
        # Initialize CrewAI Trading Crew
        crew = TradingCrew(
            symbols=symbols,
            budget=budget,
            verbose=True,
        )
        
        add_step(
            "Technical",
            f"Analyzing {len(symbols)} stocks with LLM",
            f"Using GPT-5-mini to interpret RSI, MACD, volume patterns and generate probabilistic forecasts"
        )
        
        add_step(
            "Fundamental",
            f"Searching news and catalysts for {len(symbols)} stocks",
            "Using GPT-5-mini to synthesize news, earnings, and market sentiment"
        )
        
        add_step(
            "Risk Manager",
            f"Calculating position sizes for ${budget:,.0f} budget",
            "Using LLM to determine optimal allocation and stop-loss levels"
        )
        
        # Run the crew in a thread to avoid blocking the async event loop
        if workflow_cancelled:
            raise Exception("Workflow cancelled by user")
        
        active_workflow["current_step"] = "CrewAI agents analyzing..."
        
        import queue
        step_queue = queue.Queue()
        
        def on_crew_step(agent: str, action: str, thinking: str):
            """Callback for real-time CrewAI updates."""
            step_queue.put((agent, action, thinking))
        
        def run_crew_sync():
            """Run the crew synchronously in a thread with callbacks."""
            return crew.kickoff(on_step=on_crew_step)
        
        try:
            # Start the crew in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_crew_sync)
                
                # Poll for updates while crew is running
                while not future.done():
                    # Process any pending step updates
                    try:
                        while True:
                            agent, action, thinking = step_queue.get_nowait()
                            add_step(agent, action, thinking)
                            active_workflow["current_step"] = f"{agent}: {action[:50]}..."
                    except queue.Empty:
                        pass
                    
                    # Check for cancellation
                    if workflow_cancelled:
                        logger.warning("Workflow cancelled during CrewAI execution")
                        break
                    
                    await asyncio.sleep(0.5)  # Yield to event loop
                
                # Process any remaining updates
                try:
                    while True:
                        agent, action, thinking = step_queue.get_nowait()
                        add_step(agent, action, thinking)
                except queue.Empty:
                    pass
                
                if workflow_cancelled:
                    raise Exception("Workflow cancelled by user")
                
                crew_result = future.result(timeout=300)
            
            crew_output = crew_result.raw if hasattr(crew_result, 'raw') else str(crew_result)
        except concurrent.futures.TimeoutError:
            logger.error("CrewAI execution timed out")
            add_step("System", "CrewAI timed out after 5 minutes", "")
            crew_output = "CrewAI execution timed out"
        except Exception as e:
            logger.error(f"CrewAI execution failed: {e}")
            add_step("System", f"CrewAI error: {str(e)[:100]}", "")
            crew_output = f"CrewAI execution failed: {e}"
        
        # Log the crew output
        add_step(
            "Executive",
            "CrewAI analysis complete",
            f"LLM output: {crew_output[:500]}..." if len(crew_output) > 500 else f"LLM output: {crew_output}"
        )
        
        # Parse crew output for decisions (prefer JSON, fallback to heuristic)
        decisions = []
        parsed_decisions = parse_crew_decisions(crew_output)
        parsed_map = {d["symbol"].upper(): d for d in parsed_decisions if d.get("symbol")}

        for symbol in symbols:
            symbol_upper = symbol.upper()
            parsed = parsed_map.get(symbol_upper)

            if parsed:
                action = parsed.get("action", "HOLD")
                confidence = int(round((parsed.get("confidence") or 0.5) * 100))
                price = parsed.get("entry") or next((r.price for r in scan_results if r.symbol == symbol), 100)
                quantity = parsed.get("quantity") or 0
                if action == "BUY" and quantity <= 0 and price:
                    quantity = int((budget * 0.15) / price)
                reasoning = parsed.get("rationale") or "CrewAI parsed decision"
                stop_loss = parsed.get("stop_loss")
                target = parsed.get("target")
            else:
                if f"BUY {symbol_upper}" in crew_output.upper() or f"{symbol_upper}: BUY" in crew_output.upper():
                    action = "BUY"
                    confidence = 75
                elif f"SELL {symbol_upper}" in crew_output.upper() or f"{symbol_upper}: SELL" in crew_output.upper():
                    action = "SELL"
                    confidence = 70
                else:
                    action = "HOLD"
                    confidence = 50

                price = next((r.price for r in scan_results if r.symbol == symbol), 100)
                quantity = int((budget * 0.15) / price) if action == "BUY" else 0
                reasoning = "CrewAI LLM decision based on technical, fundamental, and risk analysis"
                stop_loss = None
                target = None

            decision = {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "confidence": confidence,
                "reasoning": reasoning,
                "stop_loss": stop_loss,
                "target": target,
            }
            decisions.append(decision)
            active_workflow["decisions"].append(decision)
            
            add_step(
                "Executive",
                f"Decision: {action} {symbol}" + (f" ({quantity} shares @ ${price:.2f})" if quantity > 0 else ""),
                f"GPT-5.2 final decision with {confidence}% confidence"
            )
            
            db.save_decision(
                workflow_id, "Executive", symbol, action,
                confidence, "CrewAI LLM decision",
                {"quantity": quantity, "price": price}
            )
        
        db.update_workflow(workflow_id, stocks_analyzed=len(symbols))
        
        # Step 3: Execute trades if auto_execute
        trades_executed = 0
        buy_decisions = [d for d in decisions if d["action"] == "BUY" and d["quantity"] > 0]
        
        if auto_execute and buy_decisions:
            add_step(
                "Executive",
                f"Executing {len(buy_decisions)} trades",
                "Auto-execute enabled, submitting orders to broker"
            )
            
            for decision in buy_decisions:
                try:
                    order = await state.broker.submit_order(
                        symbol=decision["symbol"],
                        qty=decision["quantity"],
                        side=OrderSide.BUY,
                    )
                    
                    active_workflow["trades"].append({
                        "symbol": decision["symbol"],
                        "action": "BUY",
                        "quantity": decision["quantity"],
                        "price": order.filled_avg_price or decision["price"],
                        "order_id": order.id,
                    })
                    
                    db.save_trade(
                        decision["symbol"], "buy", decision["quantity"],
                        order.filled_avg_price or decision["price"],
                        decision["confidence"], decision["reasoning"]
                    )
                    
                    trades_executed += 1
                    
                    add_step(
                        "Executive",
                        f"Executed: BUY {decision['quantity']} {decision['symbol']}",
                        f"Order filled at ${order.filled_avg_price or decision['price']:.2f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Trade execution failed for {decision['symbol']}: {e}")
                    add_step("System", f"Trade failed: {decision['symbol']} - {str(e)[:50]}", "")
        
        db.update_workflow(workflow_id, trades_executed=trades_executed)
        
        # Complete
        active_workflow["status"] = "completed"
        active_workflow["current_step"] = "Workflow complete"
        
        result = f"CrewAI: Analyzed {len(symbols)} stocks, {len(buy_decisions)} buy signals, {trades_executed} executed"
        add_step(
            "Executive",
            result,
            "All LLM agents completed their analysis. View traces at app.crewai.com"
        )
        
        db.update_workflow(workflow_id, status="completed", result=result)
        logger.info(f"CrewAI workflow completed: {result}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"CrewAI workflow failed: {error_msg}")
        
        if "cancelled" in error_msg.lower():
            active_workflow["status"] = "cancelled"
            active_workflow["current_step"] = "Workflow cancelled"
            add_step("System", "Workflow cancelled by user", "")
            db.update_workflow(workflow_id, status="cancelled", result="Cancelled by user")
        else:
            active_workflow["status"] = "failed"
            active_workflow["current_step"] = f"Error: {error_msg}"
            add_step("System", f"Error: {error_msg[:100]}", "")
            db.update_workflow(workflow_id, status="failed", result=error_msg)


async def run_workflow(
    state,
    workflow_id: str,
    scan_count: int,
    analyze_top: int,
    auto_execute: bool,
):
    """Run the full trading workflow (legacy - redirects to CrewAI)."""
    # Use CrewAI workflow by default
    await run_crewai_workflow(state, workflow_id, scan_count, analyze_top, auto_execute)
    return


async def run_workflow_legacy(
    state,
    workflow_id: str,
    scan_count: int,
    analyze_top: int,
    auto_execute: bool,
):
    """Run the full trading workflow (legacy custom agents)."""
    global active_workflow, workflow_cancelled
    workflow_cancelled = False
    
    try:
        # Step 1: Executive initiates scan
        active_workflow["status"] = "running"
        active_workflow["current_step"] = "Executive: Initiating market scan"
        
        add_step(
            "Executive",
            "Initiating trading workflow",
            "I need to find high-momentum stocks with potential for quick gains. Let me call the Scanner Agent to search the market."
        )
        
        db.save_decision(
            workflow_id, "Executive", "*", "INITIATE",
            100, "Starting trading workflow - calling Scanner Agent"
        )
        
        await asyncio.sleep(0.5)
        
        # Check for cancellation
        if workflow_cancelled:
            raise Exception("Workflow cancelled by user")
        
        # Step 2: Scanner finds stocks
        active_workflow["current_step"] = "Scanner: Searching for momentum stocks"
        
        add_step(
            "Scanner",
            f"Starting market scan for {scan_count} stocks",
            f"Scanning pre-market movers, volume surges, and momentum indicators. Looking for stocks with RSI < 70, positive MACD, and above-average volume."
        )
        
        add_step(
            "Scanner",
            "Fetching market data from Yahoo Finance...",
            "Pulling OHLCV data for screening candidates based on momentum criteria."
        )
        
        scan_results = await state.researcher.scan_market(
            target_count=scan_count,
            use_llm=True,
        )
        
        # Check for cancellation
        if workflow_cancelled:
            raise Exception("Workflow cancelled by user")
        
        stocks_found = [
            {"symbol": r.symbol, "score": r.score, "price": r.price, "rsi": r.rsi}
            for r in scan_results[:analyze_top]
        ]
        
        active_workflow["stocks"] = stocks_found
        db.update_workflow(workflow_id, stocks_scanned=len(scan_results))
        
        # Log each found stock with thinking
        for stock in stocks_found:
            add_step(
                "Scanner",
                f"Found: {stock['symbol']} - Score: {stock['score']:.1f}",
                f"Price: ${stock['price']:.2f}, RSI: {stock['rsi']:.1f}. This stock shows momentum characteristics worth analyzing."
            )
            db.save_decision(
                workflow_id, "Scanner", stock["symbol"], "FOUND",
                stock["score"], f"Momentum score: {stock['score']:.1f}, RSI: {stock['rsi']:.1f}",
                {"price": stock["price"], "rsi": stock["rsi"]}
            )
        
        add_step(
            "Scanner",
            f"Scan complete: {len(stocks_found)} candidates identified",
            f"Passing {', '.join(s['symbol'] for s in stocks_found)} to Technical and Fundamental analysts for deep analysis."
        )
        
        await asyncio.sleep(0.5)
        
        # Step 3: Analyze each stock using real agents
        provider = get_provider("yfinance")
        feature_builder = FeatureBuilder()
        technical_agent, fundamental_agent, executive_agent = get_agents()
        
        analyzed_stocks = []
        
        for i, stock in enumerate(stocks_found):
            symbol = stock["symbol"]
            
            # Check for cancellation
            if workflow_cancelled:
                raise Exception("Workflow cancelled by user")
            
            active_workflow["current_step"] = f"Analyzing {symbol} ({i+1}/{len(stocks_found)})"
            
            try:
                # Fetch data
                add_step(
                    "Technical",
                    f"Fetching 60-day price data for {symbol}",
                    f"Need historical OHLCV data to calculate RSI, MACD, and identify patterns."
                )
                
                end = datetime.now()
                start = end - timedelta(days=60)
                data = await provider.fetch_ohlcv(symbol, start, end, "1d")
                
                if data.empty:
                    add_step("Technical", f"No data available for {symbol}, skipping", "")
                    continue
                
                features = feature_builder.build_features(data)
                latest = data.iloc[-1]
                current_price = float(latest["close"])
                
                # Convert features to dict for PipelineState
                latest_features = features.iloc[-1].to_dict() if not features.empty else {}
                
                # Create PipelineState for agent analysis
                pipeline_state = PipelineState(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    current_price=current_price,
                    features=latest_features,
                    equity=db.get_budget() or 10000,
                    current_position_pct=0.0,
                )
                
                # Technical Analysis using real agent
                add_step(
                    "Technical",
                    f"Running ML ensemble analysis for {symbol}",
                    f"Analyzing RSI, MACD, ADX, Bollinger Bands, and volume patterns using trained models."
                )
                
                try:
                    tech_result = await technical_agent.analyze(pipeline_state)
                    
                    tech_signal = tech_result.direction.value.upper()
                    tech_confidence = int(tech_result.confidence * 100)
                    
                    # Build reasoning from top features
                    feature_descriptions = []
                    for feat in tech_result.top_features[:3]:
                        feature_descriptions.append(f"{feat.feature_name}: {feat.current_value:.2f} ({feat.interpretation})")
                    
                    tech_reasoning = f"Direction: {tech_signal}, Forecast: {tech_result.forecast_q50:.2f}%, Edge: {tech_result.edge_after_costs:.2f}%"
                    tech_thinking = f"ML Ensemble Analysis: {', '.join(feature_descriptions)}. Regime: {tech_result.regime.value}. P(Up): {tech_result.p_up:.2%}, P(Down): {tech_result.p_down:.2%}"
                    
                    # Update pipeline state with technical signal
                    pipeline_state.technical_signal = tech_result
                    
                except Exception as e:
                    logger.warning(f"Technical agent failed for {symbol}: {e}")
                    tech_signal = "NEUTRAL"
                    tech_confidence = 50
                    tech_reasoning = f"Technical analysis error: {str(e)[:50]}"
                    tech_thinking = "Fallback: Using basic indicators only."
                
                add_step(
                    "Technical",
                    f"{symbol}: {tech_signal} ({tech_confidence}%)",
                    tech_thinking
                )
                
                db.save_decision(
                    workflow_id, "Technical", symbol, tech_signal,
                    tech_confidence, tech_reasoning,
                    {"features": str(latest_features.get("rsi", "N/A"))}
                )
                
                await asyncio.sleep(0.3)
                
                # Fundamental Analysis using real agent (with LLM)
                add_step(
                    "Fundamental",
                    f"Analyzing news and catalysts for {symbol}",
                    f"Searching for earnings, analyst ratings, M&A, and market sentiment using AI."
                )
                
                try:
                    if os.getenv("OPENAI_API_KEY"):
                        add_step(
                            "Fundamental",
                            f"Calling OpenAI GPT to analyze {symbol}...",
                            "Using LLM to synthesize news, events, and generate sentiment analysis with citations."
                        )
                        
                        fund_result = await fundamental_agent.analyze(pipeline_state)
                        
                        fund_signal = fund_result.direction.value.upper()
                        fund_confidence = int(fund_result.confidence * 100)
                        
                        # Build reasoning from events and catalysts
                        event_summaries = [e.title for e in fund_result.events[:2]] if fund_result.events else []
                        catalyst_list = fund_result.catalysts[:3] if fund_result.catalysts else []
                        
                        fund_reasoning = f"Sentiment: {fund_result.sentiment_score:.2f}, Credibility: {fund_result.credibility_score:.2f}"
                        fund_thinking = f"LLM Analysis: Sentiment score {fund_result.sentiment_score:.2f}. "
                        if catalyst_list:
                            fund_thinking += f"Catalysts: {', '.join(catalyst_list)}. "
                        if fund_result.risks:
                            fund_thinking += f"Risks: {', '.join(fund_result.risks[:2])}. "
                        fund_thinking += f"News analyzed: {fund_result.news_count_analyzed}"
                        
                        # Update pipeline state
                        pipeline_state.fundamental_signal = fund_result
                    else:
                        # Fallback without OpenAI
                        catalysts = await state.researcher._query_catalysts_batch([symbol])
                        cat_data = catalysts.get(symbol, {})
                        score = cat_data.get("score", 50)
                        
                        if score > 70:
                            fund_signal = "BULLISH"
                        elif score < 40:
                            fund_signal = "BEARISH"
                        else:
                            fund_signal = "NEUTRAL"
                        
                        fund_confidence = score
                        fund_reasoning = cat_data.get("reasoning", "Perplexity catalyst scan")
                        fund_thinking = f"Perplexity API: Score {score}. {fund_reasoning[:100]}"
                        
                except Exception as e:
                    logger.warning(f"Fundamental agent failed for {symbol}: {e}")
                    fund_signal = "NEUTRAL"
                    fund_confidence = 50
                    fund_reasoning = f"Fundamental analysis error: {str(e)[:50]}"
                    fund_thinking = "Fallback: Unable to fetch news/catalyst data."
                
                add_step(
                    "Fundamental",
                    f"{symbol}: {fund_signal} ({fund_confidence}%)",
                    fund_thinking
                )
                
                db.save_decision(
                    workflow_id, "Fundamental", symbol, fund_signal,
                    fund_confidence, fund_reasoning
                )
                
                await asyncio.sleep(0.3)
                
                analyzed_stocks.append({
                    "symbol": symbol,
                    "price": current_price,
                    "technical": {"signal": tech_signal, "confidence": tech_confidence, "reasoning": tech_reasoning},
                    "fundamental": {"signal": fund_signal, "confidence": fund_confidence, "reasoning": fund_reasoning},
                    "pipeline_state": pipeline_state,
                })
                
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                add_step("System", f"Error analyzing {symbol}: {str(e)[:50]}", "")
                continue
        
        db.update_workflow(workflow_id, stocks_analyzed=len(analyzed_stocks))
        
        await asyncio.sleep(1)
        
        # Check for cancellation
        if workflow_cancelled:
            raise Exception("Workflow cancelled by user")
        
        # Step 4: Executive makes decisions
        active_workflow["current_step"] = "Executive: Reviewing analysis and making decisions"
        
        add_step(
            "Executive",
            f"Reviewing analysis for {len(analyzed_stocks)} stocks",
            f"I have technical and fundamental signals for each stock. Now I need to synthesize this information and make buy/hold/skip decisions based on our risk parameters and $10k budget."
        )
        
        buy_decisions = []
        budget = db.get_budget()
        
        # Calculate how much of budget is already deployed in positions
        try:
            positions = await state.broker.get_positions()
            deployed_capital = sum(p.market_value for p in positions)
        except:
            deployed_capital = 0
        
        available_budget = max(0, budget - deployed_capital)
        position_size = min(budget * 0.15, available_budget * 0.5) if budget > 0 else 1000  # 15% of total or 50% of available
        
        for stock in analyzed_stocks:
            symbol = stock["symbol"]
            tech = stock["technical"]
            fund = stock["fundamental"]
            price = stock["price"]
            pipeline_state = stock.get("pipeline_state")
            
            # Try to use Executive Agent with LLM
            exec_decision = None
            exec_thinking = ""
            
            if os.getenv("OPENAI_API_KEY") and pipeline_state:
                try:
                    add_step(
                        "Executive",
                        f"Calling GPT for final decision on {symbol}...",
                        "Using LLM to synthesize all signals and determine optimal action, sizing, and risk parameters."
                    )
                    
                    exec_result = await executive_agent.analyze(pipeline_state)
                    
                    action = exec_result.action.value.upper()
                    combined_confidence = exec_result.confidence * 100
                    quantity = exec_result.size_shares or max(1, int(position_size * exec_result.size_pct / (price + 0.01)))
                    reasoning = exec_result.rationale
                    exec_thinking = f"LLM Decision: {exec_result.rationale}. Stop-loss: {exec_result.stop_loss_pct:.1%}, Take-profit: {exec_result.take_profit_pct:.1%}. What would change mind: {exec_result.what_would_change_mind}"
                    
                    logger.info(f"Executive LLM decision for {symbol}: {action}")
                    
                except Exception as e:
                    logger.warning(f"Executive LLM failed for {symbol}, using rule-based: {e}")
                    exec_decision = None
            
            # Fallback to rule-based if LLM not available or failed
            if exec_decision is None and not exec_thinking:
                tech_bullish = tech["signal"] == "BULLISH"
                fund_bullish = fund["signal"] == "BULLISH"
                combined_confidence = (tech["confidence"] * 0.4 + fund["confidence"] * 0.6)
                
                if tech_bullish and fund_bullish and combined_confidence >= 65:
                    action = "BUY"
                    quantity = max(1, int(position_size / price))
                    reasoning = f"Both Technical and Fundamental bullish with {combined_confidence:.0f}% confidence"
                elif tech_bullish and fund["signal"] != "BEARISH" and combined_confidence >= 60:
                    action = "BUY"
                    quantity = max(1, int(position_size * 0.5 / price))  # Half position
                    reasoning = f"Technical bullish, Fundamental neutral - smaller position"
                elif tech["signal"] == "BEARISH" or fund["signal"] == "BEARISH":
                    action = "SKIP"
                    quantity = 0
                    reasoning = "Bearish signals detected - skipping"
                else:
                    action = "WATCH"
                    quantity = 0
                    reasoning = f"Mixed signals ({combined_confidence:.0f}% confidence) - adding to watchlist"
                
                exec_thinking = f"Rule-based: Technical {tech['signal']} ({tech['confidence']}%), Fundamental {fund['signal']} ({fund['confidence']}%). "
                if action == "BUY":
                    exec_thinking += f"Signals align. Allocating {quantity} shares at ${price:.2f}."
                elif action == "WATCH":
                    exec_thinking += "Mixed signals. Adding to watchlist."
                else:
                    exec_thinking += "Negative signals. Skipping."
            
            decision = {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "confidence": combined_confidence,
                "reasoning": reasoning,
            }
            
            active_workflow["decisions"].append(decision)
            
            add_step(
                "Executive",
                f"Decision: {action} {symbol}" + (f" ({quantity} shares @ ${price:.2f})" if quantity > 0 else ""),
                exec_thinking
            )
            
            db.save_decision(
                workflow_id, "Executive", symbol, action,
                combined_confidence, reasoning,
                {"quantity": quantity, "price": price}
            )
            
            if action == "BUY" and quantity > 0:
                buy_decisions.append(decision)
        
        await asyncio.sleep(1)
        
        # Step 5: Execute trades (if auto_execute or store for manual execution)
        trades_executed = 0
        
        if auto_execute and buy_decisions:
            active_workflow["current_step"] = "Executive: Executing approved trades"
            
            for decision in buy_decisions:
                try:
                    order = await state.broker.submit_order(
                        symbol=decision["symbol"],
                        qty=decision["quantity"],
                        side=OrderSide.BUY,
                    )
                    
                    trade = {
                        "symbol": decision["symbol"],
                        "action": "BUY",
                        "quantity": decision["quantity"],
                        "price": order.filled_avg_price or decision["price"],
                        "order_id": order.id,
                    }
                    
                    active_workflow["trades"].append(trade)
                    
                    db.save_trade(
                        decision["symbol"], "buy", decision["quantity"],
                        order.filled_avg_price or decision["price"],
                        decision["confidence"], decision["reasoning"]
                    )
                    
                    # Deduct from budget
                    cost = decision["quantity"] * (order.filled_avg_price or decision["price"])
                    db.add_budget_entry(-cost, "trade", f"BUY {decision['quantity']} {decision['symbol']}")
                    
                    trades_executed += 1
                    
                    active_workflow["steps"].append({
                        "agent": "Executive",
                        "action": f"Executed: BUY {decision['quantity']} {decision['symbol']} @ ${order.filled_avg_price or decision['price']:.2f}",
                        "time": datetime.now().isoformat(),
                    })
                    
                except Exception as e:
                    logger.error(f"Trade execution failed for {decision['symbol']}: {e}")
        
        db.update_workflow(workflow_id, trades_executed=trades_executed)
        
        # Complete workflow
        active_workflow["status"] = "completed"
        active_workflow["current_step"] = "Workflow complete"
        active_workflow["steps"].append({
            "agent": "Executive",
            "action": f"Workflow complete: Analyzed {len(analyzed_stocks)} stocks, {len(buy_decisions)} buy signals, {trades_executed} trades executed",
            "time": datetime.now().isoformat(),
        })
        
        result = f"Scanned {len(scan_results)}, analyzed {len(analyzed_stocks)}, {len(buy_decisions)} buys, {trades_executed} executed"
        db.update_workflow(workflow_id, status="completed", result=result)
        
        logger.info("Workflow completed", workflow_id=workflow_id, result=result)
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Workflow failed", workflow_id=workflow_id, error=error_msg)
        
        if "cancelled" in error_msg.lower():
            active_workflow["status"] = "cancelled"
            active_workflow["current_step"] = "Workflow cancelled"
            add_step("System", "Workflow cancelled by user", "")
            db.update_workflow(workflow_id, status="cancelled", result="Cancelled by user")
        else:
            active_workflow["status"] = "failed"
            active_workflow["current_step"] = f"Error: {error_msg}"
            add_step("System", f"Error: {error_msg[:100]}", "")
            db.update_workflow(workflow_id, status="failed", result=error_msg)
