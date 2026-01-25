"""CLI entry point for AI Trading System."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.config import get_settings, load_yaml_config
from src.utils.logging import setup_logging, get_logger


app = typer.Typer(
    name="trading",
    help="AI-driven multi-agent trading system",
    add_completion=False,
)

console = Console()


@app.command()
def run_backtest(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    symbol: str = typer.Option(
        "AAPL",
        "--symbol", "-s",
        help="Trading symbol",
    ),
    start_date: str = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD), defaults to 1 year ago",
    ),
    end_date: str = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD), defaults to today",
    ),
    interval: str = typer.Option(
        "1h",
        "--interval", "-i",
        help="Bar interval (1m, 5m, 15m, 1h, 1d)",
    ),
    output_dir: str = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for results",
    ),
):
    """Run a backtest on historical data."""
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("cli")

    console.print(Panel.fit(
        "[bold blue]AI Trading System - Backtest[/bold blue]",
        border_style="blue",
    ))

    # Parse dates
    if end_date is None:
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        start = end - timedelta(days=365)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d")

    console.print(f"Symbol: [bold]{symbol}[/bold]")
    console.print(f"Period: {start.date()} to {end.date()}")
    console.print(f"Interval: {interval}")

    # Run backtest
    async def _run():
        from src.data.market_data import get_provider
        from src.data.feature_store import FeatureStore
        from src.backtest.engine import BacktestConfig, BacktestEngine
        from src.orchestrator.contracts import (
            Action, ExecutiveDecision, OrderType, PipelineState
        )

        # Load config
        settings = get_settings(config)

        # Get market data
        provider = get_provider("yfinance")
        data = await provider.fetch_ohlcv(symbol, start, end, interval)

        if data.empty:
            console.print("[red]No data returned[/red]")
            return

        console.print(f"Loaded [bold]{len(data)}[/bold] bars")

        # Build features
        feature_store = FeatureStore(Path("./data/features"))
        features = await feature_store.get_features(symbol, start, end, data)

        console.print(f"Computed [bold]{len(features.columns) - 7}[/bold] features")

        # Simple momentum strategy for demo
        async def simple_strategy(state: PipelineState) -> ExecutiveDecision | None:
            """Simple momentum strategy for demonstration."""
            if not state.features:
                return None

            rsi = state.features.get("rsi", 50)
            macd_hist = state.features.get("macd_hist", 0)

            # Skip if NaN
            if rsi != rsi or macd_hist != macd_hist:
                return None

            action = Action.HOLD
            confidence = 0.5

            # Oversold with positive momentum
            if rsi < 30 and macd_hist > 0:
                action = Action.BUY
                confidence = 0.7
            # Overbought with negative momentum
            elif rsi > 70 and macd_hist < 0:
                action = Action.SELL
                confidence = 0.7

            if action == Action.HOLD:
                return None

            import uuid
            return ExecutiveDecision(
                action=action,
                symbol=state.symbol,
                size_pct=0.1,
                entry_type=OrderType.MARKET,
                time_horizon="1d",
                confidence=confidence,
                rationale=f"RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}",
                what_would_change_mind="RSI reversal or MACD flip",
                decision_id=str(uuid.uuid4()),
                timestamp=state.timestamp,
            )

        # Run backtest
        backtest_config = BacktestConfig(
            initial_capital=100_000.0,
            slippage_bps=5.0,
        )

        engine = BacktestEngine(backtest_config, simple_strategy)
        results = await engine.run(symbol, data, features)

        # Display results
        metrics = results["metrics"]

        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Return", f"{metrics['total_return']:.2%}")
        table.add_row("CAGR", f"{metrics['cagr']:.2%}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        table.add_row("Total Trades", str(metrics['total_trades']))
        table.add_row("Win Rate", f"{metrics['win_rate']:.2%}")
        table.add_row("Final Equity", f"${metrics['final_equity']:,.2f}")

        console.print(table)

        # Save results if output dir specified
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save equity curve
            results["equity_curve"].to_parquet(out_path / "equity_curve.parquet")

            console.print(f"\nResults saved to: {out_path}")

    asyncio.run(_run())


@app.command()
def run_historical_backtest(
    symbol: str = typer.Option(
        "AAPL",
        "--symbol", "-s",
        help="Trading symbol",
    ),
    interval: str = typer.Option(
        "1h",
        "--interval", "-i",
        help="Bar interval (1m, 5m, 15m, 1h, 1d)",
    ),
    train_days: int = typer.Option(
        60,
        "--train-days",
        help="Number of days to use for training",
    ),
    test_days: int = typer.Option(
        30,
        "--test-days",
        help="Number of days to use for testing",
    ),
    end_date: str = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD), defaults to today",
    ),
    model_dir: str = typer.Option(
        "./models/technical",
        "--model-dir",
        help="Directory for technical models",
    ),
    save_models: bool = typer.Option(
        True,
        "--save-models/--no-save-models",
        help="Save trained models to disk",
    ),
    skip_train: bool = typer.Option(
        False,
        "--skip-train",
        help="Skip training and use any existing models on disk",
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence",
        help="Minimum confidence to trade",
    ),
    max_position_pct: float = typer.Option(
        0.15,
        "--max-position-pct",
        help="Maximum position size as fraction of equity",
    ),
    initial_capital: float = typer.Option(
        100_000.0,
        "--initial-capital",
        help="Initial capital for backtest",
    ),
    slippage_bps: float = typer.Option(
        5.0,
        "--slippage-bps",
        help="Slippage in basis points",
    ),
    output_dir: str = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for results",
    ),
):
    """Run the 60/30 historical backtesting flow."""
    setup_logging(level="INFO")
    logger = get_logger("cli")

    console.print(Panel.fit(
        "[bold blue]AI Trading System - Historical Backtest[/bold blue]",
        border_style="blue",
    ))

    if end_date is None:
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d")

    start = end - timedelta(days=train_days + test_days)
    split_date = start + timedelta(days=train_days)

    console.print(f"Symbol: [bold]{symbol}[/bold]")
    console.print(f"Interval: {interval}")
    console.print(f"Train/Test: {train_days}d / {test_days}d")
    console.print(f"Period: {start.date()} to {end.date()}")
    console.print(f"Split date: {split_date.date()}")

    async def _run():
        import pandas as pd

        from src.agents.executive import ExecutiveAgent
        from src.agents.risk_guardian import RiskGuardian
        from src.agents.technical import TechnicalAnalystAgent
        from src.backtest.engine import BacktestConfig, BacktestEngine
        from src.data.feature_store import FeatureBuilder
        from src.data.market_data import get_provider
        from src.orchestrator.crew import SimpleOrchestrator
        from src.orchestrator.contracts import PipelineState

        provider = get_provider("yfinance")
        data = await provider.fetch_ohlcv(symbol, start, end, interval)

        if data.empty:
            console.print("[red]No data returned[/red]")
            return

        data["timestamp"] = pd.to_datetime(data["timestamp"])
        if data["timestamp"].dt.tz is not None:
            data["timestamp"] = data["timestamp"].dt.tz_localize(None)
        data = data.sort_values("timestamp").reset_index(drop=True)

        feature_builder = FeatureBuilder()
        features_df = feature_builder.build_features(data)
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
        if features_df["timestamp"].dt.tz is not None:
            features_df["timestamp"] = features_df["timestamp"].dt.tz_localize(None)
        features_df = features_df.sort_values("timestamp").reset_index(drop=True)

        train_mask = features_df["timestamp"] < split_date
        train_features = features_df[train_mask].reset_index(drop=True)
        test_features = features_df[~train_mask].reset_index(drop=True)
        test_data = data[~train_mask].reset_index(drop=True)

        if train_features.empty or test_data.empty:
            console.print("[red]Insufficient data after train/test split[/red]")
            return

        console.print(f"Train bars: [bold]{len(train_features)}[/bold]")
        console.print(f"Test bars: [bold]{len(test_data)}[/bold]")

        model_path = Path(model_dir) if model_dir else None
        technical_agent = TechnicalAnalystAgent(model_dir=model_path)

        if skip_train:
            await technical_agent.load_model()
            console.print("[yellow]Skipped training, loaded models from disk[/yellow]")
        else:
            console.print("[cyan]Training technical models...[/cyan]")
            train_metrics = await technical_agent.train_models(
                train_features,
                save_models=save_models,
            )
            if train_metrics.get("models_trained"):
                console.print(f"[green]Trained models: {', '.join(train_metrics['models_trained'])}[/green]")
            else:
                console.print("[yellow]No models trained, using rule-based signals[/yellow]")

        executive_agent = ExecutiveAgent(
            use_llm=False,
            min_confidence=min_confidence,
            max_position_pct=max_position_pct,
        )
        risk_guardian = RiskGuardian(
            max_position_pct=max_position_pct,
            min_confidence=min_confidence,
        )
        orchestrator = SimpleOrchestrator(
            min_confidence=min_confidence,
            max_position_pct=max_position_pct,
            technical_agent=technical_agent,
            executive_agent=executive_agent,
            risk_guardian=risk_guardian,
        )

        async def decision_callback(state: PipelineState):
            return await orchestrator.run(state)

        backtest_config = BacktestConfig(
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
            max_position_pct=max_position_pct,
        )
        engine = BacktestEngine(backtest_config, decision_callback)
        results = await engine.run(symbol, test_data, test_features)

        metrics = results["metrics"]
        capital_gain = metrics["final_equity"] - metrics["initial_capital"]

        table = Table(title="Historical Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Return", f"{metrics['total_return']:.2%}")
        table.add_row("Capital Gain", f"${capital_gain:,.2f}")
        table.add_row("Success Rate", f"{metrics['win_rate']:.2%}")
        table.add_row("Closed Trades", str(metrics["closed_trades"]))
        table.add_row("Total Trades", str(metrics["total_trades"]))
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Final Equity", f"${metrics['final_equity']:,.2f}")

        console.print(table)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            results["equity_curve"].to_parquet(out_path / "equity_curve.parquet")
            pd.DataFrame(results["decisions"]).to_parquet(out_path / "decisions.parquet")

            console.print(f"\nResults saved to: {out_path}")

    asyncio.run(_run())


@app.command()
def run_paper(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    symbols: str = typer.Option(
        "AAPL,GOOGL,MSFT",
        "--symbols", "-s",
        help="Trading symbols (comma-separated)",
    ),
    interval: int = typer.Option(
        60,
        "--interval", "-i",
        help="Minutes between trading cycles",
    ),
    paper: bool = typer.Option(
        True,
        "--paper/--live",
        help="Use paper trading (default) or live trading",
    ),
    train: bool = typer.Option(
        True,
        "--train/--no-train",
        help="Train ML models before trading",
    ),
    cycles: int = typer.Option(
        0,
        "--cycles", "-n",
        help="Number of cycles (0=unlimited)",
    ),
    simulate: bool = typer.Option(
        False,
        "--simulate",
        help="Use simulated broker instead of Alpaca",
    ),
):
    """Run paper trading with live data and real LLM agents.
    
    This runs the full multi-agent pipeline:
    - Technical Analyst (ML models + rule-based)
    - Fundamental Analyst (OpenAI + news analysis)
    - Hybrid Analyst (signal synthesis with OpenAI)
    - Executive Agent (final decision with OpenAI)
    - Risk Guardian (deterministic safety checks)
    
    Examples:
        python -m src.cli run-paper -s AAPL,NVDA --cycles 1
        python -m src.cli run-paper -s TSLA --simulate --no-train
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    setup_logging(level="INFO")
    logger = get_logger("cli")

    console.print(Panel.fit(
        "[bold green]AI Trading System - Paper Trading[/bold green]",
        border_style="green",
    ))

    symbol_list = [s.strip() for s in symbols.split(",")]
    console.print(f"Symbols: [bold]{', '.join(symbol_list)}[/bold]")
    console.print(f"Interval: {interval} minutes")
    console.print(f"Mode: [bold]{'PAPER' if paper else 'LIVE'}[/bold]")
    console.print(f"Train Models: [bold]{'Yes' if train else 'No'}[/bold]")

    if not paper:
        if not typer.confirm("\n⚠️  LIVE TRADING MODE - Are you sure?"):
            console.print("[red]Aborted.[/red]")
            return

    # Check API keys
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    if not openai_key:
        console.print("[red]Error: OPENAI_API_KEY required for LLM agents[/red]")
        console.print("Set this in your .env file")
        return

    if not simulate and (not api_key or not api_secret):
        console.print("[yellow]Warning: ALPACA keys missing, using simulated broker[/yellow]")
        simulate = True

    async def _run_paper():
        from src.data.market_data import get_provider
        from src.data.feature_store import FeatureBuilder
        from src.agents.technical import TechnicalAnalystAgent
        from src.agents.fundamental import FundamentalAnalystAgent
        from src.agents.hybrid import HybridAnalystAgent
        from src.agents.executive import ExecutiveAgent
        from src.agents.risk_guardian import RiskGuardian
        from src.orchestrator.crew import AgentOrchestrator
        from src.orchestrator.contracts import Action, PipelineState
        from src.execution.broker import AlpacaBroker, SimulatedBroker, OrderSide, OrderType as BrokerOrderType

        # Initialize broker
        if simulate:
            broker = SimulatedBroker(initial_cash=100_000)
            console.print("[yellow]Using simulated broker[/yellow]")
        else:
            broker = AlpacaBroker(api_key=api_key, api_secret=api_secret, paper=paper)
            console.print("[green]Using Alpaca paper trading[/green]")

        provider = get_provider("yfinance")
        feature_builder = FeatureBuilder()
        model_dir = Path("./models/technical")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Technical Agent
        technical_agent = TechnicalAnalystAgent(model_dir=model_dir, threshold_pct=0.5)

        # Train models if requested
        if train:
            console.print("\n[bold]Training ML Models...[/bold]")
            for symbol in symbol_list:
                end = datetime.now()
                start = end - timedelta(days=365)
                data = await provider.fetch_ohlcv(symbol, start, end, "1d")
                if not data.empty:
                    features_df = feature_builder.build_features(data)
                    metrics = await technical_agent.train_models(features_df, save_models=True)
                    if metrics.get("models_trained"):
                        console.print(f"  [green]✓ {symbol}:[/green] {', '.join(metrics['models_trained'])}")

        # Initialize LLM agents
        fundamental_agent = FundamentalAnalystAgent(
            openai_api_key=openai_key,
            perplexity_api_key=perplexity_key,
            enable_live_search=False,
            config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
        )
        hybrid_agent = HybridAnalystAgent(
            openai_api_key=openai_key,
            use_llm=True,
            config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
        )
        executive_agent = ExecutiveAgent(
            openai_api_key=openai_key,
            use_llm=True,
            min_confidence=0.5,
            max_position_pct=0.15,
            config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
        )
        risk_guardian = RiskGuardian(
            max_position_pct=0.20,
            max_daily_loss_pct=0.05,
            max_trades_per_day=10,
            min_confidence=0.5,
        )

        orchestrator = AgentOrchestrator(
            technical_agent=technical_agent,
            fundamental_agent=fundamental_agent,
            hybrid_agent=hybrid_agent,
            executive_agent=executive_agent,
            risk_guardian=risk_guardian,
            openai_api_key=openai_key,
            config={"use_llm": True},
        )

        # Trading loop
        cycle_count = 0
        while cycles == 0 or cycle_count < cycles:
            cycle_count += 1
            console.print(f"\n[bold cyan]━━━ Cycle {cycle_count} ━━━[/bold cyan]")

            try:
                account = await broker.get_account()
                console.print(f"Equity: ${account.equity:,.2f}")
            except Exception:
                account = None

            for symbol in symbol_list:
                console.print(f"\n[bold]Processing {symbol}[/bold]")
                
                try:
                    # Get data and features
                    end = datetime.now()
                    start = end - timedelta(days=30)
                    data = await provider.fetch_ohlcv(symbol, start, end, "1h")
                    
                    if data.empty:
                        continue
                    
                    features_df = feature_builder.build_features(data)
                    latest_features = features_df.iloc[-1].to_dict()
                    current_price = float(data.iloc[-1]["close"])
                    
                    position = await broker.get_position(symbol)
                    current_qty = position.qty if position else 0
                    equity = account.equity if account else 100_000
                    
                    state = PipelineState(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        interval="1h",
                        current_price=current_price,
                        last_n_bars=data.tail(20).to_dict("records"),
                        features=latest_features,
                        current_position=current_qty,
                        current_position_pct=(current_qty * current_price / equity) if equity > 0 else 0,
                        cash=account.cash if account else 100_000,
                        equity=equity,
                        daily_pnl=0.0,
                        trades_today=0,
                    )
                    
                    # Run pipeline
                    state = await orchestrator.run_pipeline(state)
                    
                    # Log result
                    if state.executive_decision:
                        d = state.executive_decision
                        console.print(f"  Decision: [bold]{d.action.value}[/bold] | Confidence: {d.confidence:.0%}")
                        if d.action != Action.HOLD:
                            console.print(f"  Rationale: {d.rationale[:100]}...")
                    
                    # Execute if approved
                    if (state.final_action and 
                        state.final_action != Action.HOLD and 
                        state.risk_guardian_output and 
                        state.risk_guardian_output.approved):
                        
                        decision = state.executive_decision
                        
                        if isinstance(broker, SimulatedBroker):
                            broker.set_price(symbol, current_price)
                        
                        if decision.action == Action.BUY:
                            qty = int((equity * decision.size_pct) / current_price)
                            if qty > 0:
                                order = await broker.submit_order(symbol, qty, OrderSide.BUY, BrokerOrderType.MARKET)
                                console.print(f"  [green]BUY {qty} shares @ ${current_price:.2f}[/green]")
                        
                        elif decision.action == Action.SELL and current_qty > 0:
                            qty = int(current_qty * decision.size_pct)
                            if qty > 0:
                                order = await broker.submit_order(symbol, qty, OrderSide.SELL, BrokerOrderType.MARKET)
                                console.print(f"  [red]SELL {qty} shares @ ${current_price:.2f}[/red]")
                                
                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")

            # Wait for next cycle
            if cycles == 0 or cycle_count < cycles:
                console.print(f"\n[dim]Next cycle in {interval} minute(s)... Press Ctrl+C to stop[/dim]")
                await asyncio.sleep(interval * 60)

    try:
        console.print("\n[green]Starting paper trading...[/green]")
        asyncio.run(_run_paper())
    except KeyboardInterrupt:
        console.print("\n[yellow]Paper trading stopped.[/yellow]")


@app.command()
def train_rl(
    env_type: str = typer.Option(
        "execution",
        "--env", "-e",
        help="RL environment (execution, sizing)",
    ),
    timesteps: int = typer.Option(
        100000,
        "--timesteps", "-n",
        help="Total training timesteps",
    ),
    algo: str = typer.Option(
        "ppo",
        "--algo", "-a",
        help="RL algorithm (ppo, cql, bc)",
    ),
    output: str = typer.Option(
        "./models/rl",
        "--output", "-o",
        help="Output directory for trained model",
    ),
    symbol: str = typer.Option(
        "AAPL",
        "--symbol", "-s",
        help="Symbol for training data",
    ),
):
    """Train RL agent."""
    setup_logging(level="INFO")
    logger = get_logger("cli")

    console.print(Panel.fit(
        "[bold magenta]AI Trading System - RL Training[/bold magenta]",
        border_style="magenta",
    ))

    console.print(f"Environment: [bold]{env_type}[/bold]")
    console.print(f"Algorithm: [bold]{algo.upper()}[/bold]")
    console.print(f"Timesteps: [bold]{timesteps:,}[/bold]")
    console.print(f"Symbol: [bold]{symbol}[/bold]")

    async def _train():
        from src.data.market_data import get_provider
        from src.rl.trainer import RLTrainer

        # Get training data
        end = datetime.now()
        start = end - timedelta(days=365)

        console.print("\n[cyan]Fetching training data...[/cyan]")
        provider = get_provider("yfinance")
        data = await provider.fetch_ohlcv(symbol, start, end, "1h")

        if data.empty:
            console.print("[red]No data available[/red]")
            return

        console.print(f"Loaded [bold]{len(data)}[/bold] bars")

        # Initialize trainer
        trainer = RLTrainer(
            env_type=env_type,
            algorithm=algo,
            model_dir=Path(output),
        )

        console.print("\n[cyan]Starting training...[/cyan]")

        with console.status("[bold green]Training..."):
            trainer.train(
                data=data,
                total_timesteps=timesteps,
            )

        console.print(f"\n[green]✓ Model saved to {output}[/green]")

    asyncio.run(_train())


@app.command()
def scan_market(
    count: int = typer.Option(
        500,
        "--count", "-n",
        help="Number of stocks to return",
    ),
    output: str = typer.Option(
        None,
        "--output", "-o",
        help="Output CSV file path",
    ),
    top: int = typer.Option(
        20,
        "--top", "-t",
        help="Show top N results in console",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM catalyst detection (faster, cheaper)",
    ),
):
    """Scan market for momentum stocks.
    
    Uses the Researcher Agent to find stocks likely to jump 5-20% 
    in the next 2-5 trading days.
    """
    import os
    setup_logging(level="INFO")
    logger = get_logger("cli")

    console.print(Panel.fit(
        "[bold yellow]AI Trading System - Market Scanner[/bold yellow]",
        border_style="yellow",
    ))

    console.print(f"Target stocks: [bold]{count}[/bold]")
    console.print(f"Using LLM: [bold]{'No' if no_llm else 'Yes'}[/bold]")

    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    async def _scan():
        from src.agents.researcher import ResearcherAgent

        researcher = ResearcherAgent(
            openai_api_key=openai_key,
            perplexity_api_key=perplexity_key if not no_llm else None,
        )

        with console.status("[bold cyan]Scanning market..."):
            results = await researcher.scan_market(
                target_count=count,
                use_llm=not no_llm,
            )

        # Display top results
        table = Table(title=f"Top {top} Momentum Candidates")
        table.add_column("Rank", style="dim")
        table.add_column("Symbol", style="cyan bold")
        table.add_column("Score", style="green")
        table.add_column("Price", style="white")
        table.add_column("RSI", style="yellow")
        table.add_column("Trend", style="magenta")
        table.add_column("Vol Surge", style="blue")

        for i, r in enumerate(results[:top], 1):
            trend_color = "green" if r.short_term_trend == "bullish" else "red" if r.short_term_trend == "bearish" else "white"
            table.add_row(
                str(i),
                r.symbol,
                f"{r.score:.1f}",
                f"${r.price:.2f}",
                f"{r.rsi:.1f}",
                f"[{trend_color}]{r.short_term_trend}[/{trend_color}]",
                f"{r.volume_surge:+.1f}%",
            )

        console.print(table)

        # Show catalysts if available
        if results and results[0].catalysts:
            console.print("\n[bold]Top Catalysts:[/bold]")
            for r in results[:5]:
                if r.catalysts:
                    console.print(f"  [cyan]{r.symbol}[/cyan]: {', '.join(r.catalysts[:2])}")

        # Save to CSV
        if output:
            import pandas as pd
            df = pd.DataFrame([{
                "rank": i,
                "symbol": r.symbol,
                "score": r.score,
                "price": r.price,
                "rsi": r.rsi,
                "trend": r.short_term_trend,
                "volume_surge": r.volume_surge,
                "catalysts": "; ".join(r.catalysts),
            } for i, r in enumerate(results, 1)])
            df.to_csv(output, index=False)
            console.print(f"\n[green]Saved {len(results)} stocks to {output}[/green]")

        return results

    asyncio.run(_scan())


@app.command()
def run_discovery(
    scan_count: int = typer.Option(
        50,
        "--scan", "-s",
        help="Number of stocks to scan",
    ),
    analyze_count: int = typer.Option(
        5,
        "--analyze", "-a",
        help="Number of top stocks to fully analyze",
    ),
    output: str = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file path",
    ),
):
    """Run integrated discovery pipeline.
    
    Scanner → Technical → Fundamental → Hybrid → Executive → Risk Guardian
    
    This is the full end-to-end flow: discover stocks, analyze them
    through all agents, and get actionable recommendations.
    """
    import os
    setup_logging(level="INFO")
    logger = get_logger("cli")

    console.print(Panel.fit(
        "[bold cyan]AI Trading System - Discovery Pipeline[/bold cyan]",
        border_style="cyan",
    ))

    console.print(f"Scanning: [bold]{scan_count}[/bold] stocks")
    console.print(f"Analyzing: [bold]{analyze_count}[/bold] top picks")

    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    async def _run():
        from src.orchestrator.discovery_pipeline import DiscoveryPipeline

        pipeline = DiscoveryPipeline(
            openai_api_key=openai_key,
            perplexity_api_key=perplexity_key,
            max_stocks_to_analyze=analyze_count,
        )

        console.print("\n[cyan]Running discovery pipeline...[/cyan]")
        console.print("[dim]Stage 1: Scanning market[/dim]")
        console.print("[dim]Stage 2: Technical analysis[/dim]")
        console.print("[dim]Stage 3: Fundamental analysis[/dim]")
        console.print("[dim]Stage 4: Hybrid synthesis[/dim]")
        console.print("[dim]Stage 5: Executive decision[/dim]")
        console.print("[dim]Stage 6: Risk guardian check[/dim]\n")

        results = await pipeline.run(
            scan_count=scan_count,
            analyze_top_n=analyze_count,
        )

        # Display results
        table = Table(title="Discovery Pipeline Results")
        table.add_column("Symbol", style="cyan bold")
        table.add_column("Scan Score", style="yellow")
        table.add_column("Technical", style="blue")
        table.add_column("Fundamental", style="green")
        table.add_column("Action", style="magenta bold")
        table.add_column("Approved", style="white")

        for r in results:
            tech = r.technical_signal.direction.value if r.technical_signal else "-"
            fund = r.fundamental_signal.direction.value if r.fundamental_signal else "-"
            action = r.final_action.value if r.final_action else "NONE"
            approved = "✓" if r.approved else "✗"
            approved_style = "green" if r.approved else "red"

            table.add_row(
                r.symbol,
                f"{r.scan_result.score:.1f}",
                tech,
                fund,
                action,
                f"[{approved_style}]{approved}[/{approved_style}]",
            )

        console.print(table)

        # Show approved recommendations
        approved = [r for r in results if r.approved and r.final_action]
        if approved:
            console.print("\n[bold green]📈 Approved Recommendations:[/bold green]")
            for r in approved:
                console.print(f"  [cyan]{r.symbol}[/cyan]: {r.final_action.value}")
                if r.executive_decision:
                    console.print(f"    Size: {r.executive_decision.size_pct:.1%}")
                    console.print(f"    Confidence: {r.executive_decision.confidence:.0%}")
                    console.print(f"    Rationale: {r.executive_decision.rationale[:100]}...")
        else:
            console.print("\n[yellow]No approved recommendations at this time.[/yellow]")

        # Save to JSON
        if output:
            import json
            data = [r.to_dict() for r in results]
            with open(output, "w") as f:
                json.dump(data, f, indent=2, default=str)
            console.print(f"\n[green]Saved results to {output}[/green]")

        return results

    asyncio.run(_run())


@app.command()
def run_autonomous(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration file",
    ),
    scan_count: int = typer.Option(
        100,
        "--scan", "-s",
        help="Number of stocks to scan",
    ),
    analyze_count: int = typer.Option(
        10,
        "--analyze", "-a",
        help="Number of top stocks to fully analyze",
    ),
    trade_count: int = typer.Option(
        5,
        "--trade", "-t",
        help="Max stocks to hold simultaneously",
    ),
    interval: int = typer.Option(
        30,
        "--interval", "-i",
        help="Minutes between trading cycles",
    ),
    cycles: int = typer.Option(
        0,
        "--cycles", "-n",
        help="Number of cycles (0=unlimited)",
    ),
    rescan_every: int = typer.Option(
        4,
        "--rescan",
        help="Rescan market every N cycles (0=never)",
    ),
    paper: bool = typer.Option(
        True,
        "--paper/--live",
        help="Use paper trading (default) or live trading",
    ),
    train: bool = typer.Option(
        True,
        "--train/--no-train",
        help="Train ML models before trading",
    ),
    simulate: bool = typer.Option(
        False,
        "--simulate",
        help="Use simulated broker instead of Alpaca",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed agent reasoning",
    ),
):
    """Run autonomous day trading with auto-discovery.
    
    This is the full MarketSenseAI 2.0 flow:
    1. Scanner discovers momentum stocks (uses Perplexity for catalysts)
    2. Full agent crew analyzes each pick
    3. Execute approved trades via paper/live trading
    4. Monitor positions and exit on targets
    
    All agent activity is traced via:
    - CrewAI AMP (view at app.crewai.com)
    - MLflow (view at localhost:5000)
    - Local logs (./logs/traces/)
    
    Examples:
        python -m src.cli run-autonomous --cycles 3 --verbose
        python -m src.cli run-autonomous --simulate --no-train
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    setup_logging(level="DEBUG" if verbose else "INFO")
    logger = get_logger("cli")

    # Initialize tracing
    from src.observability.tracing import initialize_tracing, get_tracing_manager
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "trading_agents")
    
    tracing = initialize_tracing(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        enable_crewai=True,
        enable_mlflow=True,
        enable_file_logging=True,
    )
    
    console.print(Panel.fit(
        "[bold magenta]AI Trading System - Autonomous Day Trading[/bold magenta]\n"
        "[dim]MarketSenseAI 2.0 Flow[/dim]",
        border_style="magenta",
    ))
    
    console.print("[dim]Tracing enabled: CrewAI AMP + MLflow + Local logs[/dim]")

    console.print(f"[bold]Configuration:[/bold]")
    console.print(f"  Scan: {scan_count} stocks → Analyze: {analyze_count} → Trade: {trade_count}")
    console.print(f"  Cycle Interval: {interval} minutes")
    console.print(f"  Rescan Every: {rescan_every} cycles" if rescan_every > 0 else "  Rescan: Never")
    console.print(f"  Mode: [bold]{'PAPER' if paper else '⚠️  LIVE'}[/bold]")
    console.print(f"  Train Models: {'Yes' if train else 'No'}")

    if not paper:
        if not typer.confirm("\n⚠️  LIVE TRADING MODE - Are you sure?"):
            console.print("[red]Aborted.[/red]")
            return

    # Check API keys
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    if not openai_key:
        console.print("[red]Error: OPENAI_API_KEY required for LLM agents[/red]")
        return

    if not simulate and (not api_key or not api_secret):
        console.print("[yellow]Warning: ALPACA keys missing, using simulated broker[/yellow]")
        simulate = True

    async def _run_autonomous():
        from src.data.market_data import get_provider
        from src.data.feature_store import FeatureBuilder
        from src.agents.technical import TechnicalAnalystAgent
        from src.agents.fundamental import FundamentalAnalystAgent
        from src.agents.hybrid import HybridAnalystAgent
        from src.agents.executive import ExecutiveAgent
        from src.agents.risk_guardian import RiskGuardian
        from src.agents.researcher import ResearcherAgent
        from src.orchestrator.crew import AgentOrchestrator
        from src.orchestrator.contracts import Action, PipelineState
        from src.execution.broker import AlpacaBroker, SimulatedBroker, OrderSide, OrderType as BrokerOrderType

        # Start tracing session
        session_id = tracing.start_session(f"autonomous_{'paper' if paper else 'live'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        console.print(f"[dim]Tracing session: {session_id}[/dim]")

        # Initialize broker
        if simulate:
            broker = SimulatedBroker(initial_cash=100_000)
            console.print("\n[yellow]Using simulated broker ($100,000 starting capital)[/yellow]")
        else:
            broker = AlpacaBroker(api_key=api_key, api_secret=api_secret, paper=paper)
            console.print(f"\n[green]Using Alpaca {'paper' if paper else 'LIVE'} trading[/green]")

        provider = get_provider("yfinance")
        feature_builder = FeatureBuilder()
        model_dir = Path("./models/technical")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Technical Agent
        technical_agent = TechnicalAnalystAgent(model_dir=model_dir, threshold_pct=0.5)

        # Initialize LLM agents
        fundamental_agent = FundamentalAnalystAgent(
            openai_api_key=openai_key,
            perplexity_api_key=perplexity_key,
            enable_live_search=True,  # Enable for day trading
            config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
        )
        hybrid_agent = HybridAnalystAgent(
            openai_api_key=openai_key,
            use_llm=True,
            config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
        )
        executive_agent = ExecutiveAgent(
            openai_api_key=openai_key,
            use_llm=True,
            min_confidence=0.6,  # Higher threshold for day trading
            max_position_pct=0.20,  # Allow larger positions for momentum
            config={"model": "gpt-4-turbo-preview", "temperature": 0.1},
        )
        risk_guardian = RiskGuardian(
            max_position_pct=0.25,
            max_daily_loss_pct=0.03,  # Tighter loss limit for day trading
            max_trades_per_day=20,
            min_confidence=0.55,
        )

        orchestrator = AgentOrchestrator(
            technical_agent=technical_agent,
            fundamental_agent=fundamental_agent,
            hybrid_agent=hybrid_agent,
            executive_agent=executive_agent,
            risk_guardian=risk_guardian,
            openai_api_key=openai_key,
            config={"use_llm": True},
        )

        # Initialize Scanner
        researcher = ResearcherAgent(
            openai_api_key=openai_key,
            perplexity_api_key=perplexity_key,
        )

        # Track state
        active_symbols: list[str] = []
        cycle_count = 0
        total_trades = 0
        daily_pnl = 0.0

        async def scan_and_select():
            """Scan market and select top momentum stocks."""
            console.print("\n[bold cyan]🔍 Scanning Market for Momentum Stocks...[/bold cyan]")
            
            with console.status("[bold cyan]Running scanner (this may take 30-60 seconds)..."):
                results = await researcher.scan_market(
                    target_count=scan_count,
                    use_llm=perplexity_key is not None,
                )
            
            if not results:
                console.print("[yellow]No stocks found by scanner[/yellow]")
                return []
            
            console.print(f"[green]Found {len(results)} candidates[/green]")
            
            # Display top picks
            table = Table(title=f"Top {min(analyze_count, len(results))} Momentum Candidates")
            table.add_column("#", style="dim")
            table.add_column("Symbol", style="cyan bold")
            table.add_column("Score", style="green")
            table.add_column("RSI", style="yellow")
            table.add_column("Trend", style="magenta")
            table.add_column("Catalysts", style="white", max_width=40)
            
            for i, r in enumerate(results[:analyze_count], 1):
                table.add_row(
                    str(i),
                    r.symbol,
                    f"{r.score:.1f}",
                    f"{r.rsi:.1f}",
                    r.short_term_trend,
                    ", ".join(r.catalysts[:2]) if r.catalysts else "-",
                )
            
            console.print(table)
            
            return [r.symbol for r in results[:trade_count]]

        # Train models if requested
        if train:
            console.print("\n[bold]📚 Training ML Models...[/bold]")
            # Get initial symbols for training
            training_symbols = ["AAPL", "NVDA", "GOOGL", "MSFT", "AMZN"]
            for symbol in training_symbols:
                end = datetime.now()
                start = end - timedelta(days=365)
                data = await provider.fetch_ohlcv(symbol, start, end, "1d")
                if not data.empty:
                    features_df = feature_builder.build_features(data)
                    metrics = await technical_agent.train_models(features_df, save_models=True)
                    if metrics.get("models_trained"):
                        console.print(f"  [green]✓ {symbol}[/green]")

        # Initial scan
        active_symbols = await scan_and_select()
        
        if not active_symbols:
            console.print("[red]No stocks selected, cannot proceed[/red]")
            return

        console.print(f"\n[bold green]🎯 Trading Watchlist: {', '.join(active_symbols)}[/bold green]")

        # Trading loop
        while cycles == 0 or cycle_count < cycles:
            cycle_count += 1
            console.print(f"\n[bold cyan]{'━' * 50}[/bold cyan]")
            console.print(f"[bold cyan]Cycle {cycle_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]")
            console.print(f"[bold cyan]{'━' * 50}[/bold cyan]")

            # Rescan if needed
            if rescan_every > 0 and cycle_count > 1 and (cycle_count - 1) % rescan_every == 0:
                console.print("\n[yellow]📡 Time for market rescan...[/yellow]")
                new_symbols = await scan_and_select()
                if new_symbols:
                    active_symbols = new_symbols
                    console.print(f"[green]Updated watchlist: {', '.join(active_symbols)}[/green]")

            # Get account state
            try:
                account = await broker.get_account()
                positions = await broker.get_positions()
                position_map = {p.symbol: p for p in positions}
                console.print(f"\n[bold]Portfolio:[/bold] ${account.equity:,.2f} (Cash: ${account.cash:,.2f})")
                
                if positions:
                    console.print("[dim]Positions:[/dim]")
                    for p in positions:
                        console.print(f"  {p.symbol}: {p.qty} shares")
            except Exception as e:
                console.print(f"[red]Broker error: {e}[/red]")
                account = None
                position_map = {}

            # Process each symbol
            for symbol in active_symbols:
                console.print(f"\n[bold]📊 Analyzing {symbol}[/bold]")
                
                try:
                    # Get data
                    end = datetime.now()
                    start = end - timedelta(days=30)
                    data = await provider.fetch_ohlcv(symbol, start, end, "1h")
                    
                    if data.empty:
                        console.print(f"  [yellow]No data available[/yellow]")
                        continue
                    
                    features_df = feature_builder.build_features(data)
                    latest_features = features_df.iloc[-1].to_dict()
                    current_price = float(data.iloc[-1]["close"])
                    
                    position = position_map.get(symbol)
                    current_qty = position.qty if position else 0
                    equity = account.equity if account else 100_000
                    
                    state = PipelineState(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        interval="1h",
                        current_price=current_price,
                        last_n_bars=data.tail(20).to_dict("records"),
                        features=latest_features,
                        current_position=current_qty,
                        current_position_pct=(current_qty * current_price / equity) if equity > 0 else 0,
                        cash=account.cash if account else 100_000,
                        equity=equity,
                        daily_pnl=daily_pnl,
                        trades_today=total_trades,
                    )
                    
                    # Run agent pipeline
                    state = await orchestrator.run_pipeline(state)
                    
                    # Log agent decisions to tracing
                    if state.executive_decision:
                        tracing.log_agent_decision(
                            agent_name="ExecutiveAgent",
                            symbol=symbol,
                            decision={
                                "action": state.executive_decision.action.value,
                                "confidence": state.executive_decision.confidence,
                                "size_pct": state.executive_decision.size_pct,
                                "rationale": state.executive_decision.rationale[:200],
                            },
                            inputs={
                                "price": current_price,
                                "technical": state.technical_signal.direction.value if state.technical_signal else None,
                                "fundamental": state.fundamental_signal.direction.value if state.fundamental_signal else None,
                            },
                        )
                    
                    # Display results
                    if verbose:
                        if state.technical_signal:
                            t = state.technical_signal
                            console.print(f"  [dim]Technical:[/dim] {t.direction.value} | Forecast: {t.forecast_q50:+.1%}")
                        if state.fundamental_signal:
                            f = state.fundamental_signal
                            console.print(f"  [dim]Fundamental:[/dim] {f.direction.value} | Sentiment: {f.sentiment_score:.2f}")
                        if state.hybrid_signal:
                            h = state.hybrid_signal
                            console.print(f"  [dim]Hybrid:[/dim] {h.net_direction.value} | Conviction: {h.conviction:.2f}")
                    
                    if state.executive_decision:
                        d = state.executive_decision
                        action_color = "green" if d.action == Action.BUY else "red" if d.action == Action.SELL else "white"
                        console.print(f"  [bold {action_color}]Decision: {d.action.value}[/bold {action_color}] | Confidence: {d.confidence:.0%}")
                        
                        if verbose and d.action != Action.HOLD:
                            console.print(f"  [dim]Rationale: {d.rationale[:150]}...[/dim]")
                    
                    # Execute if approved
                    if (state.final_action and 
                        state.final_action != Action.HOLD and 
                        state.risk_guardian_output and 
                        state.risk_guardian_output.approved):
                        
                        decision = state.executive_decision
                        
                        if isinstance(broker, SimulatedBroker):
                            broker.set_price(symbol, current_price)
                        
                        if decision.action == Action.BUY:
                            qty = int((equity * decision.size_pct) / current_price)
                            if qty > 0:
                                order = await broker.submit_order(symbol, qty, OrderSide.BUY, BrokerOrderType.MARKET)
                                console.print(f"  [green bold]✓ BUY {qty} shares @ ${current_price:.2f}[/green bold]")
                                total_trades += 1
                                # Log trade to tracing
                                tracing.log_trade(
                                    symbol=symbol,
                                    action="BUY",
                                    quantity=qty,
                                    price=current_price,
                                    confidence=decision.confidence,
                                    rationale=decision.rationale,
                                )
                        
                        elif decision.action == Action.SELL and current_qty > 0:
                            qty = int(current_qty * decision.size_pct)
                            if qty <= 0:
                                qty = int(current_qty)
                            if qty > 0:
                                order = await broker.submit_order(symbol, qty, OrderSide.SELL, BrokerOrderType.MARKET)
                                console.print(f"  [red bold]✓ SELL {qty} shares @ ${current_price:.2f}[/red bold]")
                                total_trades += 1
                                # Log trade to tracing
                                tracing.log_trade(
                                    symbol=symbol,
                                    action="SELL",
                                    quantity=qty,
                                    price=current_price,
                                    confidence=decision.confidence,
                                    rationale=decision.rationale,
                                )
                    
                    elif state.risk_guardian_output and not state.risk_guardian_output.approved:
                        if verbose:
                            console.print(f"  [yellow]Blocked by Risk Guardian: {state.risk_guardian_output.veto_reasons}[/yellow]")
                                
                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")
                    if verbose:
                        import traceback
                        console.print(f"  [dim]{traceback.format_exc()}[/dim]")

            # Cycle summary
            console.print(f"\n[dim]Cycle {cycle_count} complete. Total trades this session: {total_trades}[/dim]")

            # Wait for next cycle
            if cycles == 0 or cycle_count < cycles:
                console.print(f"[dim]Next cycle in {interval} minute(s)... Press Ctrl+C to stop[/dim]")
                await asyncio.sleep(interval * 60)

        # Final summary
        console.print(f"\n[bold green]{'━' * 50}[/bold green]")
        console.print(f"[bold green]Session Complete[/bold green]")
        console.print(f"[bold green]{'━' * 50}[/bold green]")
        console.print(f"Total Cycles: {cycle_count}")
        console.print(f"Total Trades: {total_trades}")
        
        final_equity = 100_000
        try:
            final_account = await broker.get_account()
            final_equity = final_account.equity
            console.print(f"Final Equity: ${final_equity:,.2f}")
        except:
            pass
        
        # Log final performance metrics
        tracing.log_performance({
            "total_cycles": cycle_count,
            "total_trades": total_trades,
            "final_equity": final_equity,
            "return_pct": (final_equity - 100_000) / 100_000 * 100,
        })
        
        # End tracing session
        tracing.end_session("completed")
        console.print(f"\n[dim]Traces saved to: ./logs/traces/ and MLflow at {mlflow_uri}[/dim]")

    try:
        console.print("\n[green]🚀 Starting autonomous trading...[/green]")
        asyncio.run(_run_autonomous())
    except KeyboardInterrupt:
        console.print("\n[yellow]Trading stopped by user.[/yellow]")
        tracing.end_session("interrupted")


@app.command()
def run_crew(
    symbols: str = typer.Option(
        "AAPL,NVDA,GOOGL",
        "--symbols", "-s",
        help="Stock symbols to analyze (comma-separated)",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Enable verbose agent output",
    ),
):
    """Run CrewAI trading crew with full AMP tracing.
    
    This uses CrewAI's native Crew, Agent, and Task classes to enable
    full tracing on the CrewAI AMP platform.
    
    Prerequisites:
        1. Run 'crewai login' to authenticate with CrewAI AMP
        2. Set OPENAI_API_KEY in your .env file
    
    View traces at: https://app.crewai.com
    
    Examples:
        python -m src.cli run-crew -s AAPL,NVDA
        python -m src.cli run-crew -s TSLA,AMD,META --verbose
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    console.print(Panel.fit(
        "[bold blue]AI Trading System - CrewAI Crew[/bold blue]\n"
        "[dim]With Full AMP Tracing[/dim]",
        border_style="blue",
    ))
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    console.print(f"Symbols: [bold]{', '.join(symbol_list)}[/bold]")
    console.print(f"Verbose: {'Yes' if verbose else 'No'}")
    console.print("\n[cyan]Traces will be available at: https://app.crewai.com[/cyan]")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY required[/red]")
        return
    
    try:
        from src.orchestrator.crewai_crew import TradingCrew
        
        console.print("\n[green]Starting CrewAI Trading Crew...[/green]")
        console.print("[dim]This may take 2-5 minutes depending on the number of symbols[/dim]\n")
        
        crew = TradingCrew(symbols=symbol_list, verbose=verbose)
        result = crew.kickoff()
        
        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]TRADING RECOMMENDATIONS[/bold green]")
        console.print("="*60)
        console.print(result)
        console.print("="*60)
        
        console.print("\n[cyan]✓ Traces available at: https://app.crewai.com[/cyan]")
        
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Make sure crewai is installed: pip install 'crewai[tools]'[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        if verbose:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def info():
    """Display system information."""
    from src import __version__

    console.print(Panel.fit(
        f"[bold]AI Trading System v{__version__}[/bold]\n\n"
        "Multi-agent AI trading system with reinforcement learning\n\n"
        "[dim]Agents:[/dim] Technical, Fundamental, Hybrid, Researcher, Executive, Risk Guardian\n"
        "[dim]Data:[/dim] YFinance, Alpaca\n"
        "[dim]Storage:[/dim] TimescaleDB, PostgreSQL\n"
        "[dim]ML:[/dim] PyTorch, XGBoost, LightGBM\n"
        "[dim]RL:[/dim] stable-baselines3, d3rlpy\n\n"
        "[dim]Tracing:[/dim] CrewAI AMP (app.crewai.com), MLflow, Local logs",
        title="System Info",
        border_style="cyan",
    ))


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
