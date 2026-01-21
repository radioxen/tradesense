"""Reporting module.

Generates backtest reports and trading summaries.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd

from src.utils.logging import get_logger


logger = get_logger(__name__)


class BacktestReporter:
    """Generates backtest reports."""

    def __init__(self, output_dir: Path):
        """Initialize reporter.

        Args:
            output_dir: Output directory for reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        config: dict[str, Any],
        metrics: dict[str, float],
        trades: list[Any],
        equity_curve: pd.DataFrame,
        decisions: list[dict] | None = None,
    ) -> Path:
        """Generate Markdown report.

        Args:
            config: Backtest configuration.
            metrics: Performance metrics.
            trades: List of trades.
            equity_curve: Equity curve dataframe.
            decisions: Optional decision log.

        Returns:
            Path to generated report.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"backtest_report_{timestamp}.md"

        lines = [
            "# Backtest Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            "```yaml",
        ]

        # Add config
        for key, value in config.items():
            lines.append(f"{key}: {value}")
        lines.extend(["```", ""])

        # Performance Summary
        lines.extend([
            "## Performance Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        metric_labels = {
            "total_return": "Total Return",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
            "total_trades": "Total Trades",
            "win_rate": "Win Rate",
            "total_commission": "Total Commissions",
            "final_equity": "Final Equity",
        }

        for key, label in metric_labels.items():
            value = metrics.get(key, 0)
            if "pct" in key or "return" in key or "rate" in key or "drawdown" in key:
                formatted = f"{value:.2%}"
            elif "equity" in key or "commission" in key:
                formatted = f"${value:,.2f}"
            elif isinstance(value, float):
                formatted = f"{value:.2f}"
            else:
                formatted = str(value)
            lines.append(f"| {label} | {formatted} |")

        lines.append("")

        # Trade Summary
        if trades:
            lines.extend([
                "## Trade Summary",
                "",
                f"Total Trades: {len(trades)}",
                "",
                "### Recent Trades",
                "",
                "| Time | Symbol | Side | Qty | Price | Commission |",
                "|------|--------|------|-----|-------|------------|",
            ])

            for trade in trades[-20:]:
                timestamp = getattr(trade, 'timestamp', '-')
                symbol = getattr(trade, 'symbol', '-')
                side = getattr(trade, 'side', '-')
                qty = getattr(trade, 'quantity', 0)
                price = getattr(trade, 'price', 0)
                comm = getattr(trade, 'commission', 0)
                lines.append(f"| {timestamp} | {symbol} | {side} | {qty:.0f} | ${price:.2f} | ${comm:.2f} |")

            lines.append("")

        # Decision Analysis
        if decisions:
            lines.extend([
                "## Decision Analysis",
                "",
                f"Total Decisions: {len(decisions)}",
                "",
            ])

            # Count actions
            action_counts = {}
            for d in decisions:
                action = d.get("action", "UNKNOWN")
                action_counts[action] = action_counts.get(action, 0) + 1

            lines.extend([
                "### Action Distribution",
                "",
                "| Action | Count | Percentage |",
                "|--------|-------|------------|",
            ])

            total = sum(action_counts.values())
            for action, count in action_counts.items():
                pct = count / total if total > 0 else 0
                lines.append(f"| {action} | {count} | {pct:.1%} |")

            lines.append("")

        # Save report
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated report: {report_path}")
        return report_path

    def generate_equity_plot(
        self,
        equity_curve: pd.DataFrame,
        filename: str | None = None,
    ) -> Path | None:
        """Generate equity curve plot.

        Args:
            equity_curve: Equity curve dataframe.
            filename: Optional filename.

        Returns:
            Path to plot or None if failed.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

            # Equity curve
            ax1 = axes[0]
            ax1.plot(equity_curve["timestamp"], equity_curve["equity"], label="Equity", linewidth=1.5)
            ax1.fill_between(
                equity_curve["timestamp"],
                equity_curve["equity"],
                alpha=0.3,
            )
            ax1.set_ylabel("Equity ($)")
            ax1.set_title("Backtest Equity Curve")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Drawdown
            ax2 = axes[1]
            ax2.fill_between(
                equity_curve["timestamp"],
                equity_curve["drawdown"] * 100,
                color="red",
                alpha=0.5,
            )
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Date")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save
            filename = filename or f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = self.output_dir / filename
            plt.savefig(plot_path, dpi=150)
            plt.close()

            logger.info(f"Generated plot: {plot_path}")
            return plot_path

        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None


class DailyReporter:
    """Generates daily trading reports."""

    def __init__(
        self,
        output_dir: Path,
        send_email: bool = False,
        email_to: str | None = None,
    ):
        """Initialize daily reporter.

        Args:
            output_dir: Output directory.
            send_email: Whether to send email.
            email_to: Email recipient.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.send_email = send_email
        self.email_to = email_to

    def generate_daily_summary(
        self,
        date: datetime,
        equity: float,
        daily_pnl: float,
        positions: list[dict],
        trades: list[dict],
        decisions_made: int,
        decisions_executed: int,
    ) -> str:
        """Generate daily summary.

        Args:
            date: Report date.
            equity: End of day equity.
            daily_pnl: Day's P&L.
            positions: Open positions.
            trades: Day's trades.
            decisions_made: Total decisions.
            decisions_executed: Executed decisions.

        Returns:
            Report content.
        """
        lines = [
            f"# Daily Trading Report - {date.strftime('%Y-%m-%d')}",
            "",
            "## Summary",
            "",
            f"- **Equity:** ${equity:,.2f}",
            f"- **Daily P&L:** ${daily_pnl:,.2f} ({daily_pnl/equity*100:.2f}%)" if equity > 0 else f"- **Daily P&L:** ${daily_pnl:,.2f}",
            f"- **Trades Executed:** {len(trades)}",
            f"- **Decisions Made:** {decisions_made}",
            f"- **Decisions Executed:** {decisions_executed}",
            "",
        ]

        # Positions
        if positions:
            lines.extend([
                "## Open Positions",
                "",
                "| Symbol | Qty | Avg Cost | Current | Unrealized P&L |",
                "|--------|-----|----------|---------|----------------|",
            ])

            for pos in positions:
                symbol = pos.get("symbol", "-")
                qty = pos.get("qty", 0)
                avg = pos.get("avg_entry_price", 0)
                current = pos.get("current_price", 0)
                upl = pos.get("unrealized_pl", 0)
                lines.append(f"| {symbol} | {qty:.0f} | ${avg:.2f} | ${current:.2f} | ${upl:.2f} |")

            lines.append("")

        # Trades
        if trades:
            lines.extend([
                "## Today's Trades",
                "",
                "| Time | Symbol | Side | Qty | Price |",
                "|------|--------|------|-----|-------|",
            ])

            for trade in trades:
                time = trade.get("time", "-")
                symbol = trade.get("symbol", "-")
                side = trade.get("side", "-")
                qty = trade.get("qty", 0)
                price = trade.get("price", 0)
                lines.append(f"| {time} | {symbol} | {side} | {qty:.0f} | ${price:.2f} |")

            lines.append("")

        content = "\n".join(lines)

        # Save report
        report_path = self.output_dir / f"daily_report_{date.strftime('%Y%m%d')}.md"
        with open(report_path, "w") as f:
            f.write(content)

        logger.info(f"Generated daily report: {report_path}")

        return content
