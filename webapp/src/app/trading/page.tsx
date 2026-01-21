"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { NumberTicker } from "@/components/ui/number-ticker";
import { api, Position, Account } from "@/lib/api";
import {
  Play,
  Pause,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  RefreshCw,
  Loader2,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function TradingPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [closingAll, setClosingAll] = useState(false);

  const fetchData = async () => {
    try {
      const [accountData, positionsData] = await Promise.all([
        api.getAccount(),
        api.getPositions(),
      ]);
      setAccount(accountData);
      setPositions(positionsData.positions);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyzePositions = async () => {
    setAnalyzing(true);
    try {
      await api.analyzePositions();
      await fetchData();
    } catch (e) {
      console.error(e);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleCloseAll = async () => {
    if (!confirm("Are you sure you want to close ALL positions?")) return;
    
    setClosingAll(true);
    try {
      await api.closeAllPositions();
      await fetchData();
    } catch (e) {
      console.error(e);
    } finally {
      setClosingAll(false);
    }
  };

  const totalPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
  const winningPositions = positions.filter((p) => p.unrealized_pnl > 0).length;
  const winRate = positions.length > 0 ? (winningPositions / positions.length) * 100 : 0;

  return (
    <div className="min-h-screen">
      <Header
        title="Paper Trading"
        subtitle="Test strategies with simulated money"
      />

      <div className="p-6">
        {/* Trading Control Panel */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">Position Manager</h2>
              <p className="text-sm text-navy-400">
                Monitor and manage your paper trading positions
              </p>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={fetchData}
                className="flex items-center gap-2 rounded-lg border border-navy-700 bg-navy-800 px-4 py-2.5 text-sm text-navy-300 hover:border-navy-600"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </button>
              
              <ShimmerButton
                onClick={handleAnalyzePositions}
                disabled={analyzing || positions.length === 0}
                className="h-11 gap-2"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Analyze Positions
                  </>
                )}
              </ShimmerButton>

              {positions.length > 0 && (
                <button
                  onClick={handleCloseAll}
                  disabled={closingAll}
                  className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm text-red-500 hover:bg-red-500/20"
                >
                  {closingAll ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <XCircle className="h-4 w-4" />
                  )}
                  Close All
                </button>
              )}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="mt-6 grid grid-cols-4 gap-4">
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Portfolio Value</p>
              <p className="mt-1 text-2xl font-bold text-white">
                {loading ? (
                  <Loader2 className="h-6 w-6 animate-spin text-orange-500" />
                ) : (
                  <NumberTicker value={account?.portfolio_value || 0} prefix="$" decimalPlaces={2} />
                )}
              </p>
            </div>
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Unrealized P&L</p>
              <p className={cn(
                "mt-1 text-2xl font-bold",
                totalPnl >= 0 ? "text-green-500" : "text-red-500"
              )}>
                {loading ? (
                  <Loader2 className="h-6 w-6 animate-spin text-orange-500" />
                ) : (
                  <NumberTicker value={totalPnl} prefix="$" decimalPlaces={2} />
                )}
              </p>
            </div>
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Positions</p>
              <p className="mt-1 text-2xl font-bold text-white">
                <NumberTicker value={positions.length} />
              </p>
            </div>
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Win Rate</p>
              <p className={cn(
                "mt-1 text-2xl font-bold",
                winRate >= 50 ? "text-green-500" : "text-orange-500"
              )}>
                <NumberTicker value={winRate} suffix="%" decimalPlaces={0} />
              </p>
            </div>
          </div>
        </div>

        {/* Positions Table */}
        <div className="mt-6 rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">
            Open Positions
          </h2>

          {positions.length === 0 ? (
            <p className="text-navy-400 py-12 text-center">
              No open positions. Go to Scanner to find stocks and buy.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-navy-800 text-left text-sm text-navy-400">
                    <th className="pb-3 font-medium">Symbol</th>
                    <th className="pb-3 font-medium">Quantity</th>
                    <th className="pb-3 font-medium">Entry Price</th>
                    <th className="pb-3 font-medium">Current Price</th>
                    <th className="pb-3 font-medium">Market Value</th>
                    <th className="pb-3 font-medium text-right">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos) => (
                    <tr
                      key={pos.symbol}
                      className="border-b border-navy-800/50 text-sm"
                    >
                      <td className="py-4 font-medium text-white">
                        {pos.symbol}
                      </td>
                      <td className="py-4 text-navy-300">{pos.quantity}</td>
                      <td className="py-4 text-navy-300">
                        ${pos.avg_entry_price.toFixed(2)}
                      </td>
                      <td className="py-4 text-navy-300">
                        ${pos.current_price.toFixed(2)}
                      </td>
                      <td className="py-4 text-navy-300">
                        ${pos.market_value.toFixed(2)}
                      </td>
                      <td
                        className={cn(
                          "py-4 text-right font-medium",
                          pos.unrealized_pnl >= 0
                            ? "text-green-500"
                            : "text-red-500"
                        )}
                      >
                        <span className="flex items-center justify-end gap-1">
                          {pos.unrealized_pnl >= 0 ? (
                            <TrendingUp className="h-4 w-4" />
                          ) : (
                            <TrendingDown className="h-4 w-4" />
                          )}
                          ${Math.abs(pos.unrealized_pnl).toFixed(2)}
                          <span className="text-xs">
                            ({pos.unrealized_pnl_pct.toFixed(1)}%)
                          </span>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Paper Trading Notice */}
        <div className="mt-6 flex items-center gap-3 rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-4">
          <AlertCircle className="h-5 w-5 text-yellow-500" />
          <p className="text-sm text-yellow-500">
            Paper Trading Mode: All trades are simulated with Alpaca paper trading. No real money is at risk.
          </p>
        </div>
      </div>
    </div>
  );
}
