"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { api, Account, Position, Trade } from "@/lib/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Wallet,
  PlusCircle,
  MinusCircle,
  Play,
  RefreshCw,
  Loader2,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

export default function Dashboard() {
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [budget, setBudget] = useState<number>(0);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [showBudgetModal, setShowBudgetModal] = useState(false);
  const [budgetAction, setBudgetAction] = useState<"deposit" | "withdraw">("deposit");
  const [budgetAmount, setBudgetAmount] = useState("");

  // Mock chart data - would come from API in production
  const [portfolioHistory, setPortfolioHistory] = useState<any[]>([]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [accountData, positionsData, budgetData, tradesData, perfData] = await Promise.all([
        api.getAccount().catch(() => null),
        api.getPositions().catch(() => ({ positions: [] })),
        api.getBudget().catch(() => ({ current_budget: 0 })),
        api.getTrades().catch(() => ({ trades: [] })),
        api.getPerformance().catch(() => ({ portfolio_history: [] })),
      ]);
      
      setAccount(accountData);
      setPositions(positionsData.positions);
      setBudget(budgetData.current_budget);
      setTrades(tradesData.trades);
      
      // Generate chart data based on budget, not Alpaca account
      const userBudget = budgetData.current_budget || 10000;
      const deployedValue = positionsData.positions.reduce(
        (sum: number, p: any) => sum + (p.market_value || 0), 0
      );
      const currentValue = userBudget + positionsData.positions.reduce(
        (sum: number, p: any) => sum + (p.unrealized_pnl || 0), 0
      );
      
      if (perfData.portfolio_history.length > 0) {
        setPortfolioHistory(perfData.portfolio_history.map((p: any) => ({
          date: new Date(p.timestamp).toLocaleDateString(),
          value: p.portfolio_value,
        })));
      } else {
        // Show flat line at budget value until we have real data
        const today = new Date();
        setPortfolioHistory([
          { date: "Start", value: userBudget },
          { date: "Now", value: currentValue },
        ]);
      }
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

  const handleBudgetSubmit = async () => {
    const amount = parseFloat(budgetAmount);
    if (isNaN(amount) || amount <= 0) return;
    
    try {
      await api.modifyBudget(amount, budgetAction, `${budgetAction} via dashboard`);
      setShowBudgetModal(false);
      setBudgetAmount("");
      fetchData();
    } catch (e) {
      console.error(e);
    }
  };

  const totalPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
  const deployedCapital = positions.reduce((sum, p) => sum + (p.avg_entry_price * p.quantity), 0);
  const currentPositionsValue = positions.reduce((sum, p) => sum + p.market_value, 0);
  const availableCash = budget - deployedCapital + totalPnl;
  const portfolioValue = availableCash + currentPositionsValue;
  const pnlPercent = budget > 0 ? (totalPnl / budget) * 100 : 0;

  const COLORS = ["#f97316", "#0ea5e9", "#22c55e", "#a855f7", "#ef4444"];

  return (
    <div className="min-h-screen">
      <Header title="Dashboard" subtitle="Portfolio overview and capital management" />

      <div className="p-6">
        {/* Top Stats */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
          {/* Capital Card */}
          <div className="relative overflow-hidden rounded-xl border border-navy-800 bg-gradient-to-br from-navy-900 to-navy-950 p-6">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-navy-400">Trading Capital</p>
                <p className="mt-2 text-3xl font-bold text-white">
                  ${budget.toLocaleString()}
                </p>
                <div className="mt-3 flex gap-2">
                  <button
                    onClick={() => { setBudgetAction("deposit"); setShowBudgetModal(true); }}
                    className="flex items-center gap-1 rounded-lg bg-green-500/10 px-2 py-1 text-xs text-green-500 hover:bg-green-500/20"
                  >
                    <PlusCircle className="h-3 w-3" />
                    Deposit
                  </button>
                  <button
                    onClick={() => { setBudgetAction("withdraw"); setShowBudgetModal(true); }}
                    className="flex items-center gap-1 rounded-lg bg-red-500/10 px-2 py-1 text-xs text-red-500 hover:bg-red-500/20"
                  >
                    <MinusCircle className="h-3 w-3" />
                    Withdraw
                  </button>
                </div>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
                <Wallet className="h-6 w-6 text-orange-500" />
              </div>
            </div>
          </div>

          {/* Available Cash */}
          <div className="relative overflow-hidden rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-navy-400">Available Cash</p>
                <p className="mt-2 text-3xl font-bold text-white">
                  ${Math.max(0, budget - deployedCapital).toLocaleString()}
                </p>
                <p className="mt-2 text-sm text-navy-400">
                  ${deployedCapital.toFixed(0)} deployed
                </p>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
                <DollarSign className="h-6 w-6 text-orange-500" />
              </div>
            </div>
          </div>

          {/* Unrealized P&L */}
          <div className="relative overflow-hidden rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-navy-400">Unrealized P&L</p>
                <p className={cn(
                  "mt-2 text-3xl font-bold",
                  totalPnl >= 0 ? "text-green-500" : "text-red-500"
                )}>
                  {totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}
                </p>
                <p className={cn(
                  "mt-2 text-sm",
                  pnlPercent >= 0 ? "text-green-500" : "text-red-500"
                )}>
                  {pnlPercent >= 0 ? "+" : ""}{pnlPercent.toFixed(2)}% of capital
                </p>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
                {totalPnl >= 0 ? (
                  <TrendingUp className="h-6 w-6 text-green-500" />
                ) : (
                  <TrendingDown className="h-6 w-6 text-red-500" />
                )}
              </div>
            </div>
          </div>

          {/* Positions */}
          <div className="relative overflow-hidden rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-navy-400">Active Positions</p>
                <p className="mt-2 text-3xl font-bold text-white">
                  {positions.length}
                </p>
                <p className="mt-2 text-sm text-navy-400">
                  {trades.length} total trades
                </p>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
                <BarChart3 className="h-6 w-6 text-orange-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mb-6 flex items-center gap-4">
          <Link href="/command">
            <ShimmerButton className="h-12 gap-2 px-6">
              <Play className="h-5 w-5" />
              Start Trading Workflow
            </ShimmerButton>
          </Link>
          <button
            onClick={fetchData}
            className="flex items-center gap-2 rounded-lg border border-navy-700 bg-navy-800 px-4 py-3 text-sm text-navy-300 hover:border-navy-600"
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
            Refresh
          </button>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 mb-6">
          {/* Portfolio Performance Chart */}
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Portfolio Performance</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={portfolioHistory}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f315c" />
                  <XAxis dataKey="date" stroke="#7280a1" fontSize={12} />
                  <YAxis stroke="#7280a1" fontSize={12} tickFormatter={(v) => `$${v.toLocaleString()}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0f1f49", border: "1px solid #273a66" }}
                    labelStyle={{ color: "#fff" }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#f97316"
                    fillOpacity={1}
                    fill="url(#colorValue)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Positions Breakdown */}
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Portfolio Allocation</h2>
            <div className="h-64 flex items-center justify-center">
              {positions.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={positions.map(p => ({ name: p.symbol, value: p.market_value }))}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      fill="#8884d8"
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {positions.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center text-navy-400">
                  <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No active positions</p>
                  <p className="text-sm">Start a workflow to begin trading</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Positions Table */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4">Active Positions</h2>
          {positions.length === 0 ? (
            <p className="text-navy-400 py-8 text-center">
              No open positions. Start a trading workflow to find opportunities.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-navy-800 text-left text-sm text-navy-400">
                    <th className="pb-3 font-medium">Symbol</th>
                    <th className="pb-3 font-medium">Qty</th>
                    <th className="pb-3 font-medium">Entry</th>
                    <th className="pb-3 font-medium">Current</th>
                    <th className="pb-3 font-medium">Value</th>
                    <th className="pb-3 font-medium text-right">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos) => (
                    <tr key={pos.symbol} className="border-b border-navy-800/50 text-sm">
                      <td className="py-4 font-medium text-white">{pos.symbol}</td>
                      <td className="py-4 text-navy-300">{pos.quantity}</td>
                      <td className="py-4 text-navy-300">${pos.avg_entry_price.toFixed(2)}</td>
                      <td className="py-4 text-navy-300">${pos.current_price.toFixed(2)}</td>
                      <td className="py-4 text-navy-300">${pos.market_value.toFixed(2)}</td>
                      <td className={cn(
                        "py-4 text-right font-medium",
                        pos.unrealized_pnl >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        <span className="flex items-center justify-end gap-1">
                          {pos.unrealized_pnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                          ${Math.abs(pos.unrealized_pnl).toFixed(2)} ({pos.unrealized_pnl_pct.toFixed(1)}%)
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Recent Trades */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Recent Trades</h2>
          {trades.length === 0 ? (
            <p className="text-navy-400 py-8 text-center">
              No trades yet. Start a workflow to begin trading.
            </p>
          ) : (
            <div className="space-y-3">
              {trades.slice(0, 10).map((trade) => (
                <div key={trade.id} className="flex items-center justify-between rounded-lg bg-navy-800/30 p-3">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-lg",
                      trade.action === "buy" ? "bg-green-500/10" : "bg-red-500/10"
                    )}>
                      {trade.action === "buy" ? (
                        <TrendingUp className="h-5 w-5 text-green-500" />
                      ) : (
                        <TrendingDown className="h-5 w-5 text-red-500" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium text-white">
                        {trade.action.toUpperCase()} {trade.quantity} {trade.symbol}
                      </p>
                      <p className="text-sm text-navy-400">
                        @ ${trade.price.toFixed(2)} • {new Date(trade.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <p className="text-sm text-navy-300">${trade.total_value.toFixed(2)}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Budget Modal */}
      {showBudgetModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-xl border border-navy-800 bg-navy-900 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              {budgetAction === "deposit" ? "Add Capital" : "Withdraw Capital"}
            </h3>
            <input
              type="number"
              value={budgetAmount}
              onChange={(e) => setBudgetAmount(e.target.value)}
              placeholder="Enter amount"
              className="w-full rounded-lg border border-navy-700 bg-navy-800 px-4 py-3 text-white mb-4"
            />
            <div className="flex gap-3">
              <button
                onClick={() => setShowBudgetModal(false)}
                className="flex-1 rounded-lg border border-navy-700 bg-navy-800 px-4 py-3 text-navy-300"
              >
                Cancel
              </button>
              <ShimmerButton onClick={handleBudgetSubmit} className="flex-1">
                {budgetAction === "deposit" ? "Deposit" : "Withdraw"}
              </ShimmerButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
