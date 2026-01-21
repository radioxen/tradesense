"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/header";
import { api, Position } from "@/lib/api";
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
  ComposedChart,
  Bar,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Loader2,
  ChevronDown,
  ChevronUp,
  DollarSign,
  BarChart3,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface StockChart {
  symbol: string;
  data: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
  current_price: number;
  change: number;
  change_pct: number;
  high_30d: number;
  low_30d: number;
}

export default function Holdings() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [stockChart, setStockChart] = useState<StockChart | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartDays, setChartDays] = useState(30);

  const fetchPositions = async () => {
    try {
      setLoading(true);
      const data = await api.getPositions();
      setPositions(data.positions);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const fetchStockChart = async (symbol: string) => {
    try {
      setChartLoading(true);
      const chart = await api.getStockChart(symbol, chartDays);
      setStockChart(chart);
    } catch (e) {
      console.error(e);
    } finally {
      setChartLoading(false);
    }
  };

  useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedStock) {
      fetchStockChart(selectedStock);
    }
  }, [selectedStock, chartDays]);

  const totalValue = positions.reduce((sum, p) => sum + p.market_value, 0);
  const totalPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
  const totalCost = positions.reduce((sum, p) => sum + (p.avg_entry_price * p.quantity), 0);

  return (
    <div className="min-h-screen">
      <Header title="Holdings" subtitle="Your stock positions and charts" />

      <div className="p-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4 mb-6">
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-4">
            <p className="text-sm text-navy-400">Total Positions</p>
            <p className="mt-1 text-2xl font-bold text-white">{positions.length}</p>
          </div>
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-4">
            <p className="text-sm text-navy-400">Total Cost Basis</p>
            <p className="mt-1 text-2xl font-bold text-white">${totalCost.toFixed(2)}</p>
          </div>
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-4">
            <p className="text-sm text-navy-400">Current Value</p>
            <p className="mt-1 text-2xl font-bold text-white">${totalValue.toFixed(2)}</p>
          </div>
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-4">
            <p className="text-sm text-navy-400">Total P&L</p>
            <p className={cn(
              "mt-1 text-2xl font-bold",
              totalPnl >= 0 ? "text-green-500" : "text-red-500"
            )}>
              {totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}
            </p>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
          </div>
        ) : positions.length === 0 ? (
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-12 text-center">
            <BarChart3 className="h-16 w-16 mx-auto mb-4 text-navy-600" />
            <h3 className="text-xl font-semibold text-white mb-2">No Holdings</h3>
            <p className="text-navy-400">
              Start a trading workflow to acquire positions.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Positions List */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-white">Your Positions</h2>
              {positions.map((pos) => (
                <div
                  key={pos.symbol}
                  onClick={() => setSelectedStock(pos.symbol)}
                  className={cn(
                    "rounded-xl border p-4 cursor-pointer transition-all",
                    selectedStock === pos.symbol
                      ? "border-orange-500 bg-orange-500/5"
                      : "border-navy-800 bg-navy-900/50 hover:border-navy-700"
                  )}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "flex h-10 w-10 items-center justify-center rounded-lg text-lg font-bold",
                        pos.unrealized_pnl >= 0 ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"
                      )}>
                        {pos.symbol.slice(0, 2)}
                      </div>
                      <div>
                        <p className="font-bold text-white">{pos.symbol}</p>
                        <p className="text-sm text-navy-400">{pos.quantity} shares</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-white">${pos.market_value.toFixed(2)}</p>
                      <p className={cn(
                        "text-sm flex items-center gap-1 justify-end",
                        pos.unrealized_pnl >= 0 ? "text-green-500" : "text-red-500"
                      )}>
                        {pos.unrealized_pnl >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                        {pos.unrealized_pnl >= 0 ? "+" : ""}${pos.unrealized_pnl.toFixed(2)} ({pos.unrealized_pnl_pct.toFixed(1)}%)
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-navy-500">Entry</p>
                      <p className="text-white">${pos.avg_entry_price.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-navy-500">Current</p>
                      <p className="text-white">${pos.current_price.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-navy-500">Cost Basis</p>
                      <p className="text-white">${(pos.avg_entry_price * pos.quantity).toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Stock Chart */}
            <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
              {selectedStock ? (
                <>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <h2 className="text-lg font-semibold text-white">{selectedStock}</h2>
                      <button
                        onClick={() => setSelectedStock(null)}
                        className="p-1 rounded hover:bg-navy-800"
                      >
                        <X className="h-4 w-4 text-navy-400" />
                      </button>
                    </div>
                    <div className="flex gap-2">
                      {[7, 14, 30, 60].map((days) => (
                        <button
                          key={days}
                          onClick={() => setChartDays(days)}
                          className={cn(
                            "px-3 py-1 rounded text-sm",
                            chartDays === days
                              ? "bg-orange-500 text-white"
                              : "bg-navy-800 text-navy-300 hover:bg-navy-700"
                          )}
                        >
                          {days}D
                        </button>
                      ))}
                    </div>
                  </div>

                  {chartLoading ? (
                    <div className="flex items-center justify-center h-64">
                      <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
                    </div>
                  ) : stockChart ? (
                    <>
                      {/* Stats */}
                      <div className="grid grid-cols-4 gap-4 mb-4">
                        <div>
                          <p className="text-xs text-navy-500">Price</p>
                          <p className="text-lg font-bold text-white">
                            ${stockChart.current_price.toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-navy-500">Change</p>
                          <p className={cn(
                            "text-lg font-bold",
                            stockChart.change >= 0 ? "text-green-500" : "text-red-500"
                          )}>
                            {stockChart.change >= 0 ? "+" : ""}{stockChart.change_pct.toFixed(2)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-navy-500">{chartDays}D High</p>
                          <p className="text-white">${stockChart.high_30d.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-navy-500">{chartDays}D Low</p>
                          <p className="text-white">${stockChart.low_30d.toFixed(2)}</p>
                        </div>
                      </div>

                      {/* Price Chart */}
                      <div className="h-64 mb-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={stockChart.data}>
                            <defs>
                              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={stockChart.change >= 0 ? "#22c55e" : "#ef4444"} stopOpacity={0.3}/>
                                <stop offset="95%" stopColor={stockChart.change >= 0 ? "#22c55e" : "#ef4444"} stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f315c" />
                            <XAxis 
                              dataKey="date" 
                              stroke="#7280a1" 
                              fontSize={10}
                              tickFormatter={(v) => v.slice(5)}
                            />
                            <YAxis 
                              stroke="#7280a1" 
                              fontSize={10}
                              domain={['auto', 'auto']}
                              tickFormatter={(v) => `$${v}`}
                            />
                            <Tooltip
                              contentStyle={{ backgroundColor: "#0f1f49", border: "1px solid #273a66" }}
                              labelStyle={{ color: "#fff" }}
                              formatter={(value: number) => [`$${value.toFixed(2)}`, "Price"]}
                            />
                            <Area
                              type="monotone"
                              dataKey="close"
                              stroke={stockChart.change >= 0 ? "#22c55e" : "#ef4444"}
                              fillOpacity={1}
                              fill="url(#colorPrice)"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Volume Chart */}
                      <div className="h-24">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={stockChart.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f315c" />
                            <XAxis 
                              dataKey="date" 
                              stroke="#7280a1" 
                              fontSize={10}
                              tickFormatter={(v) => v.slice(5)}
                            />
                            <YAxis 
                              stroke="#7280a1" 
                              fontSize={10}
                              tickFormatter={(v) => `${(v / 1000000).toFixed(0)}M`}
                            />
                            <Tooltip
                              contentStyle={{ backgroundColor: "#0f1f49", border: "1px solid #273a66" }}
                              labelStyle={{ color: "#fff" }}
                              formatter={(value: number) => [`${(value / 1000000).toFixed(2)}M`, "Volume"]}
                            />
                            <Bar dataKey="volume" fill="#f97316" opacity={0.5} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-64 text-navy-400">
                      Failed to load chart
                    </div>
                  )}
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-80 text-navy-400">
                  <BarChart3 className="h-12 w-12 mb-3 opacity-50" />
                  <p>Select a position to view chart</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
