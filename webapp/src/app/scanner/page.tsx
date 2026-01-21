"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { NumberTicker } from "@/components/ui/number-ticker";
import { api, ScanResult, AnalysisResult } from "@/lib/api";
import {
  Search,
  TrendingUp,
  TrendingDown,
  Zap,
  ArrowUpRight,
  Loader2,
  CheckCircle,
  PlayCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function ScannerPage() {
  const [isScanning, setIsScanning] = useState(false);
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [executing, setExecuting] = useState(false);

  useEffect(() => {
    // Load existing scan results on mount
    api.getScanResults().then((data) => {
      setScanResults(data.results);
    }).catch(console.error);
  }, []);

  const handleScan = async () => {
    setIsScanning(true);
    setSelectedSymbols(new Set());
    setAnalysisResults([]);
    
    try {
      const result = await api.runScan(20, true);
      setScanResults(result.results);
    } catch (e) {
      console.error("Scan failed:", e);
    } finally {
      setIsScanning(false);
    }
  };

  const toggleSymbol = (symbol: string) => {
    const newSelected = new Set(selectedSymbols);
    if (newSelected.has(symbol)) {
      newSelected.delete(symbol);
    } else {
      newSelected.add(symbol);
    }
    setSelectedSymbols(newSelected);
  };

  const handleAnalyze = async () => {
    if (selectedSymbols.size === 0) return;
    
    setAnalyzing(true);
    try {
      const result = await api.analyzeStocks(Array.from(selectedSymbols));
      setAnalysisResults(result.results);
    } catch (e) {
      console.error("Analysis failed:", e);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleExecuteTrades = async () => {
    const buyDecisions = analysisResults.filter(
      (r) => r.decision?.action === "buy" && r.decision.quantity > 0
    );
    
    if (buyDecisions.length === 0) return;
    
    setExecuting(true);
    try {
      for (const decision of buyDecisions) {
        if (decision.decision) {
          await api.executeTrade(
            decision.symbol,
            "buy",
            decision.decision.quantity
          );
        }
      }
      // Clear selections after execution
      setSelectedSymbols(new Set());
      setAnalysisResults([]);
    } catch (e) {
      console.error("Execution failed:", e);
    } finally {
      setExecuting(false);
    }
  };

  const buyRecommendations = analysisResults.filter(
    (r) => r.decision?.action === "buy"
  );

  return (
    <div className="min-h-screen">
      <Header
        title="Market Scanner"
        subtitle="Discover momentum stocks with AI-powered screening"
      />

      <div className="p-6">
        {/* Scanner Controls */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">
                Momentum Scanner
              </h2>
              <p className="text-sm text-navy-400">
                Find stocks likely to move 5-20% in the next 1-5 days
              </p>
            </div>

            <div className="flex items-center gap-3">
              <ShimmerButton
                onClick={handleScan}
                disabled={isScanning}
                className="h-11 gap-2"
              >
                {isScanning ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Scanning...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Run Scanner
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="mt-6 grid grid-cols-4 gap-4">
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Stocks Found</p>
              <p className="mt-1 text-2xl font-bold text-white">
                <NumberTicker value={scanResults.length} />
              </p>
            </div>
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Selected</p>
              <p className="mt-1 text-2xl font-bold text-orange-500">
                <NumberTicker value={selectedSymbols.size} />
              </p>
            </div>
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Analyzed</p>
              <p className="mt-1 text-2xl font-bold text-green-500">
                <NumberTicker value={analysisResults.length} />
              </p>
            </div>
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Buy Signals</p>
              <p className="mt-1 text-2xl font-bold text-green-500">
                <NumberTicker value={buyRecommendations.length} />
              </p>
            </div>
          </div>
        </div>

        {/* Results Table */}
        <div className="mt-6 rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">
              Momentum Candidates
            </h2>
            <div className="flex items-center gap-3">
              {selectedSymbols.size > 0 && (
                <ShimmerButton
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="h-9 text-sm"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <PlayCircle className="h-4 w-4 mr-2" />
                      Analyze Selected ({selectedSymbols.size})
                    </>
                  )}
                </ShimmerButton>
              )}
              {buyRecommendations.length > 0 && (
                <ShimmerButton
                  onClick={handleExecuteTrades}
                  disabled={executing}
                  className="h-9 text-sm !bg-green-600/20 border-green-500/30"
                >
                  {executing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Executing...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Execute Buys ({buyRecommendations.length})
                    </>
                  )}
                </ShimmerButton>
              )}
            </div>
          </div>

          {scanResults.length === 0 ? (
            <p className="text-navy-400 py-12 text-center">
              No scan results. Click "Run Scanner" to find momentum stocks.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-navy-800 text-left text-sm text-navy-400">
                    <th className="pb-3 font-medium">Select</th>
                    <th className="pb-3 font-medium">Rank</th>
                    <th className="pb-3 font-medium">Symbol</th>
                    <th className="pb-3 font-medium">Price</th>
                    <th className="pb-3 font-medium">Score</th>
                    <th className="pb-3 font-medium">RSI</th>
                    <th className="pb-3 font-medium">Trend</th>
                    <th className="pb-3 font-medium">Catalyst</th>
                    <th className="pb-3 font-medium">Decision</th>
                  </tr>
                </thead>
                <tbody>
                  {scanResults.map((stock, index) => {
                    const analysis = analysisResults.find(
                      (a) => a.symbol === stock.symbol
                    );
                    return (
                      <tr
                        key={stock.symbol}
                        className={cn(
                          "border-b border-navy-800/50 text-sm cursor-pointer transition-colors",
                          selectedSymbols.has(stock.symbol)
                            ? "bg-orange-500/10"
                            : "hover:bg-navy-800/30"
                        )}
                        onClick={() => toggleSymbol(stock.symbol)}
                      >
                        <td className="py-4">
                          <input
                            type="checkbox"
                            checked={selectedSymbols.has(stock.symbol)}
                            onChange={() => toggleSymbol(stock.symbol)}
                            className="rounded border-navy-600"
                          />
                        </td>
                        <td className="py-4">
                          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-orange-500/10 text-xs font-bold text-orange-500">
                            {index + 1}
                          </span>
                        </td>
                        <td className="py-4 font-medium text-white">
                          {stock.symbol}
                        </td>
                        <td className="py-4 text-navy-300">
                          ${stock.price.toFixed(2)}
                        </td>
                        <td className="py-4">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white">
                              {stock.score.toFixed(0)}
                            </span>
                            <div className="h-1.5 w-16 rounded-full bg-navy-800">
                              <div
                                className={cn(
                                  "h-full rounded-full",
                                  stock.score >= 80
                                    ? "bg-green-500"
                                    : stock.score >= 60
                                    ? "bg-orange-500"
                                    : "bg-red-500"
                                )}
                                style={{ width: `${stock.score}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-4">
                          <span
                            className={cn(
                              "font-mono",
                              stock.rsi < 35
                                ? "text-green-500"
                                : stock.rsi > 70
                                ? "text-red-500"
                                : "text-navy-300"
                            )}
                          >
                            {stock.rsi.toFixed(0)}
                          </span>
                        </td>
                        <td className="py-4">
                          <span
                            className={cn(
                              "rounded-full px-2 py-1 text-xs font-medium",
                              stock.trend === "bullish"
                                ? "bg-green-500/10 text-green-500"
                                : stock.trend === "bearish"
                                ? "bg-red-500/10 text-red-500"
                                : "bg-navy-600/10 text-navy-400"
                            )}
                          >
                            {stock.trend}
                          </span>
                        </td>
                        <td className="py-4 max-w-[200px]">
                          {stock.catalysts.length > 0 ? (
                            <p className="text-xs text-navy-400 truncate">
                              <Zap className="inline h-3 w-3 text-orange-500 mr-1" />
                              {stock.catalysts[0]}
                            </p>
                          ) : (
                            <span className="text-xs text-navy-500">-</span>
                          )}
                        </td>
                        <td className="py-4">
                          {analysis?.decision ? (
                            <span
                              className={cn(
                                "rounded-full px-3 py-1 text-xs font-medium",
                                analysis.decision.action === "buy"
                                  ? "bg-green-500/10 text-green-500"
                                  : analysis.decision.action === "sell"
                                  ? "bg-red-500/10 text-red-500"
                                  : "bg-navy-600/10 text-navy-400"
                              )}
                            >
                              {analysis.decision.action.toUpperCase()}
                              {analysis.decision.quantity > 0 &&
                                ` (${analysis.decision.quantity})`}
                            </span>
                          ) : (
                            <span className="text-xs text-navy-500">-</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
