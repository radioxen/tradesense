"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { api, Activity } from "@/lib/api";
import {
  Activity as ActivityIcon,
  TrendingUp,
  TrendingDown,
  Bot,
  Shield,
  Clock,
  RefreshCw,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function ActivityPage() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");

  const fetchData = async () => {
    try {
      const data = await api.getActivity(100);
      setActivities(data.activities);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const filteredActivities = activities.filter((a) => {
    if (filter === "all") return true;
    if (filter === "trades") return a.type === "trade";
    if (filter === "decisions") return a.type === "decision" || a.type === "analysis";
    if (filter === "scans") return a.type === "scan";
    return true;
  });

  return (
    <div className="min-h-screen">
      <Header
        title="Activity Log"
        subtitle="Complete history of all agent decisions and trades"
      />

      <div className="p-6">
        {/* Controls */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="rounded-lg border border-navy-700 bg-navy-800 px-4 py-2.5 text-sm text-navy-300"
            >
              <option value="all">All Activity</option>
              <option value="trades">Trades Only</option>
              <option value="decisions">Decisions Only</option>
              <option value="scans">Scans Only</option>
            </select>
          </div>

          <button
            onClick={fetchData}
            className="flex items-center gap-2 rounded-lg border border-navy-700 bg-navy-800 px-4 py-2.5 text-sm text-navy-300 hover:border-navy-600"
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
            Refresh
          </button>
        </div>

        {/* Activity Timeline */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <div className="flex items-center gap-3 mb-6">
            <ActivityIcon className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">
              Real-time Activity Stream
            </h2>
            <div className="flex items-center gap-2 ml-auto">
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs text-navy-400">Live</span>
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
            </div>
          ) : filteredActivities.length === 0 ? (
            <p className="text-navy-400 py-12 text-center">
              No activity yet. Run a scan or trade to see activity.
            </p>
          ) : (
            <div className="relative">
              {/* Timeline line */}
              <div className="absolute left-[23px] top-0 bottom-0 w-px bg-navy-800" />

              <div className="space-y-4">
                {filteredActivities.map((item, idx) => (
                  <div key={idx} className="relative flex gap-4 pl-12">
                    {/* Timeline dot */}
                    <div
                      className={cn(
                        "absolute left-0 flex h-12 w-12 items-center justify-center rounded-full border-4 border-navy-950",
                        item.type === "trade"
                          ? "bg-green-500/20"
                          : item.type === "decision"
                          ? "bg-orange-500/20"
                          : "bg-navy-800"
                      )}
                    >
                      {item.type === "trade" ? (
                        item.action === "BUY" ? (
                          <TrendingUp className="h-5 w-5 text-green-500" />
                        ) : (
                          <TrendingDown className="h-5 w-5 text-red-500" />
                        )
                      ) : (
                        <Bot className="h-5 w-5 text-orange-500" />
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 rounded-lg border border-navy-800 bg-navy-800/30 p-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-white">
                              {item.agent || item.type}
                            </span>
                            {item.symbol && item.symbol !== "-" && (
                              <>
                                <span className="text-navy-500">→</span>
                                <span className="font-mono text-orange-500">
                                  {item.symbol}
                                </span>
                              </>
                            )}
                            <span
                              className={cn(
                                "rounded-full px-2 py-0.5 text-xs font-medium",
                                item.action === "BUY" ||
                                  item.action === "BULLISH" ||
                                  item.action === "COMPLETED"
                                  ? "bg-green-500/10 text-green-500"
                                  : item.action === "SELL" ||
                                    item.action === "BEARISH" ||
                                    item.action === "FAILED"
                                  ? "bg-red-500/10 text-red-500"
                                  : "bg-navy-600/10 text-navy-400"
                              )}
                            >
                              {item.action}
                            </span>
                          </div>
                          <p className="mt-1 text-sm text-navy-400">
                            {item.details}
                          </p>
                        </div>
                        <div className="flex items-center gap-1 text-xs text-navy-500">
                          <Clock className="h-3 w-3" />
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
