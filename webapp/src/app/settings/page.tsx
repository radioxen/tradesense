"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { api } from "@/lib/api";
import {
  Key,
  Shield,
  Bot,
  Save,
  Eye,
  EyeOff,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function SettingsPage() {
  const [showKeys, setShowKeys] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "connected" | "error">("checking");
  
  useEffect(() => {
    api.health()
      .then(() => setApiStatus("connected"))
      .catch(() => setApiStatus("error"));
  }, []);

  return (
    <div className="min-h-screen">
      <Header title="Settings" subtitle="Configure your trading system" />

      <div className="p-6 max-w-4xl">
        {/* API Status */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Bot className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">System Status</h2>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {apiStatus === "checking" ? (
                <Loader2 className="h-5 w-5 animate-spin text-orange-500" />
              ) : apiStatus === "connected" ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <AlertCircle className="h-5 w-5 text-red-500" />
              )}
              <span className={cn(
                "text-sm",
                apiStatus === "connected" ? "text-green-500" : 
                apiStatus === "error" ? "text-red-500" : "text-navy-400"
              )}>
                Backend API: {apiStatus === "checking" ? "Checking..." : 
                             apiStatus === "connected" ? "Connected" : "Not Connected"}
              </span>
            </div>
          </div>
          
          {apiStatus === "error" && (
            <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/10 p-4">
              <p className="text-sm text-red-400">
                Cannot connect to backend API. Make sure the API server is running:
              </p>
              <code className="block mt-2 text-xs text-red-300 bg-red-500/10 p-2 rounded">
                docker compose up -d
              </code>
            </div>
          )}
        </div>

        {/* API Keys Info */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <Key className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">API Keys</h2>
          </div>

          <p className="text-sm text-navy-400 mb-4">
            API keys are configured via environment variables in the .env file.
            Edit the .env file to update your keys, then restart the backend.
          </p>

          <div className="space-y-4">
            <div className="flex items-center justify-between rounded-lg bg-navy-800/50 p-4">
              <div>
                <p className="font-medium text-white">OpenAI API Key</p>
                <p className="text-sm text-navy-400">Required for LLM agents</p>
              </div>
              <span className="text-sm text-green-500">Configured</span>
            </div>
            
            <div className="flex items-center justify-between rounded-lg bg-navy-800/50 p-4">
              <div>
                <p className="font-medium text-white">Perplexity API Key</p>
                <p className="text-sm text-navy-400">For news and catalyst search</p>
              </div>
              <span className="text-sm text-green-500">Configured</span>
            </div>
            
            <div className="flex items-center justify-between rounded-lg bg-navy-800/50 p-4">
              <div>
                <p className="font-medium text-white">Alpaca API Keys</p>
                <p className="text-sm text-navy-400">For paper trading</p>
              </div>
              <span className="text-sm text-green-500">Configured</span>
            </div>
          </div>
        </div>

        {/* Risk Settings Info */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">
              Risk Management
            </h2>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Max Position Size</p>
              <p className="text-2xl font-bold text-white">10%</p>
              <p className="text-xs text-navy-500">of portfolio per position</p>
            </div>
            
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Take Profit</p>
              <p className="text-2xl font-bold text-green-500">+5%</p>
              <p className="text-xs text-navy-500">auto-sell trigger</p>
            </div>
            
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Stop Loss</p>
              <p className="text-2xl font-bold text-red-500">-3%</p>
              <p className="text-xs text-navy-500">auto-sell trigger</p>
            </div>
            
            <div className="rounded-lg bg-navy-800/50 p-4">
              <p className="text-sm text-navy-400">Min Confidence</p>
              <p className="text-2xl font-bold text-orange-500">65%</p>
              <p className="text-xs text-navy-500">to execute trades</p>
            </div>
          </div>
          
          <p className="mt-4 text-sm text-navy-400">
            Risk settings are managed by the scheduler. The system automatically
            analyzes positions hourly during market hours and executes trades
            based on these thresholds.
          </p>
        </div>

        {/* Scheduler Info */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Bot className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-white">
              Automated Trading
            </h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between rounded-lg bg-navy-800/50 p-4">
              <div>
                <p className="font-medium text-white">Hourly Analysis</p>
                <p className="text-sm text-navy-400">
                  Runs every hour during market hours (9:30 AM - 4:00 PM ET)
                </p>
              </div>
              <span className="rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-500">
                Active
              </span>
            </div>
            
            <div className="flex items-center justify-between rounded-lg bg-navy-800/50 p-4">
              <div>
                <p className="font-medium text-white">Auto Take-Profit</p>
                <p className="text-sm text-navy-400">
                  Automatically sells positions at +5% gain
                </p>
              </div>
              <span className="rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-500">
                Enabled
              </span>
            </div>
            
            <div className="flex items-center justify-between rounded-lg bg-navy-800/50 p-4">
              <div>
                <p className="font-medium text-white">Auto Stop-Loss</p>
                <p className="text-sm text-navy-400">
                  Automatically sells positions at -3% loss
                </p>
              </div>
              <span className="rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-500">
                Enabled
              </span>
            </div>
          </div>
        </div>

        {/* Paper Trading Notice */}
        <div className="mt-6 flex items-center gap-3 rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-4">
          <AlertCircle className="h-5 w-5 text-yellow-500" />
          <p className="text-sm text-yellow-500">
            Paper Trading Mode: The system is configured for paper trading only.
            All trades are simulated with Alpaca's paper trading API.
          </p>
        </div>
      </div>
    </div>
  );
}
