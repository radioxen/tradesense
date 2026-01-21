"use client";

import { useState, useEffect, useRef } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { api, WorkflowStatus, WorkflowDecision } from "@/lib/api";
import {
  Play,
  Search,
  TrendingUp,
  FileText,
  Crown,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  ArrowRight,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  StopCircle,
  Brain,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";

const agentConfig = {
  Executive: { icon: Crown, color: "text-purple-500", bg: "bg-purple-500/10", model: "gpt-5.2" },
  Scanner: { icon: Search, color: "text-blue-500", bg: "bg-blue-500/10", model: "gpt-5-mini" },
  Technical: { icon: TrendingUp, color: "text-green-500", bg: "bg-green-500/10", model: "gpt-5-mini" },
  Fundamental: { icon: FileText, color: "text-orange-500", bg: "bg-orange-500/10", model: "gpt-5-mini" },
  "Risk Manager": { icon: AlertCircle, color: "text-red-500", bg: "bg-red-500/10", model: "gpt-5-mini" },
  System: { icon: AlertTriangle, color: "text-yellow-500", bg: "bg-yellow-500/10", model: "" },
};

export default function CommandCenter() {
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [scanCount, setScanCount] = useState(10);
  const [analyzeTop, setAnalyzeTop] = useState(5);
  const [autoExecute, setAutoExecute] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const stepsEndRef = useRef<HTMLDivElement>(null);

  const pollWorkflow = async () => {
    try {
      const status = await api.getWorkflowStatus();
      // Only set workflow if it has valid data
      if (status && status.id) {
        // Log workflow updates
        if (status.steps?.length !== workflow?.steps?.length) {
          console.log(`[Workflow] New step: ${status.current_step}`);
          console.log(`[Workflow] Steps: ${status.steps?.length || 0}, Decisions: ${status.decisions?.length || 0}`);
        }
        
        setWorkflow(status);
        
        if (status.status === "running" || status.status === "cancelling") {
          setIsRunning(true);
        } else if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
          console.log(`[Workflow] ${status.status.toUpperCase()}`);
          setIsRunning(false);
        }
      }
    } catch (e) {
      console.error("[Workflow] Poll error:", e);
    }
  };

  useEffect(() => {
    pollWorkflow();
    const interval = setInterval(pollWorkflow, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (stepsEndRef.current && workflow?.steps?.length) {
      stepsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [workflow?.steps?.length]);

  const handleStart = async () => {
    console.log(`[Workflow] Starting with: scanCount=${scanCount}, analyzeTop=${analyzeTop}, autoExecute=${autoExecute}`);
    setIsRunning(true);
    setWorkflow(null);
    
    try {
      const response = await api.startWorkflow(scanCount, analyzeTop, autoExecute);
      console.log("[Workflow] Started:", response);
    } catch (e) {
      console.error("[Workflow] Start error:", e);
      setIsRunning(false);
    }
  };

  const handleCancel = async () => {
    try {
      await api.cancelWorkflow();
    } catch (e) {
      console.error(e);
    }
  };

  const toggleStepExpand = (index: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSteps(newExpanded);
  };

  const getAgentStyle = (agent: string) => {
    const config = agentConfig[agent as keyof typeof agentConfig];
    return config || { icon: Crown, color: "text-navy-400", bg: "bg-navy-800" };
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Command Center"
        subtitle="CrewAI-powered trading workflow with GPT-5 LLM agents"
      />

      <div className="p-6">
        {/* Workflow Controls */}
        <div className="rounded-xl border border-navy-800 bg-gradient-to-br from-navy-900 to-navy-950 p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold text-white">Trading Workflow</h2>
              <p className="text-sm text-navy-400 mt-1">
                Executive Agent orchestrates Scanner and Analysts to find and execute trades
              </p>
            </div>
            
            <div className="flex gap-3">
              {isRunning && (
                <button
                  onClick={handleCancel}
                  className="flex items-center gap-2 h-12 px-6 rounded-lg border border-red-500 bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-colors"
                >
                  <StopCircle className="h-5 w-5" />
                  Cancel
                </button>
              )}
              <ShimmerButton
                onClick={handleStart}
                disabled={isRunning}
                className="h-12 gap-2 px-6"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    Start Workflow
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          {/* Workflow Settings */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-navy-400 mb-2">Stocks to Scan</label>
              <input
                type="number"
                value={scanCount}
                onChange={(e) => setScanCount(parseInt(e.target.value) || 10)}
                className="w-full rounded-lg border border-navy-700 bg-navy-800 px-4 py-2 text-white"
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="block text-sm text-navy-400 mb-2">Analyze Top N</label>
              <input
                type="number"
                value={analyzeTop}
                onChange={(e) => setAnalyzeTop(parseInt(e.target.value) || 5)}
                className="w-full rounded-lg border border-navy-700 bg-navy-800 px-4 py-2 text-white"
                disabled={isRunning}
              />
            </div>
            <div>
              <label className="block text-sm text-navy-400 mb-2">Auto-Execute</label>
              <button
                onClick={() => setAutoExecute(!autoExecute)}
                disabled={isRunning}
                className={cn(
                  "w-full rounded-lg border px-4 py-2 text-sm",
                  autoExecute
                    ? "border-green-500 bg-green-500/10 text-green-500"
                    : "border-navy-700 bg-navy-800 text-navy-300"
                )}
              >
                {autoExecute ? "Enabled" : "Disabled"}
              </button>
            </div>
          </div>
        </div>

        {/* Agent Pipeline Visualization */}
        <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4">Agent Pipeline</h2>
          
          <div className="flex items-center justify-between">
            {[
              { name: "Executive", desc: "Orchestrates workflow", model: "gpt-5.2" },
              { name: "Scanner", desc: "Finds momentum stocks", model: "gpt-5-mini" },
              { name: "Technical", desc: "LLM price analysis", model: "gpt-5-mini" },
              { name: "Fundamental", desc: "News & catalysts", model: "gpt-5-mini" },
              { name: "Risk Manager", desc: "Position sizing", model: "gpt-5-mini" },
              { name: "Executive", desc: "Final decision", model: "gpt-5.2" },
            ].map((agent, i) => {
              const style = getAgentStyle(agent.name);
              const Icon = style.icon;
              const isActive = workflow?.current_step?.includes(agent.name) ||
                (i === 0 && workflow?.status === "starting");
              
              return (
                <div key={i} className="flex items-center">
                  <div className={cn(
                    "flex flex-col items-center p-3 rounded-xl border transition-all min-w-[100px]",
                    isActive
                      ? `${style.bg} border-current ${style.color}`
                      : "border-navy-800 bg-navy-800/50"
                  )}>
                    <div className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-lg mb-2",
                      isActive ? style.bg : "bg-navy-700"
                    )}>
                      {isActive ? (
                        <Loader2 className={cn("h-5 w-5 animate-spin", style.color)} />
                      ) : (
                        <Icon className={cn("h-5 w-5", isActive ? style.color : "text-navy-400")} />
                      )}
                    </div>
                    <p className={cn("font-medium text-xs", isActive ? style.color : "text-white")}>
                      {agent.name}
                    </p>
                    <p className="text-[10px] text-navy-500">{agent.desc}</p>
                    {agent.model && (
                      <p className="text-[9px] text-navy-600 mt-1 font-mono">{agent.model}</p>
                    )}
                  </div>
                  {i < 5 && (
                    <ArrowRight className="h-4 w-4 text-navy-600 mx-1" />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Live Decision Stream */}
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">Decision Stream</h2>
              {isRunning && (
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-xs text-navy-400">Live</span>
                </div>
              )}
            </div>

            {/* Live Thinking Indicator */}
            {isRunning && workflow?.current_step && (
              <div className="mb-4 rounded-xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-orange-500/10 p-4 animate-pulse">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="h-10 w-10 rounded-full bg-gradient-to-r from-purple-500 to-orange-500 flex items-center justify-center">
                      <Brain className="h-5 w-5 text-white animate-pulse" />
                    </div>
                    <div className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-500 to-orange-500 animate-ping opacity-30" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium bg-gradient-to-r from-purple-400 to-orange-400 bg-clip-text text-transparent">
                        Agent Processing
                      </span>
                      <div className="flex gap-1">
                        <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <span className="w-1 h-1 bg-orange-400 rounded-full animate-bounce" style={{ animationDelay: "100ms" }} />
                        <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: "200ms" }} />
                      </div>
                    </div>
                    <p className="text-sm text-navy-300 mt-1">
                      {workflow.current_step}
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {workflow?.steps?.map((step, i) => {
                const style = getAgentStyle(step.agent);
                const Icon = style.icon;
                const hasThinking = step.thinking && step.thinking.length > 0;
                const isExpanded = expandedSteps.has(i);
                
                return (
                  <div
                    key={i}
                    className={cn(
                      "rounded-lg border bg-navy-800/30 p-3 transition-all",
                      hasThinking ? "border-navy-700 cursor-pointer hover:border-navy-600" : "border-navy-800"
                    )}
                    onClick={() => hasThinking && toggleStepExpand(i)}
                  >
                    <div className="flex items-start gap-3">
                      <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg shrink-0", style.bg)}>
                        <Icon className={cn("h-4 w-4", style.color)} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={cn("font-medium text-sm", style.color)}>
                            {step.agent}
                          </span>
                          <span className="text-xs text-navy-500">
                            {new Date(step.time).toLocaleTimeString()}
                          </span>
                          {hasThinking && (
                            <Brain className="h-3 w-3 text-navy-500 ml-auto" />
                          )}
                        </div>
                        <p className="text-sm text-navy-300 mt-1">{step.action}</p>
                        
                        {/* Animated Thinking section */}
                        {hasThinking && isExpanded && (
                          <div className="mt-2 pt-2 border-t border-navy-700 animate-in fade-in slide-in-from-top-2 duration-300">
                            <div className="flex items-center gap-1 text-xs text-navy-500 mb-2">
                              <Brain className="h-3 w-3 animate-pulse" />
                              <span className="bg-gradient-to-r from-purple-400 to-orange-400 bg-clip-text text-transparent font-medium">
                                Agent Thinking
                              </span>
                            </div>
                            <div className="relative pl-3 border-l-2 border-gradient-to-b from-purple-500/50 to-orange-500/50">
                              <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-gradient-to-b from-purple-500 to-orange-500 animate-pulse" />
                              <p className="text-xs text-navy-300 leading-relaxed whitespace-pre-wrap">
                                {step.thinking}
                              </p>
                            </div>
                          </div>
                        )}
                        
                        {hasThinking && !isExpanded && (
                          <div className="flex items-center gap-2 mt-1">
                            <div className="flex gap-1">
                              <span className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                              <span className="w-1.5 h-1.5 bg-orange-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                              <span className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                            </div>
                            <span className="text-xs text-navy-500">
                              Click to reveal thinking...
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={stepsEndRef} />
              
              {!workflow?.steps?.length && !isRunning && (
                <p className="text-navy-400 text-center py-8">
                  Click "Start Workflow" to begin
                </p>
              )}
            </div>
          </div>

          {/* Agent Thinking / Executive Decisions */}
          <div className="rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">
                {workflow?.decisions?.length ? "Executive Decisions" : "Agent Thinking"}
              </h2>
              {isRunning && !workflow?.decisions?.length && (
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-purple-500 animate-pulse" />
                  <span className="text-xs text-purple-400">Live Feed</span>
                </div>
              )}
            </div>

            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {/* Show live agent thoughts when no decisions yet */}
              {isRunning && !workflow?.decisions?.length && workflow?.steps?.length ? (
                <div className="space-y-4">
                  {workflow.steps.slice(-5).reverse().map((step, i) => {
                    const style = getAgentStyle(step.agent);
                    const Icon = style.icon;
                    const isLatest = i === 0;
                    
                    return (
                      <div
                        key={i}
                        className={cn(
                          "rounded-lg border p-4 transition-all",
                          isLatest 
                            ? "border-purple-500/50 bg-gradient-to-r from-purple-500/10 to-orange-500/10"
                            : "border-navy-700 bg-navy-800/30 opacity-70"
                        )}
                      >
                        <div className="flex items-start gap-3">
                          <div className={cn(
                            "flex h-10 w-10 items-center justify-center rounded-lg shrink-0",
                            isLatest ? "bg-gradient-to-r from-purple-500/20 to-orange-500/20" : style.bg
                          )}>
                            {isLatest ? (
                              <Brain className="h-5 w-5 text-purple-400 animate-pulse" />
                            ) : (
                              <Icon className={cn("h-5 w-5", style.color)} />
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className={cn(
                                "font-medium text-sm",
                                isLatest ? "text-purple-400" : style.color
                              )}>
                                {step.agent}
                              </span>
                              <span className="text-xs text-navy-500">
                                {new Date(step.time).toLocaleTimeString()}
                              </span>
                            </div>
                            <p className={cn(
                              "text-sm mb-2",
                              isLatest ? "text-white" : "text-navy-300"
                            )}>
                              {step.action}
                            </p>
                            {step.thinking && (
                              <div className={cn(
                                "text-xs p-2 rounded border-l-2",
                                isLatest 
                                  ? "bg-purple-500/5 border-purple-500 text-purple-300"
                                  : "bg-navy-800/50 border-navy-600 text-navy-400"
                              )}>
                                {step.thinking}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
              
              {/* Show decisions when available */}
              {workflow?.decisions?.map((decision, i) => (
                <div
                  key={i}
                  className={cn(
                    "rounded-lg border p-4",
                    decision.action === "BUY"
                      ? "border-green-500/30 bg-green-500/5"
                      : decision.action === "WATCH"
                      ? "border-yellow-500/30 bg-yellow-500/5"
                      : "border-navy-800 bg-navy-800/30"
                  )}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-lg font-bold text-white">
                        {decision.symbol}
                      </span>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-xs font-medium",
                          decision.action === "BUY"
                            ? "bg-green-500/10 text-green-500"
                            : decision.action === "WATCH"
                            ? "bg-yellow-500/10 text-yellow-500"
                            : "bg-navy-600/10 text-navy-400"
                        )}
                      >
                        {decision.action}
                      </span>
                    </div>
                    <span className="text-sm text-navy-400">
                      {decision.confidence.toFixed(0)}% confidence
                    </span>
                  </div>
                  
                  {decision.quantity > 0 && (
                    <p className="text-sm text-white mb-2">
                      Buy {decision.quantity} shares @ ${decision.price.toFixed(2)}
                      <span className="text-navy-400 ml-2">
                        (${(decision.quantity * decision.price).toFixed(2)})
                      </span>
                    </p>
                  )}
                  
                  <p className="text-sm text-navy-400">{decision.reasoning}</p>
                </div>
              ))}
              
              {!workflow?.decisions?.length && !isRunning && (
                <p className="text-navy-400 text-center py-8">
                  Agent thoughts and decisions will appear here
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Workflow Status */}
        {workflow && (
          <div className="mt-6 rounded-xl border border-navy-800 bg-navy-900/50 p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {workflow.status === "running" ? (
                  <Loader2 className="h-5 w-5 animate-spin text-orange-500" />
                ) : workflow.status === "completed" ? (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                ) : workflow.status === "failed" ? (
                  <XCircle className="h-5 w-5 text-red-500" />
                ) : (
                  <Clock className="h-5 w-5 text-navy-400" />
                )}
                <div>
                  <p className="font-medium text-white">{workflow.current_step}</p>
                  <p className="text-sm text-navy-400">Workflow ID: {workflow.id}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-6 text-sm">
                <div>
                  <span className="text-navy-400">Stocks Found:</span>
                  <span className="ml-2 font-medium text-white">{workflow.stocks.length}</span>
                </div>
                <div>
                  <span className="text-navy-400">Decisions:</span>
                  <span className="ml-2 font-medium text-white">{workflow.decisions.length}</span>
                </div>
                <div>
                  <span className="text-navy-400">Trades:</span>
                  <span className="ml-2 font-medium text-white">{workflow.trades.length}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
