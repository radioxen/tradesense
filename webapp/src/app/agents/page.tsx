"use client";

import { useState, useEffect } from "react";
import { Header } from "@/components/layout/header";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { api, AgentConfig, Activity } from "@/lib/api";
import {
  Bot,
  TrendingUp,
  FileText,
  Crown,
  Save,
  X,
  Edit2,
  Loader2,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";

const agentIcons: Record<string, React.ElementType> = {
  technical: TrendingUp,
  fundamental: FileText,
  executive: Crown,
};

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingAgent, setEditingAgent] = useState<string | null>(null);
  const [editForm, setEditForm] = useState<Partial<AgentConfig>>({});
  const [saving, setSaving] = useState(false);

  const fetchData = async () => {
    try {
      const [agentsData, activityData] = await Promise.all([
        api.getAgents(),
        api.getActivity(20),
      ]);
      setAgents(agentsData.agents);
      setActivities(activityData.activities);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const startEdit = (agent: AgentConfig) => {
    setEditingAgent(agent.id);
    setEditForm({
      name: agent.name,
      role: agent.role,
      goal: agent.goal,
      model: agent.model,
      temperature: agent.temperature,
      enabled: agent.enabled,
    });
  };

  const cancelEdit = () => {
    setEditingAgent(null);
    setEditForm({});
  };

  const saveEdit = async () => {
    if (!editingAgent) return;
    
    setSaving(true);
    try {
      await api.updateAgent(editingAgent, editForm as Omit<AgentConfig, "id">);
      await fetchData();
      setEditingAgent(null);
      setEditForm({});
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header
        title="AI Agents"
        subtitle="Configure and monitor the multi-agent system"
      />

      <div className="p-6">
        {/* Agent Cards Grid */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {agents.map((agent) => {
            const Icon = agentIcons[agent.id] || Bot;
            const isEditing = editingAgent === agent.id;

            return (
              <div
                key={agent.id}
                className="rounded-xl border border-navy-800 bg-navy-900/50 p-6"
              >
                {isEditing ? (
                  // Edit Mode
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-white">
                        Edit {agent.name}
                      </h3>
                      <button
                        onClick={cancelEdit}
                        className="text-navy-400 hover:text-white"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>

                    <div>
                      <label className="block text-sm text-navy-400 mb-1">
                        Name
                      </label>
                      <input
                        type="text"
                        value={editForm.name || ""}
                        onChange={(e) =>
                          setEditForm({ ...editForm, name: e.target.value })
                        }
                        className="w-full rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 text-sm text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-navy-400 mb-1">
                        Role
                      </label>
                      <textarea
                        value={editForm.role || ""}
                        onChange={(e) =>
                          setEditForm({ ...editForm, role: e.target.value })
                        }
                        rows={2}
                        className="w-full rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 text-sm text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-sm text-navy-400 mb-1">
                        Goal
                      </label>
                      <textarea
                        value={editForm.goal || ""}
                        onChange={(e) =>
                          setEditForm({ ...editForm, goal: e.target.value })
                        }
                        rows={2}
                        className="w-full rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 text-sm text-white"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm text-navy-400 mb-1">
                          Model
                        </label>
                        <select
                          value={editForm.model || "gpt-4o"}
                          onChange={(e) =>
                            setEditForm({ ...editForm, model: e.target.value })
                          }
                          className="w-full rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 text-sm text-white"
                        >
                          <option value="gpt-4o">GPT-4o</option>
                          <option value="gpt-4o-mini">GPT-4o-mini</option>
                          <option value="gpt-4-turbo">GPT-4-turbo</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-navy-400 mb-1">
                          Temperature
                        </label>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.1"
                          value={editForm.temperature || 0.5}
                          onChange={(e) =>
                            setEditForm({
                              ...editForm,
                              temperature: parseFloat(e.target.value),
                            })
                          }
                          className="w-full rounded-lg border border-navy-700 bg-navy-800 px-3 py-2 text-sm text-white"
                        />
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={editForm.enabled ?? true}
                        onChange={(e) =>
                          setEditForm({ ...editForm, enabled: e.target.checked })
                        }
                        className="rounded border-navy-600"
                      />
                      <label className="text-sm text-navy-300">Enabled</label>
                    </div>

                    <ShimmerButton
                      onClick={saveEdit}
                      disabled={saving}
                      className="w-full h-10"
                    >
                      {saving ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <Save className="h-4 w-4 mr-2" />
                          Save Changes
                        </>
                      )}
                    </ShimmerButton>
                  </div>
                ) : (
                  // View Mode
                  <>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
                          <Icon className="h-6 w-6 text-orange-500" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">
                            {agent.name}
                          </h3>
                          <div className="flex items-center gap-2 mt-1">
                            <div
                              className={cn(
                                "h-2 w-2 rounded-full",
                                agent.enabled ? "bg-green-500" : "bg-red-500"
                              )}
                            />
                            <span className="text-xs text-navy-400">
                              {agent.enabled ? "Active" : "Disabled"}
                            </span>
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => startEdit(agent)}
                        className="text-navy-400 hover:text-orange-500"
                      >
                        <Edit2 className="h-5 w-5" />
                      </button>
                    </div>

                    <div className="mt-4 space-y-3">
                      <div>
                        <p className="text-xs text-navy-500 mb-1">Role</p>
                        <p className="text-sm text-navy-300">{agent.role}</p>
                      </div>
                      <div>
                        <p className="text-xs text-navy-500 mb-1">Goal</p>
                        <p className="text-sm text-navy-300">{agent.goal}</p>
                      </div>
                      <div className="flex gap-4">
                        <div>
                          <p className="text-xs text-navy-500">Model</p>
                          <p className="text-sm text-orange-500">{agent.model}</p>
                        </div>
                        <div>
                          <p className="text-xs text-navy-500">Temperature</p>
                          <p className="text-sm text-white">{agent.temperature}</p>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* Recent Agent Decisions */}
        <div className="mt-6 rounded-xl border border-navy-800 bg-navy-900/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">
              Recent Agent Decisions
            </h2>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs text-navy-400">Live</span>
            </div>
          </div>

          {activities.length === 0 ? (
            <p className="text-navy-400 py-8 text-center">
              No recent agent activity. Run a scan or analysis to see decisions.
            </p>
          ) : (
            <div className="space-y-4">
              {activities.slice(0, 10).map((activity, idx) => (
                <div
                  key={idx}
                  className="rounded-lg border border-navy-800 bg-navy-800/30 p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-500/10">
                        <Bot className="h-5 w-5 text-orange-500" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-white">
                            {activity.agent || activity.type}
                          </span>
                          {activity.symbol && activity.symbol !== "-" && (
                            <>
                              <span className="text-navy-500">→</span>
                              <span className="font-mono text-orange-500">
                                {activity.symbol}
                              </span>
                            </>
                          )}
                        </div>
                        <p className="mt-1 text-sm text-navy-400">
                          {activity.details}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "rounded-full px-3 py-1 text-xs font-medium",
                          activity.action === "BUY" ||
                            activity.action === "BULLISH" ||
                            activity.action === "COMPLETED"
                            ? "bg-green-500/10 text-green-500"
                            : activity.action === "SELL" ||
                              activity.action === "BEARISH" ||
                              activity.action === "FAILED"
                            ? "bg-red-500/10 text-red-500"
                            : "bg-navy-600/10 text-navy-400"
                        )}
                      >
                        {activity.action}
                      </span>
                    </div>
                  </div>
                  <div className="mt-2 flex items-center text-xs text-navy-500">
                    <Clock className="h-3 w-3 mr-1" />
                    {new Date(activity.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
