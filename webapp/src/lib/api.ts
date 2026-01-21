/**
 * TradeSense API Client
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// Types
export interface ScanResult {
  symbol: string;
  score: number;
  momentum_score: number;
  volume_score: number;
  technical_score: number;
  catalyst_score: number;
  price: number;
  rsi: number;
  trend: string;
  catalysts: string[];
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  side: string;
}

export interface Account {
  account_id: string;
  status: string;
  cash: number;
  portfolio_value: number;
  buying_power: number;
  equity: number;
  pnl_today: number;
  pnl_today_pct: number;
}

export interface AgentConfig {
  id: string;
  name: string;
  role: string;
  goal: string;
  model: string;
  temperature: number;
  enabled: boolean;
}

export interface WorkflowStep {
  agent: string;
  action: string;
  thinking?: string;
  time: string;
}

export interface WorkflowDecision {
  symbol: string;
  action: string;
  quantity: number;
  price: number;
  confidence: number;
  reasoning: string;
}

export interface WorkflowStatus {
  id: string;
  status: string;
  current_step: string;
  steps: WorkflowStep[];
  stocks: Array<{ symbol: string; score: number; price: number }>;
  decisions: WorkflowDecision[];
  trades: any[];
}

export interface Trade {
  id: number;
  symbol: string;
  action: string;
  quantity: number;
  price: number;
  total_value: number;
  confidence: number;
  reasoning: string;
  timestamp: string;
}

export interface PortfolioSnapshot {
  id: number;
  portfolio_value: number;
  cash: number;
  positions_value: number;
  pnl_day: number;
  pnl_total: number;
  timestamp: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  }

  // Health
  async health() {
    return this.request<{ status: string }>('/api/health');
  }

  // Portfolio
  async getAccount() {
    return this.request<Account>('/api/portfolio/account');
  }

  async getPositions() {
    return this.request<{ count: number; positions: Position[] }>('/api/portfolio/positions');
  }

  // Budget
  async getBudget() {
    return this.request<{ current_budget: number; history: any[] }>('/api/workflow/budget');
  }

  async modifyBudget(amount: number, action: 'deposit' | 'withdraw', description: string = '') {
    return this.request<{ status: string; new_budget: number }>('/api/workflow/budget', {
      method: 'POST',
      body: JSON.stringify({ amount, action, description }),
    });
  }

  // Workflow
  async startWorkflow(scanCount: number = 10, analyzeTop: number = 5, autoExecute: boolean = false) {
    return this.request<{ status: string; workflow_id: string }>('/api/workflow/start', {
      method: 'POST',
      body: JSON.stringify({ scan_count: scanCount, analyze_top: analyzeTop, auto_execute: autoExecute }),
    });
  }

  async cancelWorkflow() {
    return this.request<{ status: string; message: string }>('/api/workflow/cancel', {
      method: 'POST',
    });
  }

  async getWorkflowStatus() {
    return this.request<WorkflowStatus>('/api/workflow/status');
  }

  async getWorkflowHistory() {
    return this.request<{ workflows: any[] }>('/api/workflow/history');
  }

  async getWorkflowDecisions(workflowId: string) {
    return this.request<{ decisions: any[] }>(`/api/workflow/decisions/${workflowId}`);
  }

  async getTrades() {
    return this.request<{ trades: Trade[] }>('/api/workflow/trades');
  }

  async getPerformance() {
    return this.request<{ stats: any; portfolio_history: PortfolioSnapshot[] }>('/api/workflow/performance');
  }

  // Agents
  async getAgents() {
    return this.request<{ agents: AgentConfig[] }>('/api/agents/');
  }

  async updateAgent(agentId: string, config: Omit<AgentConfig, 'id'>) {
    return this.request<AgentConfig>(`/api/agents/${agentId}`, {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  // Scanner
  async runScan(targetCount: number = 20) {
    return this.request<{ status: string; count: number; results: ScanResult[] }>(
      '/api/scan/run',
      {
        method: 'POST',
        body: JSON.stringify({ target_count: targetCount, use_llm: true }),
      }
    );
  }

  // Trading
  async executeTrade(symbol: string, action: 'buy' | 'sell', quantity: number) {
    return this.request<{ status: string; order_id?: string }>('/api/trade/execute', {
      method: 'POST',
      body: JSON.stringify({ symbol, action, quantity }),
    });
  }

  async closeAllPositions() {
    return this.request<{ status: string }>('/api/portfolio/close-all', {
      method: 'POST',
    });
  }

  // Stocks
  async getStockChart(symbol: string, days: number = 30) {
    return this.request<{
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
    }>(`/api/stocks/${symbol}/chart?days=${days}`);
  }

  async getStockQuote(symbol: string) {
    return this.request<{
      symbol: string;
      price: number;
      open: number;
      high: number;
      low: number;
      volume: number;
      prev_close: number;
      change: number;
      change_pct: number;
    }>(`/api/stocks/${symbol}/quote`);
  }
}

export const api = new ApiClient();
