export interface Experiment {
  n: number;
  iteration?: number;
  target: string;
  description: string;
  reasoning?: string | null;
  status: string;
  kept: boolean;
  score: number | null;
  best_so_far: number | null;
  cost: number | null;
  cum_cost: number | null;
  tokens: string;
  error?: string | null;
}

export interface SoloData {
  meta: {
    metric: string;
    direction: string;
    model: string;
    model_dir: string;
    task: string;
    source: string;
    session: string;
    n_sessions: number;
  };
  experiments: Experiment[];
  best_series: [number, number][];
  best_run_id: string | null;
  baseline: number | null;
  summary: {
    best: number | null;
    count: number;
    kept: number;
    reverted: number;
    attempts: number;
    total_cost: number;
    first_score: number | null;
  };
}

export interface SessionInfo {
  session: string;
  agent: string;
  model: string | null;
  experiments: number;
  attempts: number;
  best: number | null;
  cost: number;
  started: string | null;
}

export interface AgentData {
  label: string;
  color: string;
  points: [number, number][];
  best: number | null;
  done: number;
  attempts: number;
  kept: number;
  cost: number;
  effort: string | null;
  hint: string;
  experiments: {
    n: number;
    target: string;
    status: string;
    score: number | null;
    kept: boolean;
    description: string;
  }[];
}

export interface ParallelData {
  meta: {
    metric: string;
    direction: string;
    run: string;
    n_agents: number;
    total_cost: number;
    best: number | null;
  };
  agents: AgentData[];
  population: [number, number][];
}

export interface ParallelRunInfo {
  name: string;
  n_agents: number | null;
  best: number | null;
  metric: string | null;
  total_cost?: number;
}
