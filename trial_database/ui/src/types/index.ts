// Trial and swarm-related types

export interface Trial {
  id: string;
  name: string;
  genome_file: string;
  trait_file: string;
  window_size: number;
  overlap: number;
  confidence_threshold: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  start_time?: string;
  end_time?: string;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface SwarmAgent {
  id: string;
  name: string;
  status: 'idle' | 'working' | 'error';
  current_task?: string;
  progress: number;
  last_heartbeat: string;
}

export interface TrialResult {
  id: string;
  trial_id: string;
  gene_id: string;
  traits: string[];
  confidence_scores: Record<string, number>;
  codon_frequencies: Record<string, Record<string, number>>;
  regulatory_context: {
    promoter_strength: number;
    enhancers: string[];
    silencers: string[];
  };
  position: {
    start: number;
    end: number;
  };
  created_at: string;
}

export interface SwarmProgress {
  trial_id: string;
  total_windows: number;
  processed_windows: number;
  active_agents: number;
  estimated_time_remaining: number;
  current_phase: 'initialization' | 'processing' | 'aggregation' | 'complete';
}

export interface DashboardStats {
  total_trials: number;
  active_trials: number;
  completed_trials: number;
  failed_trials: number;
  total_results: number;
  active_agents: number;
  average_confidence: number;
}

export interface WebSocketMessage {
  type: 'trial_update' | 'agent_update' | 'progress_update' | 'result_added' | 'error';
  data: any;
  timestamp: string;
}

export interface FilterOptions {
  status?: Trial['status'][];
  confidence_min?: number;
  confidence_max?: number;
  date_from?: string;
  date_to?: string;
  search?: string;
}

export interface SortOptions {
  field: string;
  direction: 'asc' | 'desc';
}