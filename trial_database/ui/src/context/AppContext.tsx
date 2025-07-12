import React, { createContext, useContext, useReducer, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { Trial, TrialResult, SwarmAgent, DashboardStats } from '../types';
import { wsService } from '../services/websocket';
import { dashboardApi } from '../services/api';

interface AppState {
  trials: Trial[];
  selectedTrial: Trial | null;
  results: TrialResult[];
  agents: SwarmAgent[];
  stats: DashboardStats | null;
  loading: boolean;
  error: string | null;
  wsConnected: boolean;
}

type AppAction =
  | { type: 'SET_TRIALS'; payload: Trial[] }
  | { type: 'SET_SELECTED_TRIAL'; payload: Trial | null }
  | { type: 'UPDATE_TRIAL'; payload: Trial }
  | { type: 'SET_RESULTS'; payload: TrialResult[] }
  | { type: 'ADD_RESULT'; payload: TrialResult }
  | { type: 'SET_AGENTS'; payload: SwarmAgent[] }
  | { type: 'UPDATE_AGENT'; payload: SwarmAgent }
  | { type: 'SET_STATS'; payload: DashboardStats }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_WS_CONNECTED'; payload: boolean };

const initialState: AppState = {
  trials: [],
  selectedTrial: null,
  results: [],
  agents: [],
  stats: null,
  loading: false,
  error: null,
  wsConnected: false,
};

const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_TRIALS':
      return { ...state, trials: action.payload };
    
    case 'SET_SELECTED_TRIAL':
      return { ...state, selectedTrial: action.payload };
    
    case 'UPDATE_TRIAL':
      return {
        ...state,
        trials: state.trials.map(t => 
          t.id === action.payload.id ? action.payload : t
        ),
        selectedTrial: state.selectedTrial?.id === action.payload.id 
          ? action.payload 
          : state.selectedTrial,
      };
    
    case 'SET_RESULTS':
      return { ...state, results: action.payload };
    
    case 'ADD_RESULT':
      return { ...state, results: [...state.results, action.payload] };
    
    case 'SET_AGENTS':
      return { ...state, agents: action.payload };
    
    case 'UPDATE_AGENT':
      return {
        ...state,
        agents: state.agents.map(a => 
          a.id === action.payload.id ? action.payload : a
        ),
      };
    
    case 'SET_STATS':
      return { ...state, stats: action.payload };
    
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    case 'SET_WS_CONNECTED':
      return { ...state, wsConnected: action.payload };
    
    default:
      return state;
  }
};

interface AppContextValue extends AppState {
  dispatch: React.Dispatch<AppAction>;
  refreshStats: () => Promise<void>;
}

const AppContext = createContext<AppContextValue | null>(null);

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const refreshStats = async () => {
    try {
      const stats = await dashboardApi.getStats();
      dispatch({ type: 'SET_STATS', payload: stats });
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  useEffect(() => {
    // Connect WebSocket
    wsService.connect();
    dispatch({ type: 'SET_WS_CONNECTED', payload: true });

    // Subscribe to WebSocket events
    const unsubscribers = [
      wsService.subscribe('trial_update', (trial: Trial) => {
        dispatch({ type: 'UPDATE_TRIAL', payload: trial });
      }),
      wsService.subscribe('agent_update', (agent: SwarmAgent) => {
        dispatch({ type: 'UPDATE_AGENT', payload: agent });
      }),
      wsService.subscribe('result_added', (result: TrialResult) => {
        dispatch({ type: 'ADD_RESULT', payload: result });
      }),
      wsService.subscribe('progress_update', () => {
        refreshStats();
      }),
    ];

    // Initial stats load
    refreshStats();

    return () => {
      unsubscribers.forEach(unsub => unsub());
      wsService.disconnect();
      dispatch({ type: 'SET_WS_CONNECTED', payload: false });
    };
  }, []);

  const value: AppContextValue = {
    ...state,
    dispatch,
    refreshStats,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export default AppContext;