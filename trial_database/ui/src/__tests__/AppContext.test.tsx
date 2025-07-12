import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { AppProvider, useApp } from '../context/AppContext';
import { mockTrial, mockAgent, mockResult, mockStats } from './test-utils';

// Mock the API service
const mockDashboardApi = {
  getStats: jest.fn(),
};

// Mock the WebSocket service
const mockWsService = {
  connect: jest.fn(),
  disconnect: jest.fn(),
  subscribe: jest.fn(),
};

jest.mock('../services/api', () => ({
  dashboardApi: mockDashboardApi,
}));

jest.mock('../services/websocket', () => ({
  wsService: mockWsService,
}));

// Test component to access context
const TestComponent: React.FC = () => {
  const {
    trials,
    selectedTrial,
    results,
    agents,
    stats,
    loading,
    error,
    wsConnected,
    dispatch,
    refreshStats,
  } = useApp();

  return (
    <div>
      <div data-testid="trials-count">{trials.length}</div>
      <div data-testid="selected-trial">{selectedTrial?.name || 'None'}</div>
      <div data-testid="results-count">{results.length}</div>
      <div data-testid="agents-count">{agents.length}</div>
      <div data-testid="stats">{stats ? JSON.stringify(stats) : 'None'}</div>
      <div data-testid="loading">{loading.toString()}</div>
      <div data-testid="error">{error || 'None'}</div>
      <div data-testid="ws-connected">{wsConnected.toString()}</div>
      <button onClick={() => dispatch({ type: 'SET_TRIALS', payload: [mockTrial()] })}>
        Add Trial
      </button>
      <button onClick={refreshStats}>Refresh Stats</button>
    </div>
  );
};

describe('AppContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockDashboardApi.getStats.mockResolvedValue(mockStats());
    mockWsService.subscribe.mockReturnValue(() => {});
  });

  it('provides initial state correctly', () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    expect(screen.getByTestId('trials-count')).toHaveTextContent('0');
    expect(screen.getByTestId('selected-trial')).toHaveTextContent('None');
    expect(screen.getByTestId('results-count')).toHaveTextContent('0');
    expect(screen.getByTestId('agents-count')).toHaveTextContent('0');
    expect(screen.getByTestId('loading')).toHaveTextContent('false');
    expect(screen.getByTestId('error')).toHaveTextContent('None');
    expect(screen.getByTestId('ws-connected')).toHaveTextContent('true');
  });

  it('connects to WebSocket on mount', () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    expect(mockWsService.connect).toHaveBeenCalled();
    expect(mockWsService.subscribe).toHaveBeenCalled();
  });

  it('fetches initial stats on mount', async () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    await waitFor(() => {
      expect(mockDashboardApi.getStats).toHaveBeenCalled();
    });
  });

  it('updates trials state correctly', async () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    const addButton = screen.getByText('Add Trial');
    
    await act(async () => {
      addButton.click();
    });

    expect(screen.getByTestId('trials-count')).toHaveTextContent('1');
  });

  it('handles SET_SELECTED_TRIAL action', async () => {
    const TestWithSelection: React.FC = () => {
      const { dispatch, selectedTrial } = useApp();
      
      return (
        <div>
          <div data-testid="selected-trial">{selectedTrial?.name || 'None'}</div>
          <button 
            onClick={() => dispatch({ 
              type: 'SET_SELECTED_TRIAL', 
              payload: mockTrial({ name: 'Selected Trial' }) 
            })}
          >
            Select Trial
          </button>
        </div>
      );
    };

    render(
      <AppProvider>
        <TestWithSelection />
      </AppProvider>
    );

    const selectButton = screen.getByText('Select Trial');
    
    await act(async () => {
      selectButton.click();
    });

    expect(screen.getByTestId('selected-trial')).toHaveTextContent('Selected Trial');
  });

  it('handles UPDATE_TRIAL action correctly', async () => {
    const TestWithUpdate: React.FC = () => {
      const { dispatch, trials } = useApp();
      
      React.useEffect(() => {
        // Add initial trial
        dispatch({ type: 'SET_TRIALS', payload: [mockTrial({ id: '1', name: 'Original' })] });
      }, [dispatch]);
      
      return (
        <div>
          <div data-testid="trial-name">{trials[0]?.name || 'None'}</div>
          <button 
            onClick={() => dispatch({ 
              type: 'UPDATE_TRIAL', 
              payload: mockTrial({ id: '1', name: 'Updated Trial' }) 
            })}
          >
            Update Trial
          </button>
        </div>
      );
    };

    render(
      <AppProvider>
        <TestWithUpdate />
      </AppProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('trial-name')).toHaveTextContent('Original');
    });

    const updateButton = screen.getByText('Update Trial');
    
    await act(async () => {
      updateButton.click();
    });

    expect(screen.getByTestId('trial-name')).toHaveTextContent('Updated Trial');
  });

  it('handles ADD_RESULT action', async () => {
    const TestWithResults: React.FC = () => {
      const { dispatch, results } = useApp();
      
      return (
        <div>
          <div data-testid="results-count">{results.length}</div>
          <button 
            onClick={() => dispatch({ 
              type: 'ADD_RESULT', 
              payload: mockResult() 
            })}
          >
            Add Result
          </button>
        </div>
      );
    };

    render(
      <AppProvider>
        <TestWithResults />
      </AppProvider>
    );

    const addButton = screen.getByText('Add Result');
    
    await act(async () => {
      addButton.click();
    });

    expect(screen.getByTestId('results-count')).toHaveTextContent('1');
  });

  it('handles UPDATE_AGENT action', async () => {
    const TestWithAgents: React.FC = () => {
      const { dispatch, agents } = useApp();
      
      React.useEffect(() => {
        dispatch({ type: 'SET_AGENTS', payload: [mockAgent({ id: '1', name: 'Agent 1' })] });
      }, [dispatch]);
      
      return (
        <div>
          <div data-testid="agent-name">{agents[0]?.name || 'None'}</div>
          <button 
            onClick={() => dispatch({ 
              type: 'UPDATE_AGENT', 
              payload: mockAgent({ id: '1', name: 'Updated Agent' }) 
            })}
          >
            Update Agent
          </button>
        </div>
      );
    };

    render(
      <AppProvider>
        <TestWithAgents />
      </AppProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('agent-name')).toHaveTextContent('Agent 1');
    });

    const updateButton = screen.getByText('Update Agent');
    
    await act(async () => {
      updateButton.click();
    });

    expect(screen.getByTestId('agent-name')).toHaveTextContent('Updated Agent');
  });

  it('handles error states', async () => {
    const TestWithError: React.FC = () => {
      const { dispatch, error } = useApp();
      
      return (
        <div>
          <div data-testid="error">{error || 'None'}</div>
          <button 
            onClick={() => dispatch({ 
              type: 'SET_ERROR', 
              payload: 'Test error message' 
            })}
          >
            Set Error
          </button>
        </div>
      );
    };

    render(
      <AppProvider>
        <TestWithError />
      </AppProvider>
    );

    const errorButton = screen.getByText('Set Error');
    
    await act(async () => {
      errorButton.click();
    });

    expect(screen.getByTestId('error')).toHaveTextContent('Test error message');
  });

  it('handles loading states', async () => {
    const TestWithLoading: React.FC = () => {
      const { dispatch, loading } = useApp();
      
      return (
        <div>
          <div data-testid="loading">{loading.toString()}</div>
          <button 
            onClick={() => dispatch({ 
              type: 'SET_LOADING', 
              payload: true 
            })}
          >
            Set Loading
          </button>
        </div>
      );
    };

    render(
      <AppProvider>
        <TestWithLoading />
      </AppProvider>
    );

    const loadingButton = screen.getByText('Set Loading');
    
    await act(async () => {
      loadingButton.click();
    });

    expect(screen.getByTestId('loading')).toHaveTextContent('true');
  });

  it('throws error when useApp is used outside provider', () => {
    // Suppress console.error for this test
    const originalError = console.error;
    console.error = jest.fn();

    expect(() => {
      render(<TestComponent />);
    }).toThrow('useApp must be used within AppProvider');

    console.error = originalError;
  });

  it('refreshStats function works correctly', async () => {
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    const refreshButton = screen.getByText('Refresh Stats');
    
    await act(async () => {
      refreshButton.click();
    });

    expect(mockDashboardApi.getStats).toHaveBeenCalledTimes(2); // Once on mount, once on refresh
  });

  it('handles stats refresh error gracefully', async () => {
    mockDashboardApi.getStats.mockRejectedValueOnce(new Error('API Error'));
    
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    
    render(
      <AppProvider>
        <TestComponent />
      </AppProvider>
    );

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch stats:', expect.any(Error));
    });

    consoleSpy.mockRestore();
  });
});