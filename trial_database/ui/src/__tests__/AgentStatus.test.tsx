import React from 'react';
import { render, screen } from '@testing-library/react';
import AgentStatus from '../components/Dashboard/AgentStatus';
import * as AppContext from '../context/AppContext';
import { mockAgent } from './test-utils';

// Mock utility functions
jest.mock('../utils', () => ({
  getStatusColor: jest.fn((status) => {
    const colors: Record<string, string> = {
      working: '#4caf50',
      idle: '#ff9800',
      error: '#f44336',
    };
    return colors[status] || '#9e9e9e';
  }),
  formatRelativeTime: jest.fn((date) => '2 minutes ago'),
}));

describe('AgentStatus Component', () => {
  const mockUseApp = {
    agents: [],
    stats: null,
    wsConnected: true,
    trials: [],
    selectedTrial: null,
    results: [],
    loading: false,
    error: null,
    dispatch: jest.fn(),
    refreshStats: jest.fn(),
  };

  beforeEach(() => {
    jest.spyOn(AppContext, 'useApp').mockReturnValue(mockUseApp);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders no agents message when agents array is empty', () => {
    render(<AgentStatus />);
    
    expect(screen.getByText('No active agents')).toBeInTheDocument();
  });

  it('renders agent information correctly', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Test Agent 1',
        status: 'working',
        current_task: 'Processing genome data',
        progress: 0.75,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    expect(screen.getByText('Test Agent 1')).toBeInTheDocument();
    expect(screen.getByText('working')).toBeInTheDocument();
    expect(screen.getByText('Processing genome data')).toBeInTheDocument();
    expect(screen.getByText('75% complete')).toBeInTheDocument();
    expect(screen.getByText('Last seen 2 minutes ago')).toBeInTheDocument();
  });

  it('renders multiple agents correctly', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Agent 1',
        status: 'working',
        progress: 0.5,
      }),
      mockAgent({
        id: 'agent-2',
        name: 'Agent 2',
        status: 'idle',
        progress: 0,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    expect(screen.getByText('Agent 1')).toBeInTheDocument();
    expect(screen.getByText('Agent 2')).toBeInTheDocument();
    expect(screen.getAllByText(/working|idle/)).toHaveLength(2);
  });

  it('shows progress bar only for working agents', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Working Agent',
        status: 'working',
        progress: 0.6,
      }),
      mockAgent({
        id: 'agent-2',
        name: 'Idle Agent',
        status: 'idle',
        progress: 0,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    // Should show progress for working agent
    expect(screen.getByText('60% complete')).toBeInTheDocument();
    
    // Should have progress bar (LinearProgress component)
    const progressBars = screen.getAllByRole('progressbar');
    expect(progressBars).toHaveLength(1);
  });

  it('handles agents without current task', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Agent Without Task',
        status: 'idle',
        current_task: undefined,
        progress: 0,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    expect(screen.getByText('Agent Without Task')).toBeInTheDocument();
    expect(screen.getByText('idle')).toBeInTheDocument();
    // Should not have any task description
    expect(screen.queryByText('Processing')).not.toBeInTheDocument();
  });

  it('applies correct status colors', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Working Agent',
        status: 'working',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    const statusChip = screen.getByText('working');
    expect(statusChip).toBeInTheDocument();
    
    // Verify that getStatusColor was called with correct status
    const { getStatusColor } = require('../utils');
    expect(getStatusColor).toHaveBeenCalledWith('working');
  });

  it('handles scrollable container for many agents', () => {
    const manyAgents = Array.from({ length: 10 }, (_, i) =>
      mockAgent({
        id: `agent-${i}`,
        name: `Agent ${i}`,
        status: 'working',
      })
    );

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents: manyAgents,
    });

    render(<AgentStatus />);

    // Should render all agents
    manyAgents.forEach((_, i) => {
      expect(screen.getByText(`Agent ${i}`)).toBeInTheDocument();
    });
  });

  it('formats progress percentage correctly', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Test Agent',
        status: 'working',
        progress: 0.333, // Should round to 33%
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    expect(screen.getByText('33% complete')).toBeInTheDocument();
  });

  it('handles error status agents', () => {
    const agents = [
      mockAgent({
        id: 'agent-1',
        name: 'Error Agent',
        status: 'error',
        current_task: 'Failed to process data',
        progress: 0,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    expect(screen.getByText('Error Agent')).toBeInTheDocument();
    expect(screen.getByText('error')).toBeInTheDocument();
    expect(screen.getByText('Failed to process data')).toBeInTheDocument();
  });

  it('renders with proper Material-UI components', () => {
    const agents = [mockAgent()];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      agents,
    });

    render(<AgentStatus />);

    // Check for Material-UI component classes
    const chipElement = screen.getByText('working');
    expect(chipElement).toHaveClass('MuiChip-label');
  });
});