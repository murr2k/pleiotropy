import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import { render, mockStats, mockTrial } from './test-utils';
import Dashboard from '../components/Dashboard/Dashboard';
import * as AppContext from '../context/AppContext';

// Mock the API service
jest.mock('../services/api', () => ({
  dashboardApi: {
    getStats: jest.fn().mockResolvedValue(mockStats()),
  },
}));

// Mock the WebSocket service
jest.mock('../services/websocket', () => ({
  wsService: {
    connect: jest.fn(),
    disconnect: jest.fn(),
    subscribe: jest.fn(() => () => {}),
  },
}));

describe('Dashboard Component', () => {
  const mockUseApp = {
    stats: mockStats(),
    wsConnected: true,
    trials: [mockTrial()],
    selectedTrial: null,
    results: [],
    agents: [],
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

  it('renders dashboard title correctly', () => {
    render(<Dashboard />);
    expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
  });

  it('displays connection status indicator', () => {
    render(<Dashboard />);
    expect(screen.getByText('Connected')).toBeInTheDocument();
    
    // Test disconnected state
    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      wsConnected: false,
    });
    
    render(<Dashboard />);
    expect(screen.getByText('Disconnected')).toBeInTheDocument();
  });

  it('renders all stats cards with correct data', () => {
    render(<Dashboard />);
    
    expect(screen.getByText('Total Trials')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
    
    expect(screen.getByText('Active Trials')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('5 agents working')).toBeInTheDocument();
    
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText('6')).toBeInTheDocument();
    
    expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
    expect(screen.getByText('83%')).toBeInTheDocument();
  });

  it('displays trend indicators correctly', () => {
    render(<Dashboard />);
    
    // Should show the +12% trend for total trials
    expect(screen.getByText('+12%')).toBeInTheDocument();
  });

  it('renders chart components', () => {
    render(<Dashboard />);
    
    expect(screen.getByText('Trial Progress Overview')).toBeInTheDocument();
    expect(screen.getByTestId('mock-chart')).toBeInTheDocument();
  });

  it('renders agent status section', () => {
    render(<Dashboard />);
    
    expect(screen.getByText('Swarm Agent Status')).toBeInTheDocument();
  });

  it('renders recent activity section', () => {
    render(<Dashboard />);
    
    expect(screen.getByText('Recent Activity')).toBeInTheDocument();
  });

  it('handles missing stats gracefully', () => {
    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      stats: null,
    });
    
    render(<Dashboard />);
    
    // Should show 0 values when stats are null
    expect(screen.getAllByText('0')).toHaveLength(3); // Total, Active, Completed trials
    expect(screen.getByText('0%')).toBeInTheDocument(); // Avg confidence
  });

  it('applies responsive grid layout', () => {
    render(<Dashboard />);
    
    // Check that grid items have proper responsive classes
    const statsCards = screen.getAllByText(/Total Trials|Active Trials|Completed|Avg Confidence/);
    expect(statsCards).toHaveLength(4);
  });

  it('displays proper Material-UI styling', () => {
    render(<Dashboard />);
    
    // Check for Material-UI components
    expect(screen.getByRole('heading', { level: 1 })).toHaveClass('MuiTypography-h4');
  });
});