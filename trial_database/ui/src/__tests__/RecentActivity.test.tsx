import React from 'react';
import { render, screen } from '@testing-library/react';
import RecentActivity from '../components/Dashboard/RecentActivity';
import * as AppContext from '../context/AppContext';
import { mockTrial } from './test-utils';

// Mock utility functions
jest.mock('../utils', () => ({
  getStatusColor: jest.fn((status) => {
    const colors: Record<string, string> = {
      completed: '#4caf50',
      running: '#2196f3',
      failed: '#f44336',
      pending: '#ff9800',
    };
    return colors[status] || '#9e9e9e';
  }),
  formatDate: jest.fn((date) => '2023-01-01 12:00'),
  formatPercentage: jest.fn((value) => `${Math.round(value * 100)}%`),
}));

describe('RecentActivity Component', () => {
  const mockUseApp = {
    trials: [],
    stats: null,
    wsConnected: true,
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

  it('renders no activity message when trials array is empty', () => {
    render(<RecentActivity />);
    
    expect(screen.getByText('No recent activity')).toBeInTheDocument();
  });

  it('renders trial information in table format', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Test Trial 1',
        status: 'completed',
        progress: 1.0,
        confidence_threshold: 0.85,
        updated_at: '2023-01-01T12:00:00Z',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    // Check table headers
    expect(screen.getByText('Trial Name')).toBeInTheDocument();
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('Progress')).toBeInTheDocument();
    expect(screen.getByText('Confidence')).toBeInTheDocument();
    expect(screen.getByText('Updated')).toBeInTheDocument();

    // Check trial data
    expect(screen.getByText('Test Trial 1')).toBeInTheDocument();
    expect(screen.getByText('completed')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('2023-01-01 12:00')).toBeInTheDocument();
  });

  it('sorts trials by most recent update first', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Older Trial',
        updated_at: '2023-01-01T10:00:00Z',
      }),
      mockTrial({
        id: 'trial-2',
        name: 'Newer Trial',
        updated_at: '2023-01-01T12:00:00Z',
      }),
      mockTrial({
        id: 'trial-3',
        name: 'Newest Trial',
        updated_at: '2023-01-01T14:00:00Z',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    const trialNames = screen.getAllByText(/Trial/);
    // Should be sorted by newest first
    expect(trialNames[0]).toHaveTextContent('Newest Trial');
    expect(trialNames[1]).toHaveTextContent('Newer Trial');
    expect(trialNames[2]).toHaveTextContent('Older Trial');
  });

  it('limits display to 5 most recent trials', () => {
    const trials = Array.from({ length: 10 }, (_, i) =>
      mockTrial({
        id: `trial-${i}`,
        name: `Trial ${i}`,
        updated_at: new Date(Date.now() - i * 1000).toISOString(),
      })
    );

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    // Should only display 5 trials
    const tableRows = screen.getAllByRole('row');
    // 1 header row + 5 data rows = 6 total
    expect(tableRows).toHaveLength(6);
  });

  it('handles trials with different statuses', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Completed Trial',
        status: 'completed',
      }),
      mockTrial({
        id: 'trial-2',
        name: 'Running Trial',
        status: 'running',
      }),
      mockTrial({
        id: 'trial-3',
        name: 'Failed Trial',
        status: 'failed',
      }),
      mockTrial({
        id: 'trial-4',
        name: 'Pending Trial',
        status: 'pending',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    expect(screen.getByText('completed')).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('failed')).toBeInTheDocument();
    expect(screen.getByText('pending')).toBeInTheDocument();
  });

  it('handles trials without confidence threshold', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Trial Without Confidence',
        confidence_threshold: undefined,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    expect(screen.getByText('Trial Without Confidence')).toBeInTheDocument();
    expect(screen.getByText('-')).toBeInTheDocument(); // Placeholder for missing confidence
  });

  it('truncates long trial names', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'This is a very long trial name that should be truncated to prevent layout issues',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    const trialNameElement = screen.getByText(/This is a very long trial name/);
    expect(trialNameElement).toBeInTheDocument();
    // The element should have noWrap styling applied
    expect(trialNameElement).toHaveClass('MuiTypography-noWrap');
  });

  it('applies correct status colors to chips', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Test Trial',
        status: 'completed',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    const statusChip = screen.getByText('completed');
    expect(statusChip).toBeInTheDocument();
    
    // Verify that getStatusColor was called
    const { getStatusColor } = require('../utils');
    expect(getStatusColor).toHaveBeenCalledWith('completed');
  });

  it('formats progress and confidence values correctly', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Test Trial',
        progress: 0.756,
        confidence_threshold: 0.923,
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    // Verify formatting functions were called
    const { formatPercentage } = require('../utils');
    expect(formatPercentage).toHaveBeenCalledWith(0.756);
    expect(formatPercentage).toHaveBeenCalledWith(0.923);
  });

  it('formats dates correctly', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Test Trial',
        updated_at: '2023-01-01T12:30:00Z',
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    // Verify formatDate was called
    const { formatDate } = require('../utils');
    expect(formatDate).toHaveBeenCalledWith('2023-01-01T12:30:00Z');
  });

  it('renders proper table structure', () => {
    const trials = [mockTrial()];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    // Check for table structure
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getAllByRole('columnheader')).toHaveLength(5);
    expect(screen.getAllByRole('row')).toHaveLength(2); // 1 header + 1 data row
  });

  it('handles empty confidence threshold gracefully', () => {
    const trials = [
      mockTrial({
        id: 'trial-1',
        name: 'Test Trial',
        confidence_threshold: 0, // Zero value
      }),
    ];

    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials,
    });

    render(<RecentActivity />);

    // Should show formatted zero value
    expect(screen.getByText('0%')).toBeInTheDocument();
  });
});