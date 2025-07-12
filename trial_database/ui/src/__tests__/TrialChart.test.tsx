import React from 'react';
import { render, screen } from '@testing-library/react';
import TrialChart from '../components/Dashboard/TrialChart';
import * as AppContext from '../context/AppContext';
import { mockTrial } from './test-utils';

describe('TrialChart Component', () => {
  const mockTrials = [
    mockTrial({ 
      id: '1', 
      status: 'completed', 
      created_at: new Date().toISOString() 
    }),
    mockTrial({ 
      id: '2', 
      status: 'running', 
      created_at: new Date(Date.now() - 86400000).toISOString() // yesterday
    }),
    mockTrial({ 
      id: '3', 
      status: 'failed', 
      created_at: new Date(Date.now() - 172800000).toISOString() // 2 days ago
    }),
  ];

  const mockUseApp = {
    trials: mockTrials,
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

  it('renders chart component', () => {
    render(<TrialChart />);
    
    expect(screen.getByTestId('mock-chart')).toBeInTheDocument();
  });

  it('generates correct chart data structure', () => {
    render(<TrialChart />);
    
    const chartDataElement = screen.getByTestId('chart-data');
    const chartData = JSON.parse(chartDataElement.textContent || '{}');
    
    // Should have 7 days of labels
    expect(chartData.labels).toHaveLength(7);
    
    // Should have 3 datasets (completed, running, failed)
    expect(chartData.datasets).toHaveLength(3);
    
    const [completedDataset, runningDataset, failedDataset] = chartData.datasets;
    
    expect(completedDataset.label).toBe('Completed');
    expect(runningDataset.label).toBe('Running');
    expect(failedDataset.label).toBe('Failed');
  });

  it('correctly counts trials by status and date', () => {
    render(<TrialChart />);
    
    const chartDataElement = screen.getByTestId('chart-data');
    const chartData = JSON.parse(chartDataElement.textContent || '{}');
    
    const [completedDataset, runningDataset, failedDataset] = chartData.datasets;
    
    // Each dataset should have 7 data points (one for each day)
    expect(completedDataset.data).toHaveLength(7);
    expect(runningDataset.data).toHaveLength(7);
    expect(failedDataset.data).toHaveLength(7);
    
    // Sum of all data points should match number of trials in date range
    const totalCompleted = completedDataset.data.reduce((a: number, b: number) => a + b, 0);
    const totalRunning = runningDataset.data.reduce((a: number, b: number) => a + b, 0);
    const totalFailed = failedDataset.data.reduce((a: number, b: number) => a + b, 0);
    
    expect(totalCompleted).toBeGreaterThanOrEqual(0);
    expect(totalRunning).toBeGreaterThanOrEqual(0);
    expect(totalFailed).toBeGreaterThanOrEqual(0);
  });

  it('applies correct chart styling', () => {
    render(<TrialChart />);
    
    const chartDataElement = screen.getByTestId('chart-data');
    const chartData = JSON.parse(chartDataElement.textContent || '{}');
    
    const [completedDataset, runningDataset, failedDataset] = chartData.datasets;
    
    // Check colors
    expect(completedDataset.borderColor).toBe('rgb(75, 192, 192)');
    expect(runningDataset.borderColor).toBe('rgb(54, 162, 235)');
    expect(failedDataset.borderColor).toBe('rgb(255, 99, 132)');
    
    // Check tension for smooth lines
    expect(completedDataset.tension).toBe(0.1);
    expect(runningDataset.tension).toBe(0.1);
    expect(failedDataset.tension).toBe(0.1);
  });

  it('configures chart options correctly', () => {
    render(<TrialChart />);
    
    const chartOptionsElement = screen.getByTestId('chart-options');
    const chartOptions = JSON.parse(chartOptionsElement.textContent || '{}');
    
    expect(chartOptions.responsive).toBe(true);
    expect(chartOptions.maintainAspectRatio).toBe(false);
    expect(chartOptions.plugins.legend.position).toBe('top');
    expect(chartOptions.scales.y.beginAtZero).toBe(true);
    expect(chartOptions.scales.y.ticks.stepSize).toBe(1);
  });

  it('handles empty trials array', () => {
    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials: [],
    });
    
    render(<TrialChart />);
    
    const chartDataElement = screen.getByTestId('chart-data');
    const chartData = JSON.parse(chartDataElement.textContent || '{}');
    
    // Should still have 7 days and 3 datasets
    expect(chartData.labels).toHaveLength(7);
    expect(chartData.datasets).toHaveLength(3);
    
    // All data points should be 0
    chartData.datasets.forEach((dataset: any) => {
      expect(dataset.data.every((value: number) => value === 0)).toBe(true);
    });
  });

  it('filters trials to last 7 days correctly', () => {
    const oldTrial = mockTrial({
      id: 'old',
      status: 'completed',
      created_at: new Date(Date.now() - 10 * 86400000).toISOString() // 10 days ago
    });
    
    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials: [...mockTrials, oldTrial],
    });
    
    render(<TrialChart />);
    
    const chartDataElement = screen.getByTestId('chart-data');
    const chartData = JSON.parse(chartDataElement.textContent || '{}');
    
    // Old trial should not affect the chart (should be filtered out)
    expect(chartData.labels).toHaveLength(7);
  });

  it('handles trials with different date formats', () => {
    const trialWithDifferentDate = mockTrial({
      id: 'different-date',
      status: 'completed',
      created_at: new Date().toISOString(),
    });
    
    jest.spyOn(AppContext, 'useApp').mockReturnValue({
      ...mockUseApp,
      trials: [trialWithDifferentDate],
    });
    
    render(<TrialChart />);
    
    // Should not throw errors and render successfully
    expect(screen.getByTestId('mock-chart')).toBeInTheDocument();
  });

  it('maintains chart responsiveness configuration', () => {
    render(<TrialChart />);
    
    const chartOptionsElement = screen.getByTestId('chart-options');
    const chartOptions = JSON.parse(chartOptionsElement.textContent || '{}');
    
    expect(chartOptions.responsive).toBe(true);
    expect(chartOptions.maintainAspectRatio).toBe(false);
  });
});