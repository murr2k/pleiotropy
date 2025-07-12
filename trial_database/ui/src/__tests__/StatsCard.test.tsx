import React from 'react';
import { render, screen } from '@testing-library/react';
import StatsCard from '../components/Dashboard/StatsCard';

describe('StatsCard Component', () => {
  it('renders basic card with title and value', () => {
    render(
      <StatsCard 
        title="Test Metric" 
        value={42} 
        color="primary"
      />
    );
    
    expect(screen.getByText('Test Metric')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('renders with string values', () => {
    render(
      <StatsCard 
        title="Success Rate" 
        value="95%" 
        color="success"
      />
    );
    
    expect(screen.getByText('Success Rate')).toBeInTheDocument();
    expect(screen.getByText('95%')).toBeInTheDocument();
  });

  it('displays positive trend correctly', () => {
    render(
      <StatsCard 
        title="Growth" 
        value={100} 
        color="primary"
        trend="+15%"
      />
    );
    
    expect(screen.getByText('+15%')).toBeInTheDocument();
    // Should have trending up icon
    expect(screen.getByTestId('TrendingUpIcon')).toBeInTheDocument();
  });

  it('displays negative trend correctly', () => {
    render(
      <StatsCard 
        title="Errors" 
        value={5} 
        color="error"
        trend="-8%"
      />
    );
    
    expect(screen.getByText('-8%')).toBeInTheDocument();
    // Should have trending down icon
    expect(screen.getByTestId('TrendingDownIcon')).toBeInTheDocument();
  });

  it('displays subtitle when provided', () => {
    render(
      <StatsCard 
        title="Active Tasks" 
        value={12} 
        color="info"
        subtitle="3 agents working"
      />
    );
    
    expect(screen.getByText('Active Tasks')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('3 agents working')).toBeInTheDocument();
  });

  it('renders without trend or subtitle', () => {
    render(
      <StatsCard 
        title="Simple Metric" 
        value={999} 
        color="secondary"
      />
    );
    
    expect(screen.getByText('Simple Metric')).toBeInTheDocument();
    expect(screen.getByText('999')).toBeInTheDocument();
    
    // Should not have trend icons
    expect(screen.queryByTestId('TrendingUpIcon')).not.toBeInTheDocument();
    expect(screen.queryByTestId('TrendingDownIcon')).not.toBeInTheDocument();
  });

  it('applies correct color styling', () => {
    const { rerender } = render(
      <StatsCard 
        title="Primary Test" 
        value={10} 
        color="primary"
      />
    );
    
    expect(screen.getByText('10')).toHaveStyle({ color: expect.stringContaining('primary') });
    
    rerender(
      <StatsCard 
        title="Success Test" 
        value={20} 
        color="success"
      />
    );
    
    expect(screen.getByText('20')).toHaveStyle({ color: expect.stringContaining('success') });
  });

  it('handles zero values correctly', () => {
    render(
      <StatsCard 
        title="Zero Value" 
        value={0} 
        color="warning"
      />
    );
    
    expect(screen.getByText('Zero Value')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('handles large numbers correctly', () => {
    render(
      <StatsCard 
        title="Large Number" 
        value={1234567} 
        color="info"
      />
    );
    
    expect(screen.getByText('Large Number')).toBeInTheDocument();
    expect(screen.getByText('1234567')).toBeInTheDocument();
  });

  it('applies proper Material-UI card structure', () => {
    render(
      <StatsCard 
        title="Test Structure" 
        value={100} 
        color="primary"
      />
    );
    
    // Check for Material-UI Card components
    expect(screen.getByText('Test Structure')).toHaveClass('MuiTypography-overline');
    expect(screen.getByText('100')).toHaveClass('MuiTypography-h4');
  });

  it('renders trend and subtitle together', () => {
    render(
      <StatsCard 
        title="Complex Card" 
        value={50} 
        color="warning"
        trend="+5%"
        subtitle="Last hour"
      />
    );
    
    expect(screen.getByText('Complex Card')).toBeInTheDocument();
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(screen.getByText('+5%')).toBeInTheDocument();
    expect(screen.getByText('Last hour')).toBeInTheDocument();
    expect(screen.getByTestId('TrendingUpIcon')).toBeInTheDocument();
  });
});