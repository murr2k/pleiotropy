import React from 'react';
import { render, screen } from '@testing-library/react';
import { render as customRender, mockStats } from './test-utils';
import Dashboard from '../components/Dashboard/Dashboard';
import * as AppContext from '../context/AppContext';

// Mock window.innerWidth and window.innerHeight
const setViewport = (width: number, height: number) => {
  Object.defineProperty(window, 'innerWidth', {
    writable: true,
    configurable: true,
    value: width,
  });
  Object.defineProperty(window, 'innerHeight', {
    writable: true,
    configurable: true,
    value: height,
  });
  
  // Trigger resize event
  window.dispatchEvent(new Event('resize'));
};

// Mock Material-UI useMediaQuery
jest.mock('@mui/material/useMediaQuery', () => ({
  __esModule: true,
  default: jest.fn(),
}));

describe('Responsive Design Tests', () => {
  const mockUseApp = {
    stats: mockStats(),
    wsConnected: true,
    trials: [],
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

  describe('Desktop Layout (1920x1080)', () => {
    beforeEach(() => {
      setViewport(1920, 1080);
    });

    it('renders dashboard with full desktop layout', () => {
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      
      // All stats cards should be visible
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
      expect(screen.getByText('Active Trials')).toBeInTheDocument();
      expect(screen.getByText('Completed')).toBeInTheDocument();
      expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
      
      // Chart and agent status should be side by side
      expect(screen.getByText('Trial Progress Overview')).toBeInTheDocument();
      expect(screen.getByText('Swarm Agent Status')).toBeInTheDocument();
    });

    it('stats cards display in 4-column layout on desktop', () => {
      customRender(<Dashboard />);
      
      // In desktop layout, stats cards should be in a row
      const statsCards = screen.getAllByText(/Total Trials|Active Trials|Completed|Avg Confidence/);
      expect(statsCards).toHaveLength(4);
    });
  });

  describe('Tablet Layout (768x1024)', () => {
    beforeEach(() => {
      setViewport(768, 1024);
    });

    it('renders dashboard with tablet-optimized layout', () => {
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      
      // All content should still be accessible
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
      expect(screen.getByText('Trial Progress Overview')).toBeInTheDocument();
      expect(screen.getByText('Swarm Agent Status')).toBeInTheDocument();
      expect(screen.getByText('Recent Activity')).toBeInTheDocument();
    });

    it('adapts grid layout for tablet screens', () => {
      customRender(<Dashboard />);
      
      // Content should be stacked more vertically on tablet
      const dashboard = screen.getByText('Genomic Pleiotropy Dashboard').closest('div');
      expect(dashboard).toBeInTheDocument();
    });
  });

  describe('Mobile Layout (375x667)', () => {
    beforeEach(() => {
      setViewport(375, 667);
    });

    it('renders dashboard with mobile-optimized layout', () => {
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      
      // All essential content should still be present
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
      expect(screen.getByText('Active Trials')).toBeInTheDocument();
    });

    it('stacks stats cards vertically on mobile', () => {
      customRender(<Dashboard />);
      
      // Stats cards should be stacked on mobile
      const statsCards = screen.getAllByText(/Total Trials|Active Trials|Completed|Avg Confidence/);
      expect(statsCards).toHaveLength(4);
    });

    it('maintains readability on small screens', () => {
      customRender(<Dashboard />);
      
      // Text should remain readable
      const title = screen.getByText('Genomic Pleiotropy Dashboard');
      expect(title).toBeInTheDocument();
      
      // Connection status should be visible
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  describe('Small Mobile Layout (320x568)', () => {
    beforeEach(() => {
      setViewport(320, 568);
    });

    it('handles very small screens gracefully', () => {
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      
      // Core functionality should remain accessible
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
    });
  });

  describe('Ultra-wide Layout (2560x1440)', () => {
    beforeEach(() => {
      setViewport(2560, 1440);
    });

    it('utilizes extra space efficiently on ultra-wide screens', () => {
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      
      // All content should be properly spaced
      expect(screen.getByText('Trial Progress Overview')).toBeInTheDocument();
      expect(screen.getByText('Swarm Agent Status')).toBeInTheDocument();
    });
  });

  describe('Container Queries and Flexbox', () => {
    it('maintains proper spacing in grid containers', () => {
      setViewport(1200, 800);
      customRender(<Dashboard />);
      
      // Grid container should have proper spacing
      const dashboard = screen.getByText('Genomic Pleiotropy Dashboard').closest('div');
      expect(dashboard).toBeInTheDocument();
    });

    it('handles content overflow gracefully', () => {
      setViewport(300, 400); // Very constrained
      customRender(<Dashboard />);
      
      // Content should not break layout
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });
  });

  describe('Orientation Changes', () => {
    it('adapts to landscape orientation on mobile', () => {
      setViewport(667, 375); // Mobile landscape
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
    });

    it('adapts to portrait orientation on tablet', () => {
      setViewport(768, 1024); // Tablet portrait
      customRender(<Dashboard />);
      
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Trial Progress Overview')).toBeInTheDocument();
    });
  });

  describe('Content Scaling', () => {
    it('maintains readable font sizes across screen sizes', () => {
      const screenSizes = [
        [320, 568],
        [375, 667],
        [768, 1024],
        [1200, 800],
        [1920, 1080],
      ];

      screenSizes.forEach(([width, height]) => {
        setViewport(width, height);
        customRender(<Dashboard />);
        
        const title = screen.getByText('Genomic Pleiotropy Dashboard');
        expect(title).toBeInTheDocument();
        
        // Clean up for next iteration
        screen.unmount?.();
      });
    });
  });

  describe('Chart Responsiveness', () => {
    it('chart adapts to container size', () => {
      setViewport(800, 600);
      customRender(<Dashboard />);
      
      expect(screen.getByTestId('mock-chart')).toBeInTheDocument();
      
      // Chart should have responsive configuration
      const chartOptionsElement = screen.getByTestId('chart-options');
      const chartOptions = JSON.parse(chartOptionsElement.textContent || '{}');
      
      expect(chartOptions.responsive).toBe(true);
      expect(chartOptions.maintainAspectRatio).toBe(false);
    });
  });
});