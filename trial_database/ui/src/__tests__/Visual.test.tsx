import React from 'react';
import { render } from '@testing-library/react';
import { render as customRender, mockStats, mockTrial, mockAgent } from './test-utils';
import Dashboard from '../components/Dashboard/Dashboard';
import StatsCard from '../components/Dashboard/StatsCard';
import * as AppContext from '../context/AppContext';

// Mock html2canvas for screenshot testing
jest.mock('html2canvas', () => ({
  __esModule: true,
  default: jest.fn(() => Promise.resolve({
    toDataURL: () => 'data:image/png;base64,mock-screenshot-data',
    width: 1200,
    height: 800,
  })),
}));

describe('Visual Regression Tests', () => {
  const mockUseApp = {
    stats: mockStats(),
    wsConnected: true,
    trials: [mockTrial()],
    selectedTrial: null,
    results: [],
    agents: [mockAgent()],
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

  describe('Component Visual Consistency', () => {
    it('dashboard renders consistently', () => {
      const { container } = customRender(<Dashboard />);
      
      // Take a visual snapshot (in a real implementation, this would use a visual testing tool)
      expect(container.firstChild).toMatchSnapshot();
    });

    it('stats cards render consistently with different props', () => {
      const statsCardVariants = [
        { title: 'Total Trials', value: 10, color: 'primary' as const },
        { title: 'Active Trials', value: 3, color: 'info' as const, subtitle: '5 agents working' },
        { title: 'Completed', value: 6, color: 'success' as const, trend: '+12%' },
        { title: 'Failed', value: 1, color: 'error' as const, trend: '-5%' },
      ];

      statsCardVariants.forEach((props, index) => {
        const { container } = render(<StatsCard {...props} />);
        expect(container.firstChild).toMatchSnapshot(`stats-card-variant-${index}`);
      });
    });

    it('handles different data states visually', () => {
      const dataStates = [
        { name: 'empty-state', stats: null },
        { name: 'loading-state', stats: mockStats(), loading: true },
        { name: 'error-state', stats: mockStats(), error: 'Connection failed' },
        { name: 'disconnected-state', stats: mockStats(), wsConnected: false },
      ];

      dataStates.forEach(({ name, ...stateOverrides }) => {
        jest.spyOn(AppContext, 'useApp').mockReturnValue({
          ...mockUseApp,
          ...stateOverrides,
        });

        const { container } = customRender(<Dashboard />);
        expect(container.firstChild).toMatchSnapshot(`dashboard-${name}`);
      });
    });
  });

  describe('Responsive Visual Testing', () => {
    const viewports = [
      { name: 'mobile', width: 375, height: 667 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'desktop', width: 1200, height: 800 },
      { name: 'large-desktop', width: 1920, height: 1080 },
    ];

    viewports.forEach(({ name, width, height }) => {
      it(`renders correctly on ${name} viewport`, () => {
        // Mock viewport size
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

        const { container } = customRender(<Dashboard />);
        expect(container.firstChild).toMatchSnapshot(`dashboard-${name}-${width}x${height}`);
      });
    });
  });

  describe('Theme and Styling Consistency', () => {
    it('maintains consistent Material-UI theme application', () => {
      const { container } = customRender(<Dashboard />);
      
      // Verify theme classes are applied
      const elements = container.querySelectorAll('[class*="Mui"]');
      expect(elements.length).toBeGreaterThan(0);
      
      // Take snapshot to ensure styling consistency
      expect(container.firstChild).toMatchSnapshot('themed-dashboard');
    });

    it('applies consistent color palette', () => {
      const colors = ['primary', 'secondary', 'success', 'error', 'warning', 'info'] as const;
      
      colors.forEach(color => {
        const { container } = render(
          <StatsCard 
            title={`${color} Card`} 
            value={100} 
            color={color}
          />
        );
        expect(container.firstChild).toMatchSnapshot(`stats-card-${color}`);
      });
    });
  });

  describe('Chart Visual Consistency', () => {
    it('chart renders with consistent styling', () => {
      const { container } = customRender(<Dashboard />);
      
      // Find the chart component
      const chart = container.querySelector('[data-testid="mock-chart"]');
      expect(chart).toBeInTheDocument();
      expect(chart).toMatchSnapshot('trial-chart');
    });

    it('chart adapts visually to different data sets', () => {
      const trialSets = [
        [], // Empty
        [mockTrial()], // Single trial
        [mockTrial(), mockTrial({ status: 'failed' }), mockTrial({ status: 'running' })], // Mixed
      ];

      trialSets.forEach((trials, index) => {
        jest.spyOn(AppContext, 'useApp').mockReturnValue({
          ...mockUseApp,
          trials,
        });

        const { container } = customRender(<Dashboard />);
        const chart = container.querySelector('[data-testid="mock-chart"]');
        expect(chart).toMatchSnapshot(`chart-data-set-${index}`);
      });
    });
  });

  describe('Interactive State Visuals', () => {
    it('connection indicator shows correct visual state', () => {
      const connectionStates = [
        { connected: true, label: 'connected' },
        { connected: false, label: 'disconnected' },
      ];

      connectionStates.forEach(({ connected, label }) => {
        jest.spyOn(AppContext, 'useApp').mockReturnValue({
          ...mockUseApp,
          wsConnected: connected,
        });

        const { container } = customRender(<Dashboard />);
        const statusArea = container.querySelector('[data-testid="connection-status"]') || 
                          container; // Fallback to container if no specific element
        expect(statusArea).toMatchSnapshot(`connection-${label}`);
      });
    });

    it('trend indicators display correctly', () => {
      const trends = ['+15%', '-8%', '+0%'];
      
      trends.forEach(trend => {
        const { container } = render(
          <StatsCard 
            title="Trend Test" 
            value={100} 
            color="primary"
            trend={trend}
          />
        );
        expect(container.firstChild).toMatchSnapshot(`trend-${trend.replace(/[+\-%]/g, '')}`);
      });
    });
  });

  describe('Layout and Spacing', () => {
    it('maintains consistent grid spacing', () => {
      const { container } = customRender(<Dashboard />);
      
      // Check grid layout
      const gridItems = container.querySelectorAll('[class*="MuiGrid-item"]');
      expect(gridItems.length).toBeGreaterThan(0);
      
      // Snapshot for layout consistency
      expect(container.firstChild).toMatchSnapshot('grid-layout');
    });

    it('paper components have consistent elevation and spacing', () => {
      const { container } = customRender(<Dashboard />);
      
      const papers = container.querySelectorAll('[class*="MuiPaper"]');
      expect(papers.length).toBeGreaterThan(0);
      
      papers.forEach((paper, index) => {
        expect(paper).toMatchSnapshot(`paper-component-${index}`);
      });
    });
  });

  describe('Error State Visuals', () => {
    it('displays error states with proper styling', () => {
      jest.spyOn(AppContext, 'useApp').mockReturnValue({
        ...mockUseApp,
        error: 'Failed to connect to server',
      });

      const { container } = customRender(<Dashboard />);
      expect(container.firstChild).toMatchSnapshot('error-state-display');
    });

    it('loading states maintain visual consistency', () => {
      jest.spyOn(AppContext, 'useApp').mockReturnValue({
        ...mockUseApp,
        loading: true,
      });

      const { container } = customRender(<Dashboard />);
      expect(container.firstChild).toMatchSnapshot('loading-state-display');
    });
  });

  describe('Cross-browser Compatibility Visuals', () => {
    it('renders consistently across different user agents', () => {
      const userAgents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
      ];

      userAgents.forEach((userAgent, index) => {
        Object.defineProperty(navigator, 'userAgent', {
          value: userAgent,
          configurable: true,
        });

        const { container } = customRender(<Dashboard />);
        expect(container.firstChild).toMatchSnapshot(`browser-compat-${index}`);
      });
    });
  });
});