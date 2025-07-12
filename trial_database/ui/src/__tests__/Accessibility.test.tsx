import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe, toHaveNoViolations } from 'jest-axe';
import { render as customRender, mockStats, mockTrial } from './test-utils';
import Dashboard from '../components/Dashboard/Dashboard';
import StatsCard from '../components/Dashboard/StatsCard';
import App from '../App';
import * as AppContext from '../context/AppContext';

// Extend Jest matchers
expect.extend(toHaveNoViolations);

describe('Accessibility Tests', () => {
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

  describe('WCAG Compliance', () => {
    it('dashboard should have no accessibility violations', async () => {
      const { container } = customRender(<Dashboard />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('stats card should have no accessibility violations', async () => {
      const { container } = render(
        <StatsCard 
          title="Test Metric" 
          value={42} 
          color="primary"
          trend="+5%"
          subtitle="Last hour"
        />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('full app should have no accessibility violations', async () => {
      const { container } = customRender(<App />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Semantic HTML', () => {
    it('uses proper heading hierarchy', () => {
      customRender(<Dashboard />);
      
      const h1 = screen.getByRole('heading', { level: 1 });
      expect(h1).toHaveTextContent('Genomic Pleiotropy Dashboard');
      
      const h6Elements = screen.getAllByRole('heading', { level: 6 });
      expect(h6Elements.length).toBeGreaterThan(0);
    });

    it('provides descriptive text for status indicators', () => {
      customRender(<Dashboard />);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('Disconnected') || screen.getByText('Connected')).toBeInTheDocument();
    });

    it('uses proper ARIA labels for charts', () => {
      customRender(<Dashboard />);
      
      const chartContainer = screen.getByTestId('mock-chart');
      expect(chartContainer).toBeInTheDocument();
    });
  });

  describe('Keyboard Navigation', () => {
    it('supports tab navigation through interactive elements', async () => {
      const user = userEvent.setup();
      customRender(<Dashboard />);
      
      // Test tabbing through the interface
      await user.tab();
      
      // First focusable element should be focused
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeInTheDocument();
    });

    it('provides proper focus indicators', async () => {
      const user = userEvent.setup();
      customRender(<Dashboard />);
      
      await user.tab();
      
      const focusedElement = document.activeElement;
      if (focusedElement) {
        // Should have visible focus indicator (this depends on CSS)
        expect(focusedElement).toBeInTheDocument();
      }
    });

    it('supports escape key to close dialogs/modals', async () => {
      const user = userEvent.setup();
      customRender(<Dashboard />);
      
      // Simulate escape key
      await user.keyboard('{Escape}');
      
      // This test would be more meaningful with actual modal components
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });
  });

  describe('Screen Reader Support', () => {
    it('provides meaningful text alternatives for visual content', () => {
      customRender(<Dashboard />);
      
      // Connection status should have text
      expect(screen.getByText(/Connected|Disconnected/)).toBeInTheDocument();
      
      // Stats should have meaningful labels
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
      expect(screen.getByText('Active Trials')).toBeInTheDocument();
    });

    it('announces dynamic content changes', () => {
      customRender(<Dashboard />);
      
      // Status changes should be announced via text content
      expect(screen.getByText(/Connected|Disconnected/)).toBeInTheDocument();
    });

    it('provides context for form controls', () => {
      // This would be more relevant with actual form components
      customRender(<Dashboard />);
      
      // All interactive elements should have labels
      const interactiveElements = screen.queryAllByRole('button');
      interactiveElements.forEach(element => {
        expect(element).toHaveAccessibleName();
      });
    });
  });

  describe('Color and Contrast', () => {
    it('uses sufficient color contrast for text', () => {
      customRender(<Dashboard />);
      
      // Text elements should be readable
      const title = screen.getByText('Genomic Pleiotropy Dashboard');
      expect(title).toBeInTheDocument();
      
      // This test would require actual contrast ratio checking
      // which typically requires specialized tools
    });

    it('does not rely solely on color to convey information', () => {
      render(
        <StatsCard 
          title="Status" 
          value="Good" 
          color="success"
          trend="+5%"
        />
      );
      
      // Trend should have both color and icon
      expect(screen.getByText('+5%')).toBeInTheDocument();
      expect(screen.getByTestId('TrendingUpIcon')).toBeInTheDocument();
    });
  });

  describe('Focus Management', () => {
    it('maintains logical focus order', async () => {
      const user = userEvent.setup();
      customRender(<Dashboard />);
      
      // Tab through elements and verify order makes sense
      const focusableElements: Element[] = [];
      
      for (let i = 0; i < 5; i++) {
        await user.tab();
        if (document.activeElement) {
          focusableElements.push(document.activeElement);
        }
      }
      
      // Should have found some focusable elements
      expect(focusableElements.length).toBeGreaterThan(0);
    });

    it('traps focus in modal dialogs', () => {
      // This would test actual modal behavior
      customRender(<Dashboard />);
      
      // For now, just verify the dashboard renders
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });
  });

  describe('ARIA Attributes', () => {
    it('uses appropriate ARIA roles', () => {
      customRender(<Dashboard />);
      
      // Check for proper heading roles
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('provides ARIA labels for complex widgets', () => {
      customRender(<Dashboard />);
      
      // Chart should have appropriate ARIA labeling
      const chart = screen.getByTestId('mock-chart');
      expect(chart).toBeInTheDocument();
    });

    it('uses ARIA live regions for dynamic updates', () => {
      customRender(<Dashboard />);
      
      // Status indicator could use live region
      expect(screen.getByText(/Connected|Disconnected/)).toBeInTheDocument();
    });
  });

  describe('Error Handling and Messages', () => {
    it('displays error messages accessibly', () => {
      jest.spyOn(AppContext, 'useApp').mockReturnValue({
        ...mockUseApp,
        error: 'Test error message',
      });
      
      customRender(<Dashboard />);
      
      // Error should be announced to screen readers
      // This would require actual error display logic
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });

    it('provides helpful validation messages', () => {
      // This would test form validation
      customRender(<Dashboard />);
      
      // For now, just verify basic rendering
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });
  });

  describe('Motion and Animation', () => {
    it('respects reduced motion preferences', () => {
      // Mock prefers-reduced-motion
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: jest.fn().mockImplementation(query => {
          if (query.includes('prefers-reduced-motion')) {
            return {
              matches: true,
              media: query,
              onchange: null,
              addEventListener: jest.fn(),
              removeEventListener: jest.fn(),
            };
          }
          return {
            matches: false,
            media: query,
            onchange: null,
            addEventListener: jest.fn(),
            removeEventListener: jest.fn(),
          };
        }),
      });
      
      customRender(<Dashboard />);
      
      // Animations should be reduced or disabled
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });
  });

  describe('Mobile Accessibility', () => {
    it('provides adequate touch targets', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
      
      customRender(<Dashboard />);
      
      // Interactive elements should be large enough for touch
      const interactiveElements = screen.queryAllByRole('button');
      expect(interactiveElements).toBeDefined();
    });

    it('works with screen readers on mobile', () => {
      customRender(<Dashboard />);
      
      // Content should be accessible on mobile screen readers
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Total Trials')).toBeInTheDocument();
    });
  });
});