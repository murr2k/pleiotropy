import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material';
import { BrowserRouter } from 'react-router-dom';
import { AppProvider } from '../context/AppContext';

// Test theme matching the app theme
const testTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

interface AllTheProvidersProps {
  children: React.ReactNode;
}

const AllTheProviders: React.FC<AllTheProvidersProps> = ({ children }) => {
  return (
    <ThemeProvider theme={testTheme}>
      <BrowserRouter>
        <AppProvider>
          {children}
        </AppProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options });

export * from '@testing-library/react';
export { customRender as render };

// Mock data generators
export const mockTrial = (overrides = {}) => ({
  id: 'trial-1',
  name: 'Test Trial',
  genome_file: 'test_genome.fasta',
  trait_file: 'test_traits.json',
  window_size: 1000,
  overlap: 100,
  confidence_threshold: 0.8,
  status: 'completed' as const,
  progress: 100,
  created_at: '2023-01-01T00:00:00Z',
  updated_at: '2023-01-01T01:00:00Z',
  ...overrides,
});

export const mockAgent = (overrides = {}) => ({
  id: 'agent-1',
  name: 'Test Agent',
  status: 'working' as const,
  current_task: 'Processing genome windows',
  progress: 75,
  last_heartbeat: '2023-01-01T00:00:00Z',
  ...overrides,
});

export const mockResult = (overrides = {}) => ({
  id: 'result-1',
  trial_id: 'trial-1',
  gene_id: 'gene-1',
  traits: ['growth_rate', 'antibiotic_resistance'],
  confidence_scores: {
    growth_rate: 0.85,
    antibiotic_resistance: 0.92,
  },
  codon_frequencies: {
    growth_rate: { ATG: 0.3, TGA: 0.2 },
    antibiotic_resistance: { ATG: 0.4, TGA: 0.1 },
  },
  regulatory_context: {
    promoter_strength: 0.7,
    enhancers: ['enhancer-1'],
    silencers: ['silencer-1'],
  },
  position: {
    start: 1000,
    end: 2000,
  },
  created_at: '2023-01-01T00:00:00Z',
  ...overrides,
});

export const mockStats = (overrides = {}) => ({
  total_trials: 10,
  active_trials: 3,
  completed_trials: 6,
  failed_trials: 1,
  total_results: 45,
  active_agents: 5,
  average_confidence: 0.83,
  ...overrides,
});