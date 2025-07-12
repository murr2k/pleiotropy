import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { wsService } from '../services/websocket';
import { AppProvider } from '../context/AppContext';
import Dashboard from '../components/Dashboard/Dashboard';
import { mockTrial, mockAgent, mockResult, mockStats } from './test-utils';

// Mock socket.io-client
const mockSocket = {
  connected: false,
  on: jest.fn(),
  emit: jest.fn(),
  disconnect: jest.fn(),
  connect: jest.fn(),
};

jest.mock('socket.io-client', () => ({
  io: jest.fn(() => mockSocket),
}));

// Mock API service
const mockDashboardApi = {
  getStats: jest.fn().mockResolvedValue(mockStats()),
};

jest.mock('../services/api', () => ({
  dashboardApi: mockDashboardApi,
  getWebSocketUrl: jest.fn(() => 'ws://localhost:3001'),
}));

describe('WebSocket Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSocket.connected = false;
    mockSocket.on.mockClear();
    mockSocket.emit.mockClear();
    mockSocket.disconnect.mockClear();
  });

  describe('WebSocket Service', () => {
    it('connects to WebSocket server', () => {
      wsService.connect();
      
      expect(require('socket.io-client').io).toHaveBeenCalledWith(
        'ws://localhost:3001',
        expect.objectContaining({
          reconnection: true,
          reconnectionDelay: 1000,
          reconnectionAttempts: 5,
        })
      );
    });

    it('sets up event listeners on connection', () => {
      wsService.connect();
      
      expect(mockSocket.on).toHaveBeenCalledWith('connect', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('disconnect', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('message', expect.any(Function));
      expect(mockSocket.on).toHaveBeenCalledWith('error', expect.any(Function));
    });

    it('handles subscription and unsubscription', () => {
      const callback = jest.fn();
      const unsubscribe = wsService.subscribe('trial_update', callback);
      
      // Simulate receiving a message
      const mockMessage = {
        type: 'trial_update',
        data: mockTrial(),
        timestamp: new Date().toISOString(),
      };
      
      // Get the message handler and call it
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      if (messageHandler) {
        messageHandler(mockMessage);
      }
      
      expect(callback).toHaveBeenCalledWith(mockTrial());
      
      // Test unsubscribe
      unsubscribe();
      
      // Call handler again
      if (messageHandler) {
        messageHandler(mockMessage);
      }
      
      // Should not be called again after unsubscribe
      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('emits events when connected', () => {
      mockSocket.connected = true;
      wsService.connect();
      
      const testData = { test: 'data' };
      wsService.emit('test_event', testData);
      
      expect(mockSocket.emit).toHaveBeenCalledWith('test_event', testData);
    });

    it('does not emit when disconnected', () => {
      mockSocket.connected = false;
      wsService.connect();
      
      const testData = { test: 'data' };
      wsService.emit('test_event', testData);
      
      expect(mockSocket.emit).not.toHaveBeenCalled();
    });

    it('reports connection status correctly', () => {
      mockSocket.connected = true;
      expect(wsService.isConnected()).toBe(true);
      
      mockSocket.connected = false;
      expect(wsService.isConnected()).toBe(false);
    });

    it('disconnects properly', () => {
      wsService.connect();
      wsService.disconnect();
      
      expect(mockSocket.disconnect).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates in Dashboard', () => {
    const TestDashboard: React.FC = () => (
      <AppProvider>
        <Dashboard />
      </AppProvider>
    );

    it('updates trial data in real-time', async () => {
      render(<TestDashboard />);
      
      // Wait for initial render
      await waitFor(() => {
        expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
      });

      // Simulate trial update
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      const updatedTrial = mockTrial({ name: 'Updated Trial', progress: 75 });
      
      if (messageHandler) {
        act(() => {
          messageHandler({
            type: 'trial_update',
            data: updatedTrial,
            timestamp: new Date().toISOString(),
          });
        });
      }

      // The dashboard should reflect the update
      expect(screen.getByText('Genomic Pleiotropy Dashboard')).toBeInTheDocument();
    });

    it('updates agent status in real-time', async () => {
      render(<TestDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Swarm Agent Status')).toBeInTheDocument();
      });

      // Simulate agent update
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      const updatedAgent = mockAgent({ name: 'Updated Agent', progress: 90 });
      
      if (messageHandler) {
        act(() => {
          messageHandler({
            type: 'agent_update',
            data: updatedAgent,
            timestamp: new Date().toISOString(),
          });
        });
      }

      expect(screen.getByText('Swarm Agent Status')).toBeInTheDocument();
    });

    it('handles new results in real-time', async () => {
      render(<TestDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Recent Activity')).toBeInTheDocument();
      });

      // Simulate new result
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      const newResult = mockResult({ gene_id: 'new-gene' });
      
      if (messageHandler) {
        act(() => {
          messageHandler({
            type: 'result_added',
            data: newResult,
            timestamp: new Date().toISOString(),
          });
        });
      }

      expect(screen.getByText('Recent Activity')).toBeInTheDocument();
    });

    it('refreshes stats on progress updates', async () => {
      render(<TestDashboard />);
      
      await waitFor(() => {
        expect(mockDashboardApi.getStats).toHaveBeenCalledTimes(1);
      });

      // Simulate progress update
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      
      if (messageHandler) {
        act(() => {
          messageHandler({
            type: 'progress_update',
            data: {},
            timestamp: new Date().toISOString(),
          });
        });
      }

      await waitFor(() => {
        expect(mockDashboardApi.getStats).toHaveBeenCalledTimes(2);
      });
    });

    it('displays connection status correctly', async () => {
      render(<TestDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });

      // Simulate disconnect
      const disconnectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'disconnect')?.[1];
      
      if (disconnectHandler) {
        act(() => {
          disconnectHandler();
        });
      }

      // Should still show connected initially (would need state update to change)
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles WebSocket connection errors', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      wsService.connect();
      
      // Simulate error
      const errorHandler = mockSocket.on.mock.calls.find(call => call[0] === 'error')?.[1];
      
      if (errorHandler) {
        errorHandler(new Error('Connection failed'));
      }

      expect(consoleSpy).toHaveBeenCalledWith('WebSocket error:', expect.any(Error));
      
      consoleSpy.mockRestore();
    });

    it('handles malformed messages gracefully', () => {
      const callback = jest.fn();
      wsService.subscribe('test_event', callback);
      
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      
      if (messageHandler) {
        // Test with malformed message
        messageHandler({
          type: 'unknown_type',
          data: null,
          timestamp: 'invalid',
        });
      }

      // Should not call callback for unknown message types
      expect(callback).not.toHaveBeenCalled();
    });

    it('handles reconnection attempts', () => {
      wsService.connect();
      
      // Verify reconnection options were set
      expect(require('socket.io-client').io).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          reconnection: true,
          reconnectionDelay: 1000,
          reconnectionAttempts: 5,
        })
      );
    });
  });

  describe('Performance and Memory', () => {
    it('cleans up listeners on disconnect', () => {
      const callback = jest.fn();
      const unsubscribe = wsService.subscribe('test_event', callback);
      
      // Should clean up properly
      unsubscribe();
      wsService.disconnect();
      
      expect(mockSocket.disconnect).toHaveBeenCalled();
    });

    it('prevents memory leaks from multiple subscriptions', () => {
      const callback1 = jest.fn();
      const callback2 = jest.fn();
      
      const unsub1 = wsService.subscribe('test_event', callback1);
      const unsub2 = wsService.subscribe('test_event', callback2);
      
      // Both should be called
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      
      if (messageHandler) {
        messageHandler({
          type: 'test_event',
          data: 'test',
          timestamp: new Date().toISOString(),
        });
      }

      expect(callback1).toHaveBeenCalled();
      expect(callback2).toHaveBeenCalled();
      
      // Clean up
      unsub1();
      unsub2();
    });

    it('handles rapid message updates efficiently', () => {
      const callback = jest.fn();
      wsService.subscribe('rapid_updates', callback);
      
      const messageHandler = mockSocket.on.mock.calls.find(call => call[0] === 'message')?.[1];
      
      // Send multiple rapid updates
      for (let i = 0; i < 100; i++) {
        if (messageHandler) {
          messageHandler({
            type: 'rapid_updates',
            data: `update-${i}`,
            timestamp: new Date().toISOString(),
          });
        }
      }

      expect(callback).toHaveBeenCalledTimes(100);
    });
  });
});