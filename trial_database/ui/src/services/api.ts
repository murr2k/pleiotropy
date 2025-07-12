import axios from 'axios';
import type { Trial, TrialResult, DashboardStats, FilterOptions } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Trial endpoints
export const trialApi = {
  getAll: async (filters?: FilterOptions) => {
    const response = await api.get<Trial[]>('/trials', { params: filters });
    return response.data;
  },

  getById: async (id: string) => {
    const response = await api.get<Trial>(`/trials/${id}`);
    return response.data;
  },

  create: async (trial: Partial<Trial>) => {
    const response = await api.post<Trial>('/trials', trial);
    return response.data;
  },

  update: async (id: string, updates: Partial<Trial>) => {
    const response = await api.patch<Trial>(`/trials/${id}`, updates);
    return response.data;
  },

  delete: async (id: string) => {
    await api.delete(`/trials/${id}`);
  },

  start: async (id: string) => {
    const response = await api.post<Trial>(`/trials/${id}/start`);
    return response.data;
  },

  stop: async (id: string) => {
    const response = await api.post<Trial>(`/trials/${id}/stop`);
    return response.data;
  },
};

// Result endpoints
export const resultApi = {
  getByTrialId: async (trialId: string, filters?: FilterOptions) => {
    const response = await api.get<TrialResult[]>(`/trials/${trialId}/results`, { params: filters });
    return response.data;
  },

  getById: async (id: string) => {
    const response = await api.get<TrialResult>(`/results/${id}`);
    return response.data;
  },

  exportResults: async (trialId: string, format: 'csv' | 'json') => {
    const response = await api.get(`/trials/${trialId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },
};

// Dashboard endpoints
export const dashboardApi = {
  getStats: async () => {
    const response = await api.get<DashboardStats>('/dashboard/stats');
    return response.data;
  },

  getSwarmStatus: async () => {
    const response = await api.get('/swarm/status');
    return response.data;
  },
};

// WebSocket connection
export const getWebSocketUrl = () => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsHost = import.meta.env.VITE_WS_URL || 'localhost:8000';
  return `${wsProtocol}//${wsHost}/ws`;
};

export default api;