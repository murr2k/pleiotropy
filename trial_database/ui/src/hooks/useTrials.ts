import { useState, useEffect, useCallback } from 'react';
import type { Trial, FilterOptions } from '../types';
import { trialApi } from '../services/api';
import { useApp } from '../context/AppContext';

export const useTrials = (filters?: FilterOptions) => {
  const { trials, dispatch } = useApp();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTrials = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await trialApi.getAll(filters);
      dispatch({ type: 'SET_TRIALS', payload: data });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trials');
    } finally {
      setLoading(false);
    }
  }, [filters, dispatch]);

  useEffect(() => {
    fetchTrials();
  }, [fetchTrials]);

  const createTrial = async (trial: Partial<Trial>) => {
    try {
      const newTrial = await trialApi.create(trial);
      dispatch({ type: 'UPDATE_TRIAL', payload: newTrial });
      return newTrial;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to create trial');
    }
  };

  const updateTrial = async (id: string, updates: Partial<Trial>) => {
    try {
      const updatedTrial = await trialApi.update(id, updates);
      dispatch({ type: 'UPDATE_TRIAL', payload: updatedTrial });
      return updatedTrial;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to update trial');
    }
  };

  const deleteTrial = async (id: string) => {
    try {
      await trialApi.delete(id);
      await fetchTrials();
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to delete trial');
    }
  };

  const startTrial = async (id: string) => {
    try {
      const trial = await trialApi.start(id);
      dispatch({ type: 'UPDATE_TRIAL', payload: trial });
      return trial;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to start trial');
    }
  };

  const stopTrial = async (id: string) => {
    try {
      const trial = await trialApi.stop(id);
      dispatch({ type: 'UPDATE_TRIAL', payload: trial });
      return trial;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to stop trial');
    }
  };

  return {
    trials,
    loading,
    error,
    refetch: fetchTrials,
    createTrial,
    updateTrial,
    deleteTrial,
    startTrial,
    stopTrial,
  };
};