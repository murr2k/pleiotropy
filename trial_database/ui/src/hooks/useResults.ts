import { useState, useEffect, useCallback } from 'react';
import type { FilterOptions } from '../types';
import { resultApi } from '../services/api';
import { useApp } from '../context/AppContext';
import { downloadBlob } from '../utils';

export const useResults = (trialId: string, filters?: FilterOptions) => {
  const { results, dispatch } = useApp();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = useCallback(async () => {
    if (!trialId) return;
    
    setLoading(true);
    setError(null);
    try {
      const data = await resultApi.getByTrialId(trialId, filters);
      dispatch({ type: 'SET_RESULTS', payload: data });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  }, [trialId, filters, dispatch]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  const exportResults = async (format: 'csv' | 'json') => {
    try {
      const blob = await resultApi.exportResults(trialId, format);
      const filename = `trial_${trialId}_results.${format}`;
      downloadBlob(blob, filename);
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Failed to export results');
    }
  };

  return {
    results,
    loading,
    error,
    refetch: fetchResults,
    exportResults,
  };
};