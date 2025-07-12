import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  Typography,
  Chip,
  LinearProgress,
  IconButton,
  GridLegacy as Grid,
} from '@mui/material';
import { Close, PlayArrow, Stop } from '@mui/icons-material';
import type { Trial } from '../../types';
import { formatDate, formatDuration, formatPercentage, getStatusColor } from '../../utils';
import { useTrials } from '../../hooks/useTrials';
import ResultsTable from '../Results/ResultsTable';

interface TrialDetailsProps {
  trial: Trial;
  open: boolean;
  onClose: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
};

const TrialDetails: React.FC<TrialDetailsProps> = ({ trial, open, onClose }) => {
  const { startTrial, stopTrial } = useTrials();
  const [tabValue, setTabValue] = useState(0);

  const handleAction = async () => {
    try {
      if (trial.status === 'pending') {
        await startTrial(trial.id);
      } else if (trial.status === 'running') {
        await stopTrial(trial.id);
      }
    } catch (error) {
      console.error('Failed to perform action:', error);
    }
  };

  const getDuration = () => {
    if (!trial.start_time) return '-';
    const start = new Date(trial.start_time);
    const end = trial.end_time ? new Date(trial.end_time) : new Date();
    return formatDuration(Math.floor((end.getTime() - start.getTime()) / 1000));
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6">{trial.name}</Typography>
            <Chip
              label={trial.status}
              size="small"
              sx={{
                bgcolor: getStatusColor(trial.status),
                color: 'white',
              }}
            />
          </Box>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Tabs value={tabValue} onChange={(_, value) => setTabValue(value)}>
          <Tab label="Overview" />
          <Tab label="Results" />
          <Tab label="Configuration" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <LinearProgress
                variant="determinate"
                value={trial.progress * 100}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Progress: {formatPercentage(trial.progress)}
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Status Information
              </Typography>
              <Box sx={{ mt: 1 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2">Created:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">{formatDate(trial.created_at)}</Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2">Started:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      {trial.start_time ? formatDate(trial.start_time) : '-'}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2">Duration:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">{getDuration()}</Typography>
                  </Grid>
                </Grid>
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Files
              </Typography>
              <Box sx={{ mt: 1 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Genome: {trial.genome_file}
                </Typography>
                <Typography variant="body2">
                  Traits: {trial.trait_file}
                </Typography>
              </Box>
            </Grid>

            {trial.error && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="error">
                  Error Message
                </Typography>
                <Typography variant="body2" sx={{ mt: 1, fontFamily: 'monospace' }}>
                  {trial.error}
                </Typography>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <ResultsTable trialId={trial.id} />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">Window Size</Typography>
              <Typography variant="body1">{trial.window_size} bp</Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">Overlap</Typography>
              <Typography variant="body1">{trial.overlap} bp</Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">Confidence Threshold</Typography>
              <Typography variant="body1">{formatPercentage(trial.confidence_threshold)}</Typography>
            </Grid>
          </Grid>
        </TabPanel>
      </DialogContent>

      <DialogActions>
        {(trial.status === 'pending' || trial.status === 'running') && (
          <Button
            startIcon={trial.status === 'pending' ? <PlayArrow /> : <Stop />}
            onClick={handleAction}
            variant="contained"
            color={trial.status === 'pending' ? 'primary' : 'warning'}
          >
            {trial.status === 'pending' ? 'Start Trial' : 'Stop Trial'}
          </Button>
        )}
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default TrialDetails;