import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  GridLegacy as Grid,
  Slider,
  Typography,
  Box,
} from '@mui/material';
import { useTrials } from '../../hooks/useTrials';
import type { Trial } from '../../types';

interface TrialDialogProps {
  open: boolean;
  onClose: () => void;
  trial?: Trial;
}

const TrialDialog: React.FC<TrialDialogProps> = ({ open, onClose, trial }) => {
  const { createTrial, updateTrial } = useTrials();
  const [formData, setFormData] = useState({
    name: trial?.name || '',
    genome_file: trial?.genome_file || '',
    trait_file: trial?.trait_file || '',
    window_size: trial?.window_size || 1000,
    overlap: trial?.overlap || 100,
    confidence_threshold: trial?.confidence_threshold || 0.8,
  });

  const handleChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    try {
      if (trial) {
        await updateTrial(trial.id, formData);
      } else {
        await createTrial(formData);
      }
      onClose();
    } catch (error) {
      console.error('Failed to save trial:', error);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{trial ? 'Edit Trial' : 'Create New Trial'}</DialogTitle>
      <DialogContent>
        <Grid container spacing={3} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Trial Name"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              required
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Genome File Path"
              value={formData.genome_file}
              onChange={(e) => handleChange('genome_file', e.target.value)}
              required
              helperText="Path to the FASTA genome file"
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Trait Definition File"
              value={formData.trait_file}
              onChange={(e) => handleChange('trait_file', e.target.value)}
              required
              helperText="Path to the JSON trait definition file"
            />
          </Grid>
          
          <Grid item xs={6}>
            <TextField
              fullWidth
              type="number"
              label="Window Size"
              value={formData.window_size}
              onChange={(e) => handleChange('window_size', parseInt(e.target.value))}
              inputProps={{ min: 100, max: 10000 }}
            />
          </Grid>
          
          <Grid item xs={6}>
            <TextField
              fullWidth
              type="number"
              label="Overlap"
              value={formData.overlap}
              onChange={(e) => handleChange('overlap', parseInt(e.target.value))}
              inputProps={{ min: 0, max: formData.window_size - 1 }}
            />
          </Grid>
          
          <Grid item xs={12}>
            <Box>
              <Typography gutterBottom>
                Confidence Threshold: {(formData.confidence_threshold * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={formData.confidence_threshold}
                onChange={(_, value) => handleChange('confidence_threshold', value)}
                min={0}
                max={1}
                step={0.05}
                marks={[
                  { value: 0, label: '0%' },
                  { value: 0.5, label: '50%' },
                  { value: 1, label: '100%' },
                ]}
              />
            </Box>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" color="primary">
          {trial ? 'Update' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TrialDialog;