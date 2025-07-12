import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  GridLegacy as Grid,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import type { TrialResult } from '../../types';
import { formatPercentage } from '../../utils';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

interface ResultDetailsProps {
  result: TrialResult;
  open: boolean;
  onClose: () => void;
}

const ResultDetails: React.FC<ResultDetailsProps> = ({ result, open, onClose }) => {
  const confidenceChartData = {
    labels: result.traits,
    datasets: [
      {
        data: result.traits.map(trait => result.confidence_scores[trait] || 0),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            return `${context.label}: ${formatPercentage(context.raw)}`;
          },
        },
      },
    },
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6">Gene Analysis Result</Typography>
          <IconButton onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        <Grid container spacing={3}>
          {/* Gene Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Gene Information
            </Typography>
            <Box sx={{ pl: 2 }}>
              <Typography variant="body2">
                <strong>Gene ID:</strong> {result.gene_id}
              </Typography>
              <Typography variant="body2">
                <strong>Position:</strong> {result.position.start} - {result.position.end}
              </Typography>
              <Typography variant="body2">
                <strong>Length:</strong> {result.position.end - result.position.start + 1} bp
              </Typography>
            </Box>
          </Grid>

          {/* Traits */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Associated Traits
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', pl: 2 }}>
              {result.traits.map(trait => (
                <Chip
                  key={trait}
                  label={`${trait} (${formatPercentage(result.confidence_scores[trait])})`}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>
          </Grid>

          {/* Confidence Chart */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              Confidence Distribution
            </Typography>
            <Box sx={{ height: 250 }}>
              <Doughnut data={confidenceChartData} options={chartOptions} />
            </Box>
          </Grid>

          {/* Regulatory Context */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              Regulatory Context
            </Typography>
            <Box sx={{ pl: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Promoter Strength:</strong> {formatPercentage(result.regulatory_context.promoter_strength)}
              </Typography>
              
              {result.regulatory_context.enhancers.length > 0 && (
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">
                    <strong>Enhancers:</strong>
                  </Typography>
                  <Box sx={{ pl: 2 }}>
                    {result.regulatory_context.enhancers.map((enhancer, idx) => (
                      <Typography key={idx} variant="caption">
                        {enhancer}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              )}

              {result.regulatory_context.silencers.length > 0 && (
                <Box>
                  <Typography variant="body2">
                    <strong>Silencers:</strong>
                  </Typography>
                  <Box sx={{ pl: 2 }}>
                    {result.regulatory_context.silencers.map((silencer, idx) => (
                      <Typography key={idx} variant="caption">
                        {silencer}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          </Grid>

          {/* Codon Frequencies */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Codon Usage Analysis
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Trait</TableCell>
                    <TableCell>Top Codons</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.traits.map(trait => {
                    const frequencies = result.codon_frequencies[trait] || {};
                    const topCodons = Object.entries(frequencies)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 5)
                      .map(([codon, freq]) => `${codon} (${(freq * 100).toFixed(1)}%)`)
                      .join(', ');
                    
                    return (
                      <TableRow key={trait}>
                        <TableCell>{trait}</TableCell>
                        <TableCell>{topCodons || 'No data'}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default ResultDetails;