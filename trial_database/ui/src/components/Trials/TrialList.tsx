import React, { useState } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  IconButton,
  Chip,
  Tooltip,
  LinearProgress,
  Typography,
  TextField,
  InputAdornment,
  Button,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Delete,
  Visibility,
  Search,
  Add,
} from '@mui/icons-material';
import { useTrials } from '../../hooks/useTrials';
import { getStatusColor, formatDate, formatPercentage, debounce } from '../../utils';
import type { Trial } from '../../types';
import TrialDialog from './TrialDialog';
import TrialDetails from './TrialDetails';

const TrialList: React.FC = () => {
  const { trials, loading, startTrial, stopTrial, deleteTrial } = useTrials();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedTrial, setSelectedTrial] = useState<Trial | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const handleSearch = debounce((value: string) => {
    setSearchTerm(value);
    setPage(0);
  }, 300);

  const filteredTrials = trials.filter(trial =>
    trial.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    trial.genome_file.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleStartTrial = async (id: string) => {
    try {
      await startTrial(id);
    } catch (error) {
      console.error('Failed to start trial:', error);
    }
  };

  const handleStopTrial = async (id: string) => {
    try {
      await stopTrial(id);
    } catch (error) {
      console.error('Failed to stop trial:', error);
    }
  };

  const handleDeleteTrial = async (id: string) => {
    if (window.confirm('Are you sure you want to delete this trial?')) {
      try {
        await deleteTrial(id);
      } catch (error) {
        console.error('Failed to delete trial:', error);
      }
    }
  };

  const handleViewDetails = (trial: Trial) => {
    setSelectedTrial(trial);
    setDetailsOpen(true);
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Trials Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateDialogOpen(true)}
        >
          New Trial
        </Button>
      </Box>

      <Paper>
        <Box sx={{ p: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Search trials..."
            onChange={(e) => handleSearch(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
          />
        </Box>

        {loading && <LinearProgress />}

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Progress</TableCell>
                <TableCell>Genome File</TableCell>
                <TableCell>Window Size</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Created</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredTrials
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((trial) => (
                  <TableRow key={trial.id}>
                    <TableCell>{trial.name}</TableCell>
                    <TableCell>
                      <Chip
                        label={trial.status}
                        size="small"
                        sx={{
                          bgcolor: getStatusColor(trial.status),
                          color: 'white',
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={trial.progress * 100}
                          sx={{ width: 100, height: 6, borderRadius: 3 }}
                        />
                        <Typography variant="caption">
                          {formatPercentage(trial.progress)}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>{trial.genome_file}</TableCell>
                    <TableCell>{trial.window_size}</TableCell>
                    <TableCell>{formatPercentage(trial.confidence_threshold)}</TableCell>
                    <TableCell>{formatDate(trial.created_at)}</TableCell>
                    <TableCell align="right">
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleViewDetails(trial)}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      {trial.status === 'pending' && (
                        <Tooltip title="Start Trial">
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => handleStartTrial(trial.id)}
                          >
                            <PlayArrow />
                          </IconButton>
                        </Tooltip>
                      )}
                      {trial.status === 'running' && (
                        <Tooltip title="Stop Trial">
                          <IconButton
                            size="small"
                            color="warning"
                            onClick={() => handleStopTrial(trial.id)}
                          >
                            <Stop />
                          </IconButton>
                        </Tooltip>
                      )}
                      <Tooltip title="Delete Trial">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteTrial(trial.id)}
                          disabled={trial.status === 'running'}
                        >
                          <Delete />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>

        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={filteredTrials.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={(_, newPage) => setPage(newPage)}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
        />
      </Paper>

      <TrialDialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
      />

      {selectedTrial && (
        <TrialDetails
          trial={selectedTrial}
          open={detailsOpen}
          onClose={() => setDetailsOpen(false)}
        />
      )}
    </Box>
  );
};

export default TrialList;