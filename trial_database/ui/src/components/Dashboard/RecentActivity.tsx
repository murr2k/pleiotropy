import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Typography,
} from '@mui/material';
import { useApp } from '../../context/AppContext';
import { formatDate, getStatusColor, formatPercentage } from '../../utils';

const RecentActivity: React.FC = () => {
  const { trials } = useApp();

  // Get the 5 most recent trials
  const recentTrials = [...trials]
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    .slice(0, 5);

  if (recentTrials.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
        No recent activity
      </Typography>
    );
  }

  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Trial Name</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Progress</TableCell>
            <TableCell>Confidence</TableCell>
            <TableCell>Updated</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {recentTrials.map(trial => (
            <TableRow key={trial.id}>
              <TableCell>
                <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                  {trial.name}
                </Typography>
              </TableCell>
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
              <TableCell>{formatPercentage(trial.progress)}</TableCell>
              <TableCell>
                {trial.confidence_threshold ? 
                  formatPercentage(trial.confidence_threshold) : 
                  '-'
                }
              </TableCell>
              <TableCell>
                <Typography variant="caption">
                  {formatDate(trial.updated_at)}
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default RecentActivity;