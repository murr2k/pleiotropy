import React from 'react';
import { GridLegacy as Grid, Paper, Typography, Box } from '@mui/material';
import { useApp } from '../../context/AppContext';
import StatsCard from './StatsCard';
import TrialChart from './TrialChart';
import AgentStatus from './AgentStatus';
import RecentActivity from './RecentActivity';

const Dashboard: React.FC = () => {
  const { stats, wsConnected } = useApp();

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Genomic Pleiotropy Dashboard
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              bgcolor: wsConnected ? 'success.main' : 'error.main',
            }}
          />
          <Typography variant="body2" color="text.secondary">
            {wsConnected ? 'Connected' : 'Disconnected'}
          </Typography>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Total Trials"
            value={stats?.total_trials || 0}
            color="primary"
            trend="+12%"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Active Trials"
            value={stats?.active_trials || 0}
            color="info"
            subtitle={`${stats?.active_agents || 0} agents working`}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Completed"
            value={stats?.completed_trials || 0}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Avg Confidence"
            value={`${Math.round((stats?.average_confidence || 0) * 100)}%`}
            color="warning"
          />
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Trial Progress Overview
            </Typography>
            <TrialChart />
          </Paper>
        </Grid>

        {/* Agent Status */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Swarm Agent Status
            </Typography>
            <AgentStatus />
          </Paper>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            <RecentActivity />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;