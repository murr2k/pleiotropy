import React from 'react';
import { Box, Typography, LinearProgress, Chip, Stack } from '@mui/material';
import { useApp } from '../../context/AppContext';
import { getStatusColor, formatRelativeTime } from '../../utils';

const AgentStatus: React.FC = () => {
  const { agents } = useApp();

  if (agents.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body2" color="text.secondary">
          No active agents
        </Typography>
      </Box>
    );
  }

  return (
    <Stack spacing={2} sx={{ maxHeight: 320, overflow: 'auto' }}>
      {agents.map(agent => (
        <Box key={agent.id} sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2">{agent.name}</Typography>
            <Chip
              label={agent.status}
              size="small"
              sx={{
                bgcolor: getStatusColor(agent.status),
                color: 'white',
              }}
            />
          </Box>
          
          {agent.current_task && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {agent.current_task}
            </Typography>
          )}
          
          {agent.status === 'working' && (
            <Box sx={{ mb: 1 }}>
              <LinearProgress
                variant="determinate"
                value={agent.progress * 100}
                sx={{ height: 6, borderRadius: 3 }}
              />
              <Typography variant="caption" color="text.secondary">
                {Math.round(agent.progress * 100)}% complete
              </Typography>
            </Box>
          )}
          
          <Typography variant="caption" color="text.secondary">
            Last seen {formatRelativeTime(agent.last_heartbeat)}
          </Typography>
        </Box>
      ))}
    </Stack>
  );
};

export default AgentStatus;