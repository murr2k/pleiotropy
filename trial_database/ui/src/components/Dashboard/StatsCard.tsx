import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface StatsCardProps {
  title: string;
  value: number | string;
  color: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
  trend?: string;
  subtitle?: string;
}

const StatsCard: React.FC<StatsCardProps> = ({ title, value, color, trend, subtitle }) => {
  const isPositiveTrend = trend?.startsWith('+');

  return (
    <Card>
      <CardContent>
        <Typography color="text.secondary" gutterBottom variant="overline">
          {title}
        </Typography>
        <Typography variant="h4" component="div" color={`${color}.main`}>
          {value}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
          {trend && (
            <Box sx={{ display: 'flex', alignItems: 'center', mr: 1 }}>
              {isPositiveTrend ? (
                <TrendingUpIcon fontSize="small" color="success" />
              ) : (
                <TrendingDownIcon fontSize="small" color="error" />
              )}
              <Typography
                variant="body2"
                color={isPositiveTrend ? 'success.main' : 'error.main'}
              >
                {trend}
              </Typography>
            </Box>
          )}
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default StatsCard;