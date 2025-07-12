import React, { useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import type { ChartOptions } from 'chart.js';
import { useApp } from '../../context/AppContext';
import { format, subDays } from 'date-fns';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrialChart: React.FC = () => {
  const { trials } = useApp();

  const chartData = useMemo(() => {
    // Generate last 7 days
    const days = Array.from({ length: 7 }, (_, i) => {
      const date = subDays(new Date(), 6 - i);
      return format(date, 'MM/dd');
    });

    // Count trials by day and status
    const completedByDay = new Array(7).fill(0);
    const runningByDay = new Array(7).fill(0);
    const failedByDay = new Array(7).fill(0);

    trials.forEach(trial => {
      const trialDate = new Date(trial.created_at);
      const dayIndex = days.findIndex(day => 
        format(trialDate, 'MM/dd') === day
      );

      if (dayIndex !== -1) {
        switch (trial.status) {
          case 'completed':
            completedByDay[dayIndex]++;
            break;
          case 'running':
            runningByDay[dayIndex]++;
            break;
          case 'failed':
            failedByDay[dayIndex]++;
            break;
        }
      }
    });

    return {
      labels: days,
      datasets: [
        {
          label: 'Completed',
          data: completedByDay,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1,
        },
        {
          label: 'Running',
          data: runningByDay,
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.5)',
          tension: 0.1,
        },
        {
          label: 'Failed',
          data: failedByDay,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.1,
        },
      ],
    };
  }, [trials]);

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  };

  return <Line options={options} data={chartData} />;
};

export default TrialChart;