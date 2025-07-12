import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Chip,
  Button,
  TextField,
  InputAdornment,
  IconButton,
  Tooltip,
  Typography,
  LinearProgress,
} from '@mui/material';
import { Search, Download, Visibility } from '@mui/icons-material';
import { useResults } from '../../hooks/useResults';
import { calculateAverageConfidence, formatPercentage, debounce } from '../../utils';
import ResultDetails from './ResultDetails';
import type { TrialResult } from '../../types';

interface ResultsTableProps {
  trialId: string;
}

const ResultsTable: React.FC<ResultsTableProps> = ({ trialId }) => {
  const { results, loading, exportResults } = useResults(trialId);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedResult, setSelectedResult] = useState<TrialResult | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const handleSearch = debounce((value: string) => {
    setSearchTerm(value);
    setPage(0);
  }, 300);

  const filteredResults = results.filter(result =>
    result.gene_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    result.traits.some(trait => trait.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      await exportResults(format);
    } catch (error) {
      console.error('Failed to export results:', error);
    }
  };

  const handleViewDetails = (result: TrialResult) => {
    setSelectedResult(result);
    setDetailsOpen(true);
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <TextField
          size="small"
          placeholder="Search genes or traits..."
          onChange={(e) => handleSearch(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
        />
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            size="small"
            startIcon={<Download />}
            onClick={() => handleExport('csv')}
          >
            Export CSV
          </Button>
          <Button
            size="small"
            startIcon={<Download />}
            onClick={() => handleExport('json')}
          >
            Export JSON
          </Button>
        </Box>
      </Box>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Gene ID</TableCell>
              <TableCell>Position</TableCell>
              <TableCell>Traits</TableCell>
              <TableCell>Avg Confidence</TableCell>
              <TableCell>Promoter Strength</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredResults
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((result) => (
                <TableRow key={result.id}>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {result.gene_id}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {result.position.start}-{result.position.end}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {result.traits.map(trait => (
                        <Chip
                          key={trait}
                          label={trait}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </TableCell>
                  <TableCell>
                    {formatPercentage(calculateAverageConfidence(result.confidence_scores))}
                  </TableCell>
                  <TableCell>
                    {formatPercentage(result.regulatory_context.promoter_strength)}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="View Details">
                      <IconButton
                        size="small"
                        onClick={() => handleViewDetails(result)}
                      >
                        <Visibility />
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
        count={filteredResults.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={(_, newPage) => setPage(newPage)}
        onRowsPerPageChange={(e) => {
          setRowsPerPage(parseInt(e.target.value, 10));
          setPage(0);
        }}
      />

      {selectedResult && (
        <ResultDetails
          result={selectedResult}
          open={detailsOpen}
          onClose={() => setDetailsOpen(false)}
        />
      )}
    </Box>
  );
};

export default ResultsTable;