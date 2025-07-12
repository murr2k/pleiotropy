# Genomic Pleiotropy Dashboard UI

A real-time React dashboard for monitoring and managing genomic pleiotropy cryptanalysis trials.

## Features

- Real-time swarm progress monitoring via WebSocket
- Interactive dashboard with live statistics
- Trial management (create, start, stop, delete)
- Results visualization with confidence scores
- Data export (CSV/JSON)
- Responsive design for various screen sizes

## Tech Stack

- React 18 with TypeScript
- Material-UI for components
- Chart.js for data visualization
- Socket.io for real-time updates
- Vite for build tooling

## Setup

1. Install dependencies:
```bash
npm install
```

2. Copy environment configuration:
```bash
cp .env.example .env
```

3. Update `.env` with your API endpoints:
```
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=localhost:8000
```

## Development

Start the development server:
```bash
npm run dev
```

The app will be available at http://localhost:5173

## Build

Create a production build:
```bash
npm run build
```

Preview the build:
```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── Dashboard/     # Dashboard components
│   ├── Layout/        # App layout and navigation
│   ├── Results/       # Results display components
│   └── Trials/        # Trial management components
├── context/           # React context for state management
├── hooks/             # Custom React hooks
├── services/          # API and WebSocket services
├── types/             # TypeScript type definitions
└── utils/             # Utility functions
```

## Key Components

### Dashboard
- Real-time statistics cards
- Trial progress chart
- Swarm agent status monitor
- Recent activity feed

### Trials Management
- Create new trials with configuration
- Start/stop trial execution
- View detailed trial information
- Filter and search capabilities

### Results Viewer
- Tabular display with sorting
- Confidence score visualization
- Codon frequency analysis
- Export functionality

## API Integration

The UI expects the following API endpoints:
- `GET /api/trials` - List all trials
- `POST /api/trials` - Create new trial
- `GET /api/trials/:id` - Get trial details
- `POST /api/trials/:id/start` - Start trial
- `POST /api/trials/:id/stop` - Stop trial
- `GET /api/trials/:id/results` - Get trial results
- `GET /api/dashboard/stats` - Get dashboard statistics

WebSocket events:
- `trial_update` - Trial status changes
- `agent_update` - Agent status updates
- `progress_update` - Progress notifications
- `result_added` - New result notifications
