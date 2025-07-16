# Web Deployment Documentation

## Overview
The Pleiotropy Genomic Cryptanalysis project includes a full web interface with React frontend and FastAPI backend, deployed as a subdirectory application.

## Architecture

### Frontend
- **Framework**: React with TypeScript
- **Features**: File upload, real-time analysis, results visualization
- **Location**: `~/website/projects/pleiotropy/`
- **Entry Point**: `index.html`

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (lightweight alternative to Redis)
- **Port**: 8080
- **Service**: Systemd unit `pleiotropy-api.service`

### Web Server
- **Production**: Apache/Nginx with proxy configuration
- **Development**: Python HTTP server with API proxy
- **Port**: 8001 (development), 443/80 (production)

## Deployment Status (July 15, 2025)

### ✅ Completed
1. Website structure created at `~/website/`
2. API server running as systemd service
3. Symbolic link created: `/var/www/murraykopit.com/html/projects`
4. Local deployment successful on port 8001
5. All deployment scripts created and tested

### 📁 File Structure
```
~/website/
├── index.html              # Main projects page (tabbed interface)
├── style.css               # Main styles
├── projects/
│   └── pleiotropy/
│       ├── index.html      # Pleiotropy React app
│       └── style.css       # App styles

~/projects/agentic/pleiotropy/
├── api_server.py           # FastAPI backend
├── deploy_with_sudo.sh     # Production deployment script
├── fix_deployment.sh       # Deployment fix script
├── start_web_server.py     # Python web server with proxy
├── run_local_server.sh     # Quick start script
└── venv/                   # Python virtual environment
```

## Deployment Instructions

### Local Development
```bash
# Quick start
./run_local_server.sh

# Manual start
python3 start_web_server.py 8001

# Access at http://localhost:8001/projects/pleiotropy/
```

### Production Deployment
```bash
# Deploy with sudo
sudo ./deploy_with_sudo.sh

# Fix any issues
sudo ./fix_deployment.sh

# Configure Apache (add to config)
<Location /projects/pleiotropy/api>
    ProxyPass http://localhost:8080/api
    ProxyPassReverse http://localhost:8080/api
</Location>
```

### Remote Access
```bash
# SSH tunnel from local machine
ssh -L 8001:localhost:8001 -L 8080:localhost:8080 user@server

# Access at http://localhost:8001/projects/pleiotropy/
```

## Service Management

### API Service
```bash
# Status
sudo systemctl status pleiotropy-api

# Start/Stop/Restart
sudo systemctl start pleiotropy-api
sudo systemctl stop pleiotropy-api
sudo systemctl restart pleiotropy-api

# Logs
sudo journalctl -u pleiotropy-api -f
```

### Web Server
```bash
# Find process
ps aux | grep start_web_server

# Kill and restart
pkill -f start_web_server
./run_local_server.sh
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/info` - API information
- `POST /api/analyze` - Submit genome for analysis
- `GET /api/analysis/{id}` - Get analysis results
- `GET /api/analyses` - List all analyses

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port
lsof -i :8000
lsof -i :8001

# Use different port
python3 start_web_server.py 8002
```

### API Not Responding
```bash
# Check if running
curl http://localhost:8080/health

# Restart service
sudo systemctl restart pleiotropy-api

# Check logs
sudo journalctl -u pleiotropy-api -n 50
```

### Permission Denied
```bash
# Fix permissions
chmod +x deploy_with_sudo.sh
chmod +x fix_deployment.sh
chmod +x run_local_server.sh
```

## Security Considerations

1. **CORS**: Configured to allow all origins (update for production)
2. **Authentication**: Not implemented (add for production)
3. **File Upload**: Limited to FASTA files
4. **API Rate Limiting**: Not implemented (recommended)
5. **HTTPS**: Configure SSL certificates for production

## Future Enhancements

1. Add user authentication
2. Implement WebSocket for real-time updates
3. Add result caching with Redis
4. Deploy to cloud platform (AWS/GCP/Azure)
5. Set up CI/CD pipeline
6. Add monitoring and alerting