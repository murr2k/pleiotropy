# Pleiotropy Website Deployment Architecture

## Server Constraints & Considerations
- **RAM**: 1.9GB total (limited for full Docker stack)
- **Storage**: 46GB available
- **OS**: AlmaLinux 8.10
- **Web Server**: Apache httpd with SSL configured
- **Domain**: murraykopit.com

## Deployment Strategy

### Option 1: Lightweight Non-Docker Deployment (Recommended)
Given the memory constraints, we'll deploy without Docker:

#### Architecture:
```
┌─────────────────────────────────────────────────┐
│           Apache httpd (Port 80/443)            │
│  ┌──────────────────┬────────────────────────┐  │
│  │ murraykopit.com  │ pleiotropy.murraykopit │  │
│  │  (existing site) │      .com OR           │  │
│  │                  │ murraykopit.com/       │  │
│  │                  │    pleiotropy/         │  │
│  └──────────────────┴────────────────────────┘  │
│                     │                            │
│                     ├─► React Static Files      │
│                     │   (build output)          │
│                     │                            │
│                     └─► Reverse Proxy to:       │
│                         - FastAPI (Port 8080)   │
│                         - Node.js API (Port 8000)│
└─────────────────────────────────────────────────┘

Backend Services (systemd managed):
┌────────────────────┐  ┌─────────────────────┐
│ FastAPI Service    │  │ Redis-lite/SQLite   │
│ (Python 3.6)       │  │ (Embedded DB)       │
│ Port: 8080         │  │                     │
└────────────────────┘  └─────────────────────┘

Optional Analysis Workers (on-demand):
┌────────────────────┐  ┌─────────────────────┐
│ Rust Analyzer      │  │ Python Visualizer   │
│ (subprocess)       │  │ (subprocess)        │
└────────────────────┘  └─────────────────────┘
```

#### Implementation Steps:
1. **Frontend Deployment**:
   - Build React app locally
   - Upload static files to `/var/www/pleiotropy/html/`
   - Configure Apache virtual host

2. **Backend Deployment**:
   - Install Python dependencies in virtual environment
   - Run FastAPI with Gunicorn as systemd service
   - Use SQLite instead of Redis for persistence
   - Implement file-based queuing instead of Redis queues

3. **Analysis Components**:
   - Compile Rust binaries locally
   - Upload to server
   - Execute as subprocesses from API

### Option 2: Minimal Docker Deployment
If we install Docker despite constraints:

#### Architecture:
```
Only deploy critical services:
- Web UI (nginx container) → Apache proxy
- API Backend (FastAPI container)
- SQLite (mounted volume)

Skip:
- Redis (use file-based queue)
- Monitoring stack (Prometheus/Grafana)
- Multiple agent containers
```

### Option 3: Hybrid Cloud Deployment
Deploy compute-intensive parts elsewhere:

#### Architecture:
```
Server (murraykopit.com):
- Static React UI
- Lightweight API proxy
- Result storage

Cloud (AWS Lambda/Google Cloud Run):
- Rust analyzer functions
- Python visualization
- GPU-accelerated analysis
```

## Recommended Approach: Option 1

### Directory Structure on Server:
```
/var/www/pleiotropy/
├── html/               # React build output
├── api/                # FastAPI application
├── data/               # SQLite DB & uploads
├── logs/               # Application logs
└── bin/                # Compiled Rust binaries

/home/webadmin/pleiotropy/
├── venv/               # Python virtual environment
├── config/             # Configuration files
└── scripts/            # Deployment scripts
```

### Apache Configuration:
```apache
# Subdomain approach
<VirtualHost *:443>
    ServerName pleiotropy.murraykopit.com
    DocumentRoot /var/www/pleiotropy/html
    
    # Serve React app
    <Directory /var/www/pleiotropy/html>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # Proxy API requests
    ProxyPass /api http://localhost:8080/api
    ProxyPassReverse /api http://localhost:8080/api
    
    # WebSocket support
    RewriteEngine on
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule ^/?(.*) "ws://localhost:8080/$1" [P,L]
    
    # SSL configuration (reuse existing certs)
    Include /etc/httpd/conf.d/murraykopit-ssl.conf
</VirtualHost>
```

### Systemd Service:
```ini
[Unit]
Description=Pleiotropy API Service
After=network.target

[Service]
Type=exec
User=webadmin
WorkingDirectory=/home/webadmin/pleiotropy
Environment="PATH=/home/webadmin/pleiotropy/venv/bin"
ExecStart=/home/webadmin/pleiotropy/venv/bin/gunicorn \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    api.main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

## Resource Optimization

### Memory Management:
- Limit worker processes to 2
- Use SQLite instead of Redis
- Implement result pagination
- Clear old analysis results periodically

### Performance Tuning:
- Enable Apache caching for static assets
- Compress API responses
- Use CDN for React dependencies
- Implement request rate limiting

### Security Considerations:
- Run services as non-root user
- Implement API authentication
- Use existing SSL certificates
- Configure firewall rules
- Regular security updates

## Deployment Checklist

1. [ ] Create deployment directories
2. [ ] Build React frontend locally
3. [ ] Prepare Python API without Docker dependencies
4. [ ] Compile Rust binaries
5. [ ] Transfer files to server
6. [ ] Set up Python virtual environment
7. [ ] Configure Apache virtual host
8. [ ] Create systemd service
9. [ ] Set up log rotation
10. [ ] Test complete functionality
11. [ ] Configure monitoring alerts
12. [ ] Document maintenance procedures

## Monitoring Without Docker

### Simple Monitoring Setup:
- Use Apache mod_status
- Python logging to files
- Cron job for health checks
- Email alerts for failures
- Basic metrics endpoint in API

## Backup Strategy

### Daily Backups:
```bash
#!/bin/bash
# /home/webadmin/pleiotropy/scripts/backup.sh
DATE=$(date +%Y%m%d)
BACKUP_DIR="/home/webadmin/backups"

# Backup database
cp /var/www/pleiotropy/data/trials.db $BACKUP_DIR/trials-$DATE.db

# Backup uploads
tar czf $BACKUP_DIR/uploads-$DATE.tar.gz /var/www/pleiotropy/data/uploads/

# Keep only last 7 days
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

## Conclusion

The lightweight non-Docker deployment (Option 1) is recommended due to:
- Memory constraints (1.9GB RAM)
- Existing Apache infrastructure
- Simplified maintenance
- Lower resource usage
- Easier debugging

This approach maintains all core functionality while being practical for the VPS environment.