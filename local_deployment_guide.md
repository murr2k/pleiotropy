# Pleiotropy Local Deployment Guide

## Current Situation
- ✅ Website files ready at `~/website/`
- ✅ API server can run
- ⚠️ No web server fully configured on this system
- ⚠️ `/var/www/murraykopit.com/html` doesn't exist (suggesting this isn't the production server)

## Immediate Solution - Local Testing

### Option 1: Use the Python Web Server (Recommended)
```bash
# Terminal 1 - Stop any existing API server first
pkill -f api_server.py

# Terminal 2 - Start the complete server
cd /home/murr2k/projects/agentic/pleiotropy
./start_web_server.py
```

Then access:
- Main page: http://localhost:8000/
- Projects: http://localhost:8000/projects/
- Pleiotropy: http://localhost:8000/projects/pleiotropy/

### Option 2: Fix the Deployment Issues
Run this in your terminal:
```bash
sudo ./fix_deployment.sh
```

This will:
1. Create the missing directories
2. Set up the symbolic link
3. Fix the systemd service

## For Production Deployment

If this is meant to be deployed on a different server (murraykopit.com), you need to:

1. **Transfer the files** to the production server:
```bash
# Create deployment package
cd ~/projects/agentic/pleiotropy
tar -czf pleiotropy-deployment.tar.gz \
  website_package.tar.gz \
  api_server.py \
  deploy_with_sudo.sh \
  fix_deployment.sh \
  requirements.txt

# Transfer to production server
scp pleiotropy-deployment.tar.gz user@murraykopit.com:~/
```

2. **On the production server**, extract and run:
```bash
tar -xzf pleiotropy-deployment.tar.gz
sudo ./deploy_with_sudo.sh
```

## Testing Without Web Server

The Python web server (`start_web_server.py`) includes:
- Static file serving from `~/website/`
- API proxy to port 8080
- CORS headers for API requests
- Full functionality without needing Apache/Nginx

## Current Issues Summary

1. **No production web server**: This system doesn't have Apache/Nginx properly installed
2. **Wrong domain path**: `/var/www/murraykopit.com/` suggests this should be on the actual murraykopit.com server
3. **API conflict**: The systemd service conflicts with the already-running API

## Recommended Next Steps

1. **For local testing**: Use `./start_web_server.py`
2. **For production**: Deploy to the actual murraykopit.com server
3. **Alternative**: Install and configure Nginx locally if needed