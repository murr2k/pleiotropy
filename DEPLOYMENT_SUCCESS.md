# 🎉 Pleiotropy Website Deployment - SUCCESS!

## ✅ Everything is now working!

### System Status:
- ✅ **API Server**: Running as systemd service on port 8080
- ✅ **Web Server**: Running on port 8001 with full functionality
- ✅ **Directory Structure**: Properly set up with symbolic link
- ✅ **All Files**: Website and app files ready at ~/website/

## 🌐 Access URLs:

### Local Access:
- **Main Projects Page**: http://localhost:8001/projects/
- **Pleiotropy App**: http://localhost:8001/projects/pleiotropy/
- **API Health Check**: http://localhost:8080/health

### Remote Access (SSH Tunnel):
From your local machine:
```bash
ssh -L 8001:localhost:8001 -L 8080:localhost:8080 murr2k@server
```
Then access http://localhost:8001/projects/pleiotropy/ in your browser

## 🔧 Service Management:

### Check Status:
```bash
# API Service
sudo systemctl status pleiotropy-api

# Web Server Process
ps aux | grep start_web_server
```

### Restart Services:
```bash
# API Service
sudo systemctl restart pleiotropy-api

# Web Server (kill and restart)
pkill -f start_web_server
python3 /home/murr2k/projects/agentic/pleiotropy/start_web_server.py 8001 &
```

### View Logs:
```bash
# API Logs
sudo journalctl -u pleiotropy-api -f

# Web Server (runs in foreground, shows logs directly)
```

## 📁 File Locations:
- Website files: `~/website/`
- API server: `~/projects/agentic/pleiotropy/api_server.py`
- Symbolic link: `/var/www/murraykopit.com/html/projects -> /home/murr2k/website`

## 🚀 What's Working:
1. Full website with tabbed project interface
2. Pleiotropy genomic analysis app
3. File upload and analysis (demo mode)
4. API endpoints for health, info, and analysis
5. Proper URL routing through subdirectories

## 🔒 Production Considerations:
To deploy this on a real domain (murraykopit.com):
1. Transfer these files to the production server
2. Configure Apache/Nginx with the proxy rules
3. Set up SSL certificates
4. Configure firewall rules
5. Set up proper domain DNS

## 💡 Quick Test:
Open your browser and go to:
http://localhost:8001/projects/pleiotropy/

You should see the Pleiotropy Genomic Cryptanalysis interface!

---
Deployment completed successfully on July 15, 2025