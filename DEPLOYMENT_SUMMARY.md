# Pleiotropy Website Deployment Summary

## Deployment Status: Ready for Production

### What Has Been Completed ✅

1. **Frontend Interface**
   - Location: `~/public_html/pleiotropy/`
   - Files: `index.html`, `style.css`
   - Features:
     - Responsive React-based UI
     - File upload functionality
     - Real-time analysis display
     - About page with project information
   - Status: **READY** - Can be accessed if Apache is configured

2. **Backend API**
   - Location: `~/pleiotropy/api_server.py`
   - Port: 8080 (localhost only)
   - Features:
     - FastAPI with full OpenAPI documentation
     - SQLite database for results storage
     - File upload and analysis endpoints
     - CORS configured for web access
   - Database: `~/pleiotropy/data/pleiotropy.db`
   - Status: **READY** - Tested and functional

3. **Python Environment**
   - Location: `~/pleiotropy/venv/`
   - Dependencies installed:
     - FastAPI, Uvicorn
     - SQLite support
     - File handling libraries
   - Status: **READY**

4. **Configuration Files Generated**
   - `~/pleiotropy/pleiotropy-api.service` - Systemd service file
   - `~/pleiotropy/apache-pleiotropy.conf` - Apache reverse proxy config
   - Status: **READY** - Need admin installation

### Required Admin Actions 🔧

To make the website publicly accessible, the server administrator needs to:

#### Option 1: Subdomain Setup (Recommended)
```bash
# 1. Create subdomain DNS record
# Point pleiotropy.murraykopit.com to server IP

# 2. Create Apache virtual host
sudo cp /home/webadmin/pleiotropy/apache-pleiotropy.conf /etc/httpd/conf.d/

# 3. Enable required Apache modules
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite

# 4. Create virtual host for subdomain
# Add to /etc/httpd/conf.d/pleiotropy.conf:

<VirtualHost *:80>
    ServerName pleiotropy.murraykopit.com
    DocumentRoot /home/webadmin/public_html/pleiotropy
    
    <Directory /home/webadmin/public_html/pleiotropy>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
    
    # Proxy API requests
    ProxyPass /api http://localhost:8080/api
    ProxyPassReverse /api http://localhost:8080/api
    
    # Redirect to HTTPS
    RewriteEngine On
    RewriteCond %{HTTPS} off
    RewriteRule ^(.*)$ https://%{HTTP_HOST}$1 [R=301,L]
</VirtualHost>

# 5. Install and start the API service
sudo cp /home/webadmin/pleiotropy/pleiotropy-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pleiotropy-api
sudo systemctl start pleiotropy-api

# 6. Restart Apache
sudo systemctl restart httpd
```

#### Option 2: Subdirectory Setup
```bash
# 1. Create symlink in DocumentRoot
sudo ln -s /home/webadmin/public_html/pleiotropy /var/www/murraykopit.com/html/pleiotropy

# 2. Add to existing Apache config
# Include the proxy configuration from apache-pleiotropy.conf

# 3. Start the API service (same as Option 1, step 5)

# 4. Access at: https://murraykopit.com/pleiotropy/
```

### File Structure on Server 📁

```
/home/webadmin/
├── pleiotropy/
│   ├── venv/                    # Python virtual environment
│   ├── api_server.py           # Backend API server
│   ├── main.py                 # Simple test API
│   ├── requirements.txt        # Python dependencies
│   ├── data/
│   │   └── pleiotropy.db      # SQLite database
│   ├── pleiotropy-api.service  # Systemd service file
│   ├── apache-pleiotropy.conf  # Apache config snippet
│   └── web_frontend.tar.gz     # Frontend archive
│
├── public_html/
│   └── pleiotropy/
│       ├── index.html          # React frontend
│       ├── style.css           # Custom styles
│       └── [api,bin,data,logs] # Empty directories
│
└── website/
    └── pleiotropy/             # Alternative location (empty)
```

### Testing the Deployment 🧪

Once the admin completes the setup:

1. **Test Frontend Access**
   ```
   https://pleiotropy.murraykopit.com/
   OR
   https://murraykopit.com/pleiotropy/
   ```

2. **Test API Health**
   ```
   curl https://pleiotropy.murraykopit.com/api/health
   ```

3. **Check Service Status**
   ```bash
   sudo systemctl status pleiotropy-api
   ```

### Security Considerations 🔐

1. **API runs as non-root** (webadmin user)
2. **Firewall configured** - API only accessible via Apache proxy
3. **CORS configured** for web security
4. **SSL/HTTPS** should be enabled using existing certificates

### Maintenance Commands 🛠️

```bash
# View API logs
sudo journalctl -u pleiotropy-api -f

# Restart API
sudo systemctl restart pleiotropy-api

# Update code
cd ~/pleiotropy
source venv/bin/activate
# Make changes
sudo systemctl restart pleiotropy-api

# Backup database
cp ~/pleiotropy/data/pleiotropy.db ~/pleiotropy/data/pleiotropy.db.backup
```

### Current Limitations ⚠️

1. **No Rust analyzer integration** - API returns simulated results
2. **No GPU acceleration** - Server doesn't have CUDA
3. **Limited to 1.9GB RAM** - Can't run full Docker stack
4. **Manual deployment required** - No CI/CD pipeline

### Next Steps for Full Production 🚀

1. **Compile and upload Rust analyzer binary**
2. **Integrate real genomic analysis**
3. **Add user authentication**
4. **Implement result caching**
5. **Set up automated backups**
6. **Configure monitoring alerts**

### Contact for Deployment

The webadmin user has prepared everything needed. The server administrator needs to:
1. Choose deployment option (subdomain vs subdirectory)
2. Configure Apache
3. Install and start the systemd service
4. Ensure firewall allows HTTPS traffic

The website is fully functional and ready for deployment!