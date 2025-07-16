#!/bin/bash
# Deployment script for pleiotropy website as subdirectory

echo "=== Pleiotropy Subdirectory Deployment Script ==="
echo

# Check if running as webadmin
if [ "$USER" != "webadmin" ]; then
    echo "Error: This script must be run as webadmin user"
    exit 1
fi

# Set up directories
echo "1. Setting up directory structure..."
mkdir -p ~/website/projects/pleiotropy
mkdir -p ~/pleiotropy/logs

# Check if API is running
echo "2. Checking API status..."
if pgrep -f "api_server.py" > /dev/null; then
    echo "   API is already running"
else
    echo "   Starting API server..."
    cd ~/pleiotropy
    source venv/bin/activate
    nohup python api_server.py > logs/api.log 2>&1 &
    sleep 3
    if pgrep -f "api_server.py" > /dev/null; then
        echo "   API started successfully"
    else
        echo "   Failed to start API"
        exit 1
    fi
fi

# Test API
echo "3. Testing API endpoint..."
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "   API is responding correctly"
else
    echo "   API health check failed"
fi

# Create admin instructions
cat > ~/website/ADMIN_INSTRUCTIONS.txt << 'EOF'
ADMIN SETUP REQUIRED
===================

To complete the deployment, the server administrator needs to:

1. Link the website directory to Apache DocumentRoot:
   sudo ln -s /home/webadmin/website /var/www/murraykopit.com/html/projects

2. Enable Apache modules (if not already enabled):
   sudo a2enmod proxy proxy_http headers rewrite

3. Add to Apache configuration (/etc/httpd/conf.d/murraykopit.conf):

   # Proxy for Pleiotropy API
   <Location /projects/pleiotropy/api>
       ProxyPass http://localhost:8080/api
       ProxyPassReverse http://localhost:8080/api
   </Location>

4. Restart Apache:
   sudo systemctl restart httpd

5. Set up systemd service for API persistence:
   sudo cp /home/webadmin/pleiotropy/pleiotropy-api.service /etc/systemd/system/
   sudo systemctl enable pleiotropy-api
   sudo systemctl start pleiotropy-api

Once complete, the website will be accessible at:
- Main projects page: https://murraykopit.com/projects/
- Pleiotropy app: https://murraykopit.com/projects/pleiotropy/

EOF

echo
echo "4. Directory structure created:"
echo "   ~/website/"
echo "   ├── index.html         (main projects page)"
echo "   ├── style.css"
echo "   ├── .htaccess"
echo "   └── projects/"
echo "       └── pleiotropy/"
echo "           ├── index.html (pleiotropy app)"
echo "           └── style.css"
echo
echo "5. Next steps:"
echo "   - Review ADMIN_INSTRUCTIONS.txt in ~/website/"
echo "   - Contact server admin to complete Apache configuration"
echo
echo "Deployment preparation complete!"