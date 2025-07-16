#!/bin/bash
# Deployment script for pleiotropy website - Run this with sudo in a terminal

echo "=== Pleiotropy Website Deployment Script ==="
echo "This script will deploy the website to production"
echo

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run this script with sudo:"
    echo "sudo $0"
    exit 1
fi

# Get the actual user who called sudo
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$ACTUAL_USER)

echo "1. Creating symbolic link for website..."
if [ -d "$USER_HOME/website" ]; then
    ln -sf "$USER_HOME/website" /var/www/murraykopit.com/html/projects
    echo "   ✓ Symbolic link created"
else
    echo "   ✗ Error: $USER_HOME/website directory not found"
    exit 1
fi

echo "2. Setting up Apache configuration..."
# Check if Apache modules are enabled
a2enmod proxy proxy_http headers rewrite 2>/dev/null || true

# Create Apache configuration snippet
cat > /tmp/pleiotropy-apache.conf << 'EOF'
# Pleiotropy API Proxy Configuration
<Location /projects/pleiotropy/api>
    ProxyPass http://localhost:8080/api
    ProxyPassReverse http://localhost:8080/api
    ProxyPreserveHost On
</Location>

# Allow .htaccess in projects directory
<Directory /var/www/murraykopit.com/html/projects>
    AllowOverride All
    Require all granted
</Directory>
EOF

echo "3. Apache configuration created at /tmp/pleiotropy-apache.conf"
echo "   Please add this to your Apache configuration"

echo "4. Creating systemd service for API..."
cat > /etc/systemd/system/pleiotropy-api.service << EOF
[Unit]
Description=Pleiotropy Genomic Analysis API
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$USER_HOME/projects/agentic/pleiotropy
Environment="PATH=$USER_HOME/projects/agentic/pleiotropy/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$USER_HOME/projects/agentic/pleiotropy/venv/bin/python $USER_HOME/projects/agentic/pleiotropy/api_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "   ✓ Systemd service created"

echo "5. Enabling and starting API service..."
systemctl daemon-reload
systemctl enable pleiotropy-api
systemctl start pleiotropy-api

echo "6. Checking service status..."
sleep 2
if systemctl is-active --quiet pleiotropy-api; then
    echo "   ✓ API service is running"
else
    echo "   ✗ API service failed to start"
    systemctl status pleiotropy-api
fi

echo
echo "=== Deployment Complete ==="
echo
echo "Next steps:"
echo "1. Add the Apache configuration from /tmp/pleiotropy-apache.conf to your Apache config"
echo "2. Restart Apache: systemctl restart apache2 (or httpd)"
echo "3. Test the website at: https://murraykopit.com/projects/"
echo "4. Test the API at: https://murraykopit.com/projects/pleiotropy/api/health"
echo
echo "To check API logs: journalctl -u pleiotropy-api -f"