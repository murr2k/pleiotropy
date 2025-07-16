#!/bin/bash
# Fix deployment issues

echo "=== Fixing Pleiotropy Deployment Issues ==="
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

echo "1. Creating missing directory structure..."
mkdir -p /var/www/murraykopit.com/html
echo "   ✓ Directory created"

echo "2. Creating symbolic link..."
ln -sf "$USER_HOME/website" /var/www/murraykopit.com/html/projects
echo "   ✓ Symbolic link created"
ls -la /var/www/murraykopit.com/html/

echo "3. Checking if API is already running..."
if pgrep -f "api_server.py" > /dev/null; then
    echo "   ! API is already running, stopping it first..."
    pkill -f "api_server.py"
    sleep 2
fi

echo "4. Checking systemd service..."
systemctl stop pleiotropy-api 2>/dev/null
systemctl daemon-reload

echo "5. Starting systemd service..."
systemctl start pleiotropy-api
sleep 3

echo "6. Checking service status..."
if systemctl is-active --quiet pleiotropy-api; then
    echo "   ✓ API service is running"
    systemctl status pleiotropy-api --no-pager
else
    echo "   ✗ API service failed to start"
    echo "   Checking logs:"
    journalctl -u pleiotropy-api -n 20 --no-pager
fi

echo
echo "7. Testing API endpoint..."
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "   ✓ API is responding correctly"
else
    echo "   ✗ API health check failed"
fi

echo
echo "=== Next Steps ==="
echo "1. Check if Apache/Nginx is installed and running"
echo "2. Add the configuration from /tmp/pleiotropy-apache.conf"
echo "3. The website should be accessible once web server is configured"
echo
echo "For local testing without web server:"
echo "cd $USER_HOME/projects/agentic/pleiotropy && ./start_web_server.py"