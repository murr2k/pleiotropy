#!/bin/bash
# Test local deployment setup

echo "=== Testing Pleiotropy Local Deployment ==="
echo

echo "1. Checking website files..."
if [ -d "$HOME/website" ]; then
    echo "   ✓ Website directory exists"
    echo "   Files:"
    find $HOME/website -type f -name "*.html" | head -5
else
    echo "   ✗ Website directory not found"
fi

echo
echo "2. Checking API server..."
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "   ✓ API is running and healthy"
else
    echo "   ✗ API is not responding"
fi

echo
echo "3. Testing API endpoints..."
echo "   Health check:"
curl -s http://localhost:8080/health | jq . 2>/dev/null || echo "   Failed"

echo
echo "   API info:"
curl -s http://localhost:8080/api/info | jq . 2>/dev/null || echo "   Failed"

echo
echo "4. Local access URLs (using port forwarding):"
echo "   Main site: http://localhost:8000"
echo "   API: http://localhost:8080"
echo
echo "To start local web server for testing:"
echo "   cd ~/website && python3 -m http.server 8000"
echo
echo "To create SSH tunnel for remote access:"
echo "   ssh -L 8000:localhost:8000 -L 8080:localhost:8080 user@server"