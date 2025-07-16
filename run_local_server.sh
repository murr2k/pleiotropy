#!/bin/bash
# Simple script to run the Pleiotropy website locally

echo "=== Starting Pleiotropy Local Server ==="
echo

# Kill any existing processes
echo "1. Stopping any existing servers..."
pkill -f api_server.py 2>/dev/null
pkill -f start_web_server.py 2>/dev/null
sleep 2

# Start the web server (which includes API proxy)
echo "2. Starting web server with API proxy..."
echo "   Access the site at: http://localhost:8000/"
echo "   Projects page: http://localhost:8000/projects/"
echo "   Pleiotropy app: http://localhost:8000/projects/pleiotropy/"
echo
echo "   Press Ctrl+C to stop the server"
echo

# Run the server
python3 start_web_server.py