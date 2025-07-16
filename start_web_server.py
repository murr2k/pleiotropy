#!/usr/bin/env python3
"""
Alternative web server for Pleiotropy deployment
Serves both static files and proxies API requests
"""

import http.server
import socketserver
import urllib.request
import urllib.parse
import os
import json
from http.server import SimpleHTTPRequestHandler

class PleiotropyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        self.base_directory = os.path.expanduser("~/website")
        super().__init__(*args, directory=self.base_directory, **kwargs)
    
    def do_GET(self):
        # Proxy API requests
        if self.path.startswith('/projects/pleiotropy/api'):
            self.proxy_api_request()
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        # Proxy API requests
        if self.path.startswith('/projects/pleiotropy/api'):
            self.proxy_api_request()
        else:
            self.send_error(404, "Not found")
    
    def proxy_api_request(self):
        """Proxy requests to the API server"""
        # Convert path to API endpoint
        api_path = self.path.replace('/projects/pleiotropy/api', '/api')
        api_url = f'http://localhost:8080{api_path}'
        
        try:
            # Set up the request
            headers = {key: val for key, val in self.headers.items() 
                      if key.lower() not in ['host', 'connection']}
            
            # Read POST data if present
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else None
            
            # Make the request
            req = urllib.request.Request(api_url, data=post_data, headers=headers, 
                                       method=self.command)
            
            # Get the response
            with urllib.request.urlopen(req) as response:
                # Send response status
                self.send_response(response.getcode())
                
                # Copy headers
                for header, value in response.headers.items():
                    if header.lower() not in ['connection', 'transfer-encoding']:
                        self.send_header(header, value)
                self.end_headers()
                
                # Copy response body
                self.wfile.write(response.read())
                
        except urllib.error.URLError as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def end_headers(self):
        # Add CORS headers for API requests
        if self.path.startswith('/projects/pleiotropy/api'):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    import sys
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print(f"Starting Pleiotropy web server on port {PORT}")
    print(f"Serving from: {os.path.expanduser('~/website')}")
    print(f"Access the site at: http://localhost:{PORT}/")
    print(f"Projects page: http://localhost:{PORT}/projects/")
    print(f"Pleiotropy app: http://localhost:{PORT}/projects/pleiotropy/")
    print()
    print("Press Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", PORT), PleiotropyHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    main()