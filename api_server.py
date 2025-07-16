#!/usr/bin/env python3
"""
Pleiotropy Genomic Analysis API Server
Lightweight implementation for VPS deployment
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sqlite3
import json
import os
import hashlib
import datetime
from typing import Dict, List, Optional
import subprocess
import tempfile

# Initialize FastAPI app
app = FastAPI(
    title="Pleiotropy Genomic Analysis API",
    version="1.0.0",
    description="Web API for genomic pleiotropy analysis using cryptanalytic methods"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = os.path.expanduser("~/pleiotropy/data/pleiotropy.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            results TEXT,
            error_message TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS genes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT,
            gene_name TEXT,
            confidence REAL,
            traits TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pleiotropy Genomic Analysis API",
        "status": "online",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pleiotropy-api"}

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Pleiotropy Genomic Cryptanalysis System",
        "version": "1.0.0",
        "description": "Web API for genomic pleiotropy analysis using cryptanalytic methods",
        "features": [
            "Genomic sequence analysis",
            "Trait correlation detection",
            "Codon usage analysis",
            "Real-time processing"
        ],
        "endpoints": {
            "GET /": "API root",
            "GET /health": "Health check",
            "GET /api/info": "API information",
            "POST /api/analyze": "Submit sequence for analysis",
            "GET /api/analysis/{id}": "Get analysis results",
            "GET /api/analyses": "List all analyses"
        }
    }

@app.post("/api/analyze")
async def analyze_sequence(file: UploadFile = File(...)):
    """Analyze uploaded genomic sequence"""
    try:
        # Generate unique analysis ID
        content = await file.read()
        analysis_id = hashlib.md5(content + str(datetime.datetime.now()).encode()).hexdigest()[:12]
        
        # Save file temporarily
        temp_path = f"/tmp/{analysis_id}.fasta"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Store analysis record
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analyses (id, filename, status) VALUES (?, ?, ?)",
            (analysis_id, file.filename, "processing")
        )
        conn.commit()
        conn.close()
        
        # Simulate analysis (in production, this would call Rust analyzer)
        results = await simulate_analysis(analysis_id, temp_path, file.filename)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis results by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM analyses WHERE id = ?",
        (analysis_id,)
    )
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    results = json.loads(row[4]) if row[4] else None
    
    return {
        "id": row[0],
        "filename": row[1],
        "upload_time": row[2],
        "status": row[3],
        "results": results,
        "error_message": row[5]
    }

@app.get("/api/analyses")
async def list_analyses(limit: int = 10):
    """List recent analyses"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, filename, upload_time, status FROM analyses ORDER BY upload_time DESC LIMIT ?",
        (limit,)
    )
    
    analyses = []
    for row in cursor.fetchall():
        analyses.append({
            "id": row[0],
            "filename": row[1],
            "upload_time": row[2],
            "status": row[3]
        })
    
    conn.close()
    return {"analyses": analyses, "count": len(analyses)}

async def simulate_analysis(analysis_id: str, filepath: str, filename: str) -> Dict:
    """Simulate genomic analysis (placeholder for actual implementation)"""
    # In production, this would:
    # 1. Call Rust analyzer binary
    # 2. Process results
    # 3. Store in database
    
    # For now, return simulated results
    results = {
        "organism": f"Sample from {filename}",
        "sequence_length": os.path.getsize(filepath),
        "genes_analyzed": 42,
        "pleiotropic_genes": 7,
        "average_confidence": 0.73,
        "traits": ["Carbon metabolism", "Stress response", "Regulatory"],
        "genes": [
            {"name": "gene1", "confidence": 0.85, "traits": ["Carbon metabolism", "Regulatory"]},
            {"name": "gene2", "confidence": 0.72, "traits": ["Stress response"]},
            {"name": "gene3", "confidence": 0.68, "traits": ["Carbon metabolism", "Stress response"]},
        ],
        "analysis_time": "2.3 seconds",
        "method": "NeuroDNA v0.0.2"
    }
    
    # Update database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE analyses SET status = ?, results = ? WHERE id = ?",
        ("completed", json.dumps(results), analysis_id)
    )
    
    # Store genes
    for gene in results["genes"]:
        cursor.execute(
            "INSERT INTO genes (analysis_id, gene_name, confidence, traits) VALUES (?, ?, ?, ?)",
            (analysis_id, gene["name"], gene["confidence"], json.dumps(gene["traits"]))
        )
    
    conn.commit()
    conn.close()
    
    return results

# Create systemd service file
def create_service_file():
    """Generate systemd service configuration"""
    service_content = """[Unit]
Description=Pleiotropy API Service
After=network.target

[Service]
Type=exec
User=webadmin
WorkingDirectory=/home/webadmin/pleiotropy
Environment="PATH=/home/webadmin/pleiotropy/venv/bin:/usr/bin"
ExecStart=/home/webadmin/pleiotropy/venv/bin/python /home/webadmin/pleiotropy/api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open(os.path.expanduser("~/pleiotropy/pleiotropy-api.service"), "w") as f:
        f.write(service_content)
    
    print("Service file created at ~/pleiotropy/pleiotropy-api.service")
    print("To install: sudo cp ~/pleiotropy/pleiotropy-api.service /etc/systemd/system/")
    print("Then: sudo systemctl enable pleiotropy-api && sudo systemctl start pleiotropy-api")

# Create Apache config snippet
def create_apache_config():
    """Generate Apache configuration for reverse proxy"""
    config_content = """# Pleiotropy API Reverse Proxy Configuration
# Add this to your Apache virtual host configuration

# Enable required modules:
# sudo a2enmod proxy proxy_http proxy_wstunnel rewrite

# API proxy
ProxyPass /api http://localhost:8080/api
ProxyPassReverse /api http://localhost:8080/api

# WebSocket support
RewriteEngine on
RewriteCond %{HTTP:Upgrade} websocket [NC]
RewriteCond %{HTTP:Connection} upgrade [NC]
RewriteRule ^/?(.*) "ws://localhost:8080/$1" [P,L]

# Optional: Serve static files from public_html
Alias /pleiotropy /home/webadmin/public_html/pleiotropy
<Directory /home/webadmin/public_html/pleiotropy>
    Options -Indexes +FollowSymLinks
    AllowOverride All
    Require all granted
</Directory>
"""
    
    with open(os.path.expanduser("~/pleiotropy/apache-pleiotropy.conf"), "w") as f:
        f.write(config_content)
    
    print("Apache config created at ~/pleiotropy/apache-pleiotropy.conf")

if __name__ == "__main__":
    # Create configuration files
    create_service_file()
    create_apache_config()
    
    # Run the server
    print("Starting Pleiotropy API Server on http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)