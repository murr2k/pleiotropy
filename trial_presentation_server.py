#!/usr/bin/env python3
"""
Trial Presentation Server - Interactive web interface for viewing pleiotropy analysis trials
"""

import json
import sqlite3
import os
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from pathlib import Path

# Configuration
DATABASE_PATH = "trial_database/database/trials.db"
PORT = 8888

class TrialPresentationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_index()
        elif self.path.startswith('/api/trials'):
            self.serve_trials_api()
        elif self.path.startswith('/api/results/'):
            trial_id = self.path.split('/')[-1]
            self.serve_results_api(trial_id)
        elif self.path.startswith('/trial/'):
            trial_id = self.path.split('/')[-1]
            self.serve_trial_details(trial_id)
        else:
            self.send_error(404)
    
    def serve_index(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genomic Pleiotropy Analysis - Trial Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1e3c72;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 1.2em;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .sort-controls {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        select, button {
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
        }
        select:hover, button:hover {
            border-color: #2a5298;
        }
        .trials-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        .trial-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }
        .trial-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .trial-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }
        .trial-id {
            background: #f0f0f0;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            color: #666;
        }
        .trial-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .status-running { background: #4CAF50; color: white; }
        .status-completed { background: #2196F3; color: white; }
        .status-failed { background: #f44336; color: white; }
        .status-pending { background: #ff9800; color: white; }
        .trial-name {
            font-size: 1.3em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .trial-details {
            color: #666;
            line-height: 1.6;
        }
        .trial-metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .metric-label {
            font-weight: 500;
        }
        .metric-value {
            color: #2a5298;
            font-weight: bold;
        }
        .trial-description {
            margin-top: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
            font-style: italic;
        }
        .view-results-btn {
            margin-top: 15px;
            width: 100%;
            background: #2a5298;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 5px;
            font-weight: bold;
            transition: background 0.3s;
        }
        .view-results-btn:hover {
            background: #1e3c72;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: white;
            font-size: 1.2em;
        }
        .stats-summary {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2a5298;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Genomic Pleiotropy Analysis</h1>
            <p class="subtitle">Cryptanalytic Approach to Multi-Trait Gene Detection</p>
        </div>
        
        <div class="stats-summary" id="stats-summary">
            <div class="stat-box">
                <div class="stat-value" id="total-trials">-</div>
                <div class="stat-label">Total Trials</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="genes-analyzed">-</div>
                <div class="stat-label">Genes Analyzed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="avg-confidence">-</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="success-rate">-</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="sort-controls">
                <label>Sort by:</label>
                <select id="sort-field">
                    <option value="created_at">Date Created</option>
                    <option value="name">Trial Name</option>
                    <option value="status">Status</option>
                    <option value="id">Trial ID</option>
                </select>
                <select id="sort-order">
                    <option value="desc">Descending</option>
                    <option value="asc">Ascending</option>
                </select>
                <button onclick="loadTrials()">Refresh</button>
            </div>
        </div>
        
        <div id="trials-container" class="trials-grid">
            <div class="loading">Loading trials...</div>
        </div>
    </div>
    
    <script>
        let trials = [];
        
        async function loadTrials() {
            try {
                const response = await fetch('/api/trials');
                const data = await response.json();
                trials = data.trials;
                updateStats(data.stats);
                displayTrials();
            } catch (error) {
                console.error('Error loading trials:', error);
                document.getElementById('trials-container').innerHTML = 
                    '<div class="loading">Error loading trials. Please try again.</div>';
            }
        }
        
        function updateStats(stats) {
            document.getElementById('total-trials').textContent = stats.total_trials || 0;
            document.getElementById('genes-analyzed').textContent = stats.total_genes || 0;
            document.getElementById('avg-confidence').textContent = 
                (stats.avg_confidence || 0).toFixed(2) + '%';
            document.getElementById('success-rate').textContent = 
                (stats.success_rate || 0).toFixed(1) + '%';
        }
        
        function displayTrials() {
            const container = document.getElementById('trials-container');
            const sortField = document.getElementById('sort-field').value;
            const sortOrder = document.getElementById('sort-order').value;
            
            // Sort trials
            const sortedTrials = [...trials].sort((a, b) => {
                let aVal = a[sortField];
                let bVal = b[sortField];
                
                if (sortField === 'created_at') {
                    aVal = new Date(aVal);
                    bVal = new Date(bVal);
                }
                
                if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
                if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
                return 0;
            });
            
            container.innerHTML = sortedTrials.map(trial => createTrialCard(trial)).join('');
        }
        
        function createTrialCard(trial) {
            const params = JSON.parse(trial.parameters || '{}');
            const statusClass = `status-${trial.status.toLowerCase()}`;
            const createdDate = new Date(trial.created_at).toLocaleDateString();
            
            return `
                <div class="trial-card" onclick="viewTrialDetails(${trial.id})">
                    <div class="trial-header">
                        <span class="trial-id">Trial #${trial.id}</span>
                        <span class="trial-status ${statusClass}">${trial.status}</span>
                    </div>
                    <h3 class="trial-name">${trial.name}</h3>
                    <div class="trial-details">
                        <div class="trial-metric">
                            <span class="metric-label">Created:</span>
                            <span class="metric-value">${createdDate}</span>
                        </div>
                        <div class="trial-metric">
                            <span class="metric-label">Method:</span>
                            <span class="metric-value">${params.method || 'Cryptanalytic'}</span>
                        </div>
                        <div class="trial-metric">
                            <span class="metric-label">Target Genes:</span>
                            <span class="metric-value">${params.target_genes?.length || 0}</span>
                        </div>
                        <div class="trial-metric">
                            <span class="metric-label">Confidence:</span>
                            <span class="metric-value">${(params.confidence_threshold || 0.7) * 100}%</span>
                        </div>
                    </div>
                    ${trial.description ? 
                        `<div class="trial-description">${trial.description}</div>` : ''}
                    <button class="view-results-btn" onclick="event.stopPropagation(); viewTrialDetails(${trial.id})">
                        View Results →
                    </button>
                </div>
            `;
        }
        
        function viewTrialDetails(trialId) {
            window.location.href = `/trial/${trialId}`;
        }
        
        // Load trials on page load
        window.addEventListener('load', loadTrials);
        
        // Auto-refresh every 30 seconds
        setInterval(loadTrials, 30000);
    </script>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_trials_api(self):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all trials
            cursor.execute("""
                SELECT * FROM trials 
                ORDER BY created_at DESC
            """)
            trials = [dict(row) for row in cursor.fetchall()]
            
            # Calculate statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT t.id) as total_trials,
                    COUNT(DISTINCT r.id) as total_results,
                    AVG(CAST(json_extract(r.metrics, '$.confidence_score') AS REAL)) * 100 as avg_confidence
                FROM trials t
                LEFT JOIN results r ON t.id = r.trial_id
            """)
            stats_row = cursor.fetchone()
            
            stats = {
                'total_trials': stats_row['total_trials'] or 0,
                'total_results': stats_row['total_results'] or 0,
                'avg_confidence': stats_row['avg_confidence'] or 0,
                'total_genes': 5,  # Known from our test data
                'success_rate': 0  # Known from validation results
            }
            
            conn.close()
            
            response = {
                'trials': trials,
                'stats': stats
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def serve_results_api(self, trial_id):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get trial details
            cursor.execute("SELECT * FROM trials WHERE id = ?", (trial_id,))
            trial = dict(cursor.fetchone()) if cursor.fetchone() else None
            
            # Get results for this trial
            cursor.execute("""
                SELECT r.*, a.name as agent_name 
                FROM results r
                JOIN agents a ON r.agent_id = a.id
                WHERE r.trial_id = ?
                ORDER BY r.timestamp DESC
            """, (trial_id,))
            results = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            response = {
                'trial': trial,
                'results': results
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def serve_trial_details(self, trial_id):
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trial #{trial_id} - Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .back-link {{
            color: white;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            transition: background 0.3s;
        }}
        .back-link:hover {{
            background: rgba(255,255,255,0.2);
        }}
        .trial-header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1e3c72;
            margin-bottom: 20px;
        }}
        .trial-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .info-box {{
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .info-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .info-value {{
            font-weight: bold;
            color: #2a5298;
            font-size: 1.1em;
        }}
        .results-section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        .result-card {{
            border: 1px solid #eee;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            transition: border-color 0.3s;
        }}
        .result-card:hover {{
            border-color: #2a5298;
        }}
        .result-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }}
        .agent-name {{
            font-weight: bold;
            color: #1e3c72;
        }}
        .confidence-score {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .confidence-high {{ background: #4CAF50; }}
        .confidence-medium {{ background: #ff9800; }}
        .confidence-low {{ background: #f44336; }}
        .result-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric {{
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .metric-name {{
            color: #666;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .no-results {{
            text-align: center;
            padding: 50px;
            color: #999;
        }}
        .visualization-section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        .viz-container {{
            margin-top: 20px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-box {{
            background: #f0f7ff;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .summary-title {{
            font-weight: bold;
            color: #1976D2;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back to All Trials</a>
        
        <div class="trial-header">
            <h1 id="trial-title">Loading Trial...</h1>
            <div class="trial-info" id="trial-info"></div>
        </div>
        
        <div class="visualization-section">
            <h2>Analysis Summary</h2>
            <div class="summary-box">
                <div class="summary-title">Key Findings</div>
                <div id="key-findings">Loading analysis summary...</div>
            </div>
            <div class="viz-container">
                <p>Visualization dashboards and detailed reports available in the project directory.</p>
                <p style="margin-top: 10px;">
                    <strong>Dashboard:</strong> <code>memory/swarm-pleiotropy-analysis-*/visualization-generator/index.html</code>
                </p>
            </div>
        </div>
        
        <div class="results-section">
            <h2>Analysis Results</h2>
            <div id="results-container">
                <div class="no-results">Loading results...</div>
            </div>
        </div>
    </div>
    
    <script>
        const trialId = {trial_id};
        
        async function loadTrialDetails() {{
            try {{
                const response = await fetch(`/api/results/${{trialId}}`);
                const data = await response.json();
                
                if (data.trial) {{
                    displayTrialInfo(data.trial);
                }}
                
                if (data.results && data.results.length > 0) {{
                    displayResults(data.results);
                }} else {{
                    document.getElementById('results-container').innerHTML = 
                        '<div class="no-results">No results found for this trial.</div>';
                }}
                
                // Display summary based on known validation results
                displaySummary();
                
            }} catch (error) {{
                console.error('Error loading trial details:', error);
                document.getElementById('results-container').innerHTML = 
                    '<div class="no-results">Error loading results. Please try again.</div>';
            }}
        }}
        
        function displayTrialInfo(trial) {{
            document.getElementById('trial-title').textContent = trial.name || `Trial #${{trial.id}}`;
            
            const params = JSON.parse(trial.parameters || '{{}}');
            const createdDate = new Date(trial.created_at).toLocaleString();
            const updatedDate = new Date(trial.updated_at).toLocaleString();
            
            const infoHtml = `
                <div class="info-box">
                    <div class="info-label">Status</div>
                    <div class="info-value">${{trial.status}}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Created</div>
                    <div class="info-value">${{createdDate}}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Method</div>
                    <div class="info-value">${{params.method || 'Cryptanalytic'}}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Target Genes</div>
                    <div class="info-value">${{params.target_genes?.join(', ') || 'N/A'}}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Confidence Threshold</div>
                    <div class="info-value">${{(params.confidence_threshold || 0.7) * 100}}%</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Input File</div>
                    <div class="info-value">${{params.input_file || 'test_ecoli_sample.fasta'}}</div>
                </div>
            `;
            
            document.getElementById('trial-info').innerHTML = infoHtml;
        }}
        
        function displayResults(results) {{
            const resultsHtml = results.map(result => {{
                const metrics = JSON.parse(result.metrics || '{{}}');
                const confidence = (metrics.confidence_score || 0) * 100;
                const confidenceClass = confidence >= 80 ? 'confidence-high' : 
                                       confidence >= 60 ? 'confidence-medium' : 'confidence-low';
                
                return `
                    <div class="result-card">
                        <div class="result-header">
                            <span class="agent-name">${{result.agent_name || 'Unknown Agent'}}</span>
                            <span class="confidence-score ${{confidenceClass}}">
                                ${{confidence.toFixed(1)}}% Confidence
                            </span>
                        </div>
                        <div class="result-metrics">
                            <div class="metric">
                                <div class="metric-name">Genes Analyzed</div>
                                <div class="metric-value">${{metrics.genes_analyzed || 0}}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Traits Detected</div>
                                <div class="metric-value">${{metrics.traits_detected || 0}}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Execution Time</div>
                                <div class="metric-value">${{metrics.execution_time || 'N/A'}}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Data Points</div>
                                <div class="metric-value">${{metrics.data_points || 0}}</div>
                            </div>
                        </div>
                    </div>
                `;
            }}).join('');
            
            document.getElementById('results-container').innerHTML = resultsHtml;
        }}
        
        function displaySummary() {{
            // Display known validation results
            const summaryHtml = `
                <p><strong>Analysis Type:</strong> E. coli K-12 Pleiotropic Gene Detection</p>
                <p><strong>Result:</strong> Technical Success, Biological Failure</p>
                <ul style="margin-top: 10px; margin-left: 20px;">
                    <li>✅ Cryptanalysis engine executed successfully</li>
                    <li>✅ Statistical framework functioning correctly</li>
                    <li>✅ Visualization dashboards generated</li>
                    <li>❌ 0/5 known pleiotropic genes detected</li>
                    <li>❌ No trait associations identified</li>
                </ul>
                <p style="margin-top: 15px;"><strong>Conclusion:</strong> 
                The cryptanalytic approach demonstrated strong technical capabilities but failed to achieve 
                biological relevance. Algorithm requires integration of biological constraints.</p>
            `;
            
            document.getElementById('key-findings').innerHTML = summaryHtml;
        }}
        
        // Load trial details on page load
        window.addEventListener('load', loadTrialDetails);
    </script>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server():
    print(f"\n🌐 Starting Trial Presentation Server on http://localhost:{PORT}")
    print(f"   Access the presentation at: http://localhost:{PORT}/")
    print(f"   Press Ctrl+C to stop the server\n")
    
    server = HTTPServer(('localhost', PORT), TrialPresentationHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.shutdown()

if __name__ == '__main__':
    # Check if database exists
    if not os.path.exists(DATABASE_PATH):
        print(f"Error: Database not found at {DATABASE_PATH}")
        print("Please ensure the trial database exists before running the server.")
        exit(1)
    
    run_server()
