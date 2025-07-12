#!/usr/bin/env python3
"""
Create test data for UI testing
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
import random

def create_test_data():
    """Create test data in the database"""
    
    # Connect to the database
    db_path = "/home/murr2k/projects/agentic/pleiotropy/trial_database/database/trials.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Existing tables: {[t[0] for t in tables]}")
    
    # If no tables, create basic ones for testing
    if not tables:
        # Create agents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            capabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create trials table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            organism TEXT,
            genome_file TEXT,
            parameters TEXT,
            status TEXT NOT NULL,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            progress INTEGER DEFAULT 0,
            FOREIGN KEY(created_by) REFERENCES agents(id)
        )
        """)
        
        # Create results table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trial_results (
            id TEXT PRIMARY KEY,
            trial_id TEXT NOT NULL,
            sequence_region TEXT,
            codon_frequencies TEXT,
            trait_predictions TEXT,
            confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(trial_id) REFERENCES trials(id)
        )
        """)
        
        conn.commit()
        print("Created database schema")
    
    # Create test agents
    test_agents = [
        {
            'name': 'Rust Analyzer Agent',
            'role': 'rust_analyzer',
            'hashed_password': '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewFyLawz3iL.2yy.',  # password: test123
            'capabilities': json.dumps(['crypto_analysis', 'sequence_parsing', 'frequency_analysis']),
            'active': True
        },
        {
            'name': 'Python Visualizer Agent',
            'role': 'python_visualizer',
            'hashed_password': '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewFyLawz3iL.2yy.',  # password: test123
            'capabilities': json.dumps(['data_visualization', 'statistical_analysis', 'report_generation']),
            'active': True
        }
    ]
    
    agent_ids = []
    for agent in test_agents:
        cursor.execute("""
        INSERT INTO agents (name, role, hashed_password, capabilities, active)
        VALUES (?, ?, ?, ?, ?)
        """, (agent['name'], agent['role'], agent['hashed_password'], agent['capabilities'], agent['active']))
        agent_ids.append(cursor.lastrowid)
    
    print(f"Created {len(test_agents)} test agents")
    
    # Create test trials
    trial_statuses = ['pending', 'running', 'completed', 'failed']
    organisms = ['E. coli K-12', 'S. cerevisiae', 'D. melanogaster', 'H. sapiens']
    
    test_trials = []
    trial_ids = []
    for i in range(10):
        created_date = datetime.now() - timedelta(days=random.randint(0, 30))
        status = random.choice(trial_statuses)
        organism = random.choice(organisms)
        
        trial = {
            'name': f'Cryptanalysis Trial {i+1}: {organism}',
            'description': f'Genomic pleiotropy analysis for {organism} using cryptographic pattern detection',
            'organism': organism,
            'genome_file': f'{organism.lower().replace(" ", "_").replace(".", "")}.fasta',
            'parameters': json.dumps({
                'window_size': random.choice([500, 1000, 1500, 2000]),
                'overlap': random.choice([50, 100, 150, 200]),
                'min_confidence': random.uniform(0.7, 0.95),
                'codon_table': random.choice([1, 2, 3, 4])
            }),
            'status': status,
            'created_by': random.choice(agent_ids),
            'created_at': created_date.isoformat(),
            'updated_at': (created_date + timedelta(hours=random.randint(1, 72))).isoformat(),
        }
        
        if status in ['running', 'completed']:
            trial['started_at'] = (created_date + timedelta(minutes=random.randint(5, 60))).isoformat()
        
        if status == 'completed':
            trial['completed_at'] = (created_date + timedelta(hours=random.randint(2, 48))).isoformat()
        
        test_trials.append(trial)
    
    for trial in test_trials:
        cursor.execute("""
        INSERT INTO trials 
        (name, description, organism, genome_file, parameters, status, created_by, 
         created_at, updated_at, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial['name'], trial['description'], trial['organism'],
            trial['genome_file'], trial['parameters'], trial['status'], trial['created_by'],
            trial['created_at'], trial['updated_at'], 
            trial.get('started_at'), trial.get('completed_at')
        ))
        trial_ids.append(cursor.lastrowid)
    
    print(f"Created {len(test_trials)} test trials")
    
    # Create test results for completed trials
    completed_trial_indices = [i for i, t in enumerate(test_trials) if t['status'] == 'completed']
    test_results = []
    
    for trial_idx in completed_trial_indices:
        trial_id = trial_ids[trial_idx]
        # Create 5-15 results per completed trial
        num_results = random.randint(5, 15)
        for j in range(num_results):
            result = {
                'trial_id': trial_id,
                'gene_id': f'gene_{random.randint(1000, 9999)}',
                'traits': json.dumps([
                    'metabolic_efficiency',
                    'stress_resistance', 
                    'growth_rate'
                ]),
                'confidence_scores': json.dumps({
                    'metabolic_efficiency': random.uniform(0.7, 0.95),
                    'stress_resistance': random.uniform(0.6, 0.9),
                    'growth_rate': random.uniform(0.5, 0.85)
                }),
                'codon_usage_bias': json.dumps({
                    'TTT': random.uniform(0.01, 0.05),
                    'TTC': random.uniform(0.01, 0.05),
                    'TTA': random.uniform(0.01, 0.03),
                    'TTG': random.uniform(0.01, 0.04),
                    'AGT': random.uniform(0.01, 0.03),
                    'AGC': random.uniform(0.01, 0.04)
                }),
                'regulatory_context': json.dumps({
                    'promoter_strength': random.uniform(0.3, 1.0),
                    'enhancer_elements': random.randint(0, 5),
                    'silencer_elements': random.randint(0, 2)
                }),
                'validated': random.choice([True, False]),
                'validated_by': random.choice(agent_ids) if random.choice([True, False]) else None
            }
            test_results.append(result)
    
    for result in test_results:
        cursor.execute("""
        INSERT INTO results 
        (trial_id, gene_id, traits, confidence_scores, codon_usage_bias, 
         regulatory_context, validated, validated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result['trial_id'], result['gene_id'], result['traits'],
            result['confidence_scores'], result['codon_usage_bias'], 
            result['regulatory_context'], result['validated'], result['validated_by']
        ))
    
    # Create progress entries for running and completed trials
    progress_entries = []
    for trial_idx, trial in enumerate(test_trials):
        if trial['status'] in ['running', 'completed']:
            trial_id = trial_ids[trial_idx]
            stages = ['initialization', 'sequence_parsing', 'frequency_analysis', 'trait_prediction', 'validation']
            
            if trial['status'] == 'completed':
                # All stages completed
                for stage_idx, stage in enumerate(stages):
                    progress = {
                        'trial_id': trial_id,
                        'stage': stage,
                        'progress_percentage': 100.0,
                        'current_task': f'Completed {stage}',
                        'genes_processed': random.randint(500, 2000),
                        'total_genes': random.randint(500, 2000),
                        'error_count': random.randint(0, 5)
                    }
                    progress_entries.append(progress)
            else:
                # Running trial - some stages completed
                current_stage = random.randint(1, len(stages) - 1)
                for stage_idx in range(current_stage):
                    stage = stages[stage_idx]
                    progress = {
                        'trial_id': trial_id,
                        'stage': stage,
                        'progress_percentage': 100.0,
                        'current_task': f'Completed {stage}',
                        'genes_processed': random.randint(100, 500),
                        'total_genes': random.randint(1000, 2000),
                        'error_count': random.randint(0, 3)
                    }
                    progress_entries.append(progress)
                
                # Current stage in progress
                current_progress = random.uniform(10, 80)
                progress = {
                    'trial_id': trial_id,
                    'stage': stages[current_stage],
                    'progress_percentage': current_progress,
                    'current_task': f'Processing {stages[current_stage]}',
                    'genes_processed': int(random.randint(500, 1500) * current_progress / 100),
                    'total_genes': random.randint(1500, 2000),
                    'error_count': random.randint(0, 2)
                }
                progress_entries.append(progress)
    
    for progress in progress_entries:
        cursor.execute("""
        INSERT INTO progress 
        (trial_id, stage, progress_percentage, current_task, genes_processed, 
         total_genes, error_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            progress['trial_id'], progress['stage'], progress['progress_percentage'],
            progress['current_task'], progress['genes_processed'], 
            progress['total_genes'], progress['error_count']
        ))
    
    print(f"Created {len(test_results)} test results")
    print(f"Created {len(progress_entries)} progress entries")
    
    conn.commit()
    conn.close()
    
    print("\n✅ Test data creation completed!")
    print(f"   - {len(test_agents)} agents")
    print(f"   - {len(test_trials)} trials")
    print(f"   - {len(test_results)} results")
    print(f"   - {len(progress_entries)} progress entries")

if __name__ == "__main__":
    create_test_data()