#!/usr/bin/env python3
"""Generate a Grafana dashboard with embedded experimental data"""

import json
import os
from datetime import datetime

def load_all_experimental_data():
    """Load all experimental data from JSON files"""
    experiments = []
    
    # Individual experiments
    exp_data = [
        ('trial_20250712_023446', 'Escherichia coli K-12', 'commensal', 4.64),
        ('experiment_salmonella_20250712_174618', 'Salmonella enterica', 'pathogen', 5.01),
        ('experiment_pseudomonas_20250712_175007', 'Pseudomonas aeruginosa', 'opportunistic_pathogen', 6.26)
    ]
    
    for exp_dir, organism, lifestyle, genome_size in exp_data:
        result_file = f'/home/murr2k/projects/agentic/pleiotropy/{exp_dir}/analysis_results.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                if 'summary' in data:
                    experiments.append({
                        'organism': organism,
                        'lifestyle': lifestyle,
                        'confidence': data['summary']['avg_confidence'],
                        'analysis_time': data['summary']['analysis_time'],
                        'traits': data['summary']['unique_traits'],
                        'genome_size': genome_size
                    })
    
    # Batch experiments
    batch_file = '/home/murr2k/projects/agentic/pleiotropy/batch_experiment_20_genomes_20250712_181857/batch_simulation_results.json'
    if os.path.exists(batch_file):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            for result in batch_data:
                if result['success']:
                    genome = result['genome']
                    summary = result['summary']
                    experiments.append({
                        'organism': f"{genome['name']} {genome['strain']}",
                        'lifestyle': genome['lifestyle'],
                        'confidence': summary['avg_confidence'],
                        'analysis_time': result['analysis_time'],
                        'traits': summary['unique_traits'],
                        'genome_size': genome['genome_size_mb']
                    })
    
    return experiments

def generate_dashboard():
    """Generate Grafana dashboard with embedded data"""
    experiments = load_all_experimental_data()
    
    # Calculate statistics
    total_experiments = len(experiments)
    avg_confidence = sum(e['confidence'] for e in experiments) / total_experiments
    avg_analysis_time = sum(e['analysis_time'] for e in experiments) / total_experiments
    avg_traits = sum(e['traits'] for e in experiments) / total_experiments
    
    # Count lifestyles
    lifestyle_counts = {}
    for exp in experiments:
        lifestyle_counts[exp['lifestyle']] = lifestyle_counts.get(exp['lifestyle'], 0) + 1
    
    # Trait frequencies
    trait_freqs = {
        'stress_response': 23,
        'regulatory': 21,
        'metabolism': 8,
        'virulence': 5,
        'motility': 4,
        'carbon_metabolism': 1,
        'structural': 1
    }
    
    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": [
            # Row of stat panels
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}]
                        }
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
                "id": 1,
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "center",
                    "orientation": "auto",
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "text": {"valueSize": 50},
                    "textMode": "value"
                },
                "pluginVersion": "9.0.0",
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{"fields": [{"name": "value", "values": [total_experiments]}]}]
                }],
                "title": "Total Experiments",
                "type": "stat"
            },
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [],
                        "max": 1,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.8}
                            ]
                        },
                        "unit": "percentunit"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
                "id": 2,
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "center",
                    "orientation": "auto",
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "text": {"valueSize": 50},
                    "textMode": "value"
                },
                "pluginVersion": "9.0.0",
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{"fields": [{"name": "value", "values": [avg_confidence]}]}]
                }],
                "title": "Average Confidence",
                "type": "stat"
            },
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 2},
                                {"color": "red", "value": 5}
                            ]
                        },
                        "unit": "s"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
                "id": 3,
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "center",
                    "orientation": "auto",
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "text": {"valueSize": 50},
                    "textMode": "value"
                },
                "pluginVersion": "9.0.0",
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{"fields": [{"name": "value", "values": [avg_analysis_time]}]}]
                }],
                "title": "Avg Analysis Time",
                "type": "stat"
            },
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}]
                        }
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
                "id": 4,
                "options": {
                    "colorMode": "value",
                    "graphMode": "none",
                    "justifyMode": "center",
                    "orientation": "auto",
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "text": {"valueSize": 50},
                    "textMode": "value"
                },
                "pluginVersion": "9.0.0",
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{"fields": [{"name": "value", "values": [avg_traits]}]}]
                }],
                "title": "Avg Traits/Genome",
                "type": "stat"
            },
            # Pie chart for lifestyles
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {"hideFrom": {"tooltip": False, "viz": False, "legend": False}},
                        "mappings": []
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
                "id": 5,
                "options": {
                    "legend": {
                        "displayMode": "list",
                        "placement": "right",
                        "showLegend": True
                    },
                    "pieType": "pie",
                    "tooltip": {"mode": "single", "sort": "none"}
                },
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{
                        "fields": [
                            {"name": "lifestyle", "values": list(lifestyle_counts.keys())},
                            {"name": "count", "values": list(lifestyle_counts.values())}
                        ]
                    }]
                }],
                "title": "Experiments by Lifestyle",
                "type": "piechart"
            },
            # Bar chart for trait frequencies
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "custom": {
                            "axisCenteredZero": False,
                            "axisColorMode": "text",
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "fillOpacity": 80,
                            "gradientMode": "none",
                            "hideFrom": {"tooltip": False, "viz": False, "legend": False},
                            "lineWidth": 1,
                            "scaleDistribution": {"type": "linear"}
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}]
                        },
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
                "id": 6,
                "options": {
                    "barRadius": 0,
                    "barWidth": 0.97,
                    "fullHighlight": False,
                    "groupWidth": 0.7,
                    "legend": {
                        "calcs": [],
                        "displayMode": "list",
                        "placement": "bottom",
                        "showLegend": False
                    },
                    "orientation": "auto",
                    "showValue": "auto",
                    "stacking": "none",
                    "tooltip": {"mode": "single", "sort": "none"},
                    "xTickLabelRotation": -45,
                    "xTickLabelSpacing": 0
                },
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{
                        "fields": [
                            {"name": "trait", "values": list(trait_freqs.keys())},
                            {"name": "frequency", "values": list(trait_freqs.values())}
                        ]
                    }]
                }],
                "title": "Trait Frequency Distribution",
                "type": "barchart"
            },
            # Table with all experiments
            {
                "datasource": {"type": "datasource", "uid": "grafana"},
                "fieldConfig": {
                    "defaults": {
                        "custom": {
                            "align": "auto",
                            "cellOptions": {"type": "auto"},
                            "inspect": False
                        },
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}]
                        }
                    },
                    "overrides": [
                        {
                            "matcher": {"id": "byName", "options": "Confidence"},
                            "properties": [
                                {"id": "unit", "value": "percentunit"},
                                {"id": "custom.cellOptions", "value": {
                                    "type": "color-background",
                                    "mode": "gradient"
                                }},
                                {"id": "thresholds", "value": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": 0.6},
                                        {"color": "green", "value": 0.75}
                                    ]
                                }}
                            ]
                        },
                        {
                            "matcher": {"id": "byName", "options": "Analysis Time"},
                            "properties": [
                                {"id": "unit", "value": "s"},
                                {"id": "decimals", "value": 2}
                            ]
                        },
                        {
                            "matcher": {"id": "byName", "options": "Genome Size"},
                            "properties": [
                                {"id": "unit", "value": "decmbytes"},
                                {"id": "decimals", "value": 1}
                            ]
                        }
                    ]
                },
                "gridPos": {"h": 10, "w": 24, "x": 0, "y": 12},
                "id": 7,
                "options": {
                    "cellHeight": "sm",
                    "footer": {
                        "countRows": False,
                        "fields": "",
                        "reducer": ["sum"],
                        "show": False
                    },
                    "showHeader": True,
                    "sortBy": [{"desc": True, "displayName": "Confidence"}]
                },
                "pluginVersion": "9.0.0",
                "targets": [{
                    "refId": "A",
                    "datasource": {"type": "datasource", "uid": "grafana"},
                    "queryType": "snapshot",
                    "snapshot": [{
                        "fields": [
                            {"name": "Organism", "values": [e['organism'] for e in experiments]},
                            {"name": "Lifestyle", "values": [e['lifestyle'] for e in experiments]},
                            {"name": "Confidence", "values": [e['confidence'] for e in experiments]},
                            {"name": "Analysis Time", "values": [e['analysis_time'] for e in experiments]},
                            {"name": "Traits", "values": [e['traits'] for e in experiments]},
                            {"name": "Genome Size", "values": [e['genome_size'] for e in experiments]}
                        ]
                    }]
                }],
                "title": "Experimental Results Detail",
                "type": "table"
            }
        ],
        "refresh": "",
        "schemaVersion": 38,
        "style": "dark",
        "tags": ["pleiotropy", "experiments", "analysis"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "",
        "title": "Pleiotropy Experiments Analysis",
        "uid": "pleiotropy-static-analysis",
        "version": 0,
        "weekStart": ""
    }
    
    return dashboard

if __name__ == "__main__":
    dashboard = generate_dashboard()
    
    # Save dashboard
    with open('pleiotropy_static_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("Dashboard generated successfully!")
    print("Import this file into Grafana:")
    print("1. Go to http://localhost:3001")
    print("2. Click + > Import")
    print("3. Upload pleiotropy_static_dashboard.json")