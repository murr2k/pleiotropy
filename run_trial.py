#!/usr/bin/env python3
"""
Run a comprehensive pleiotropy analysis trial
"""

import json
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

# Create trial directory
trial_id = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
trial_dir = Path(trial_id)
trial_dir.mkdir(exist_ok=True)

print(f"🧬 Starting Pleiotropy Analysis Trial: {trial_id}")
print("=" * 60)

# Trial configuration
trial_config = {
    "trial_id": trial_id,
    "timestamp": datetime.now().isoformat(),
    "analyses": [],
    "summary": {}
}

# 1. Create enhanced synthetic test data
print("\n📊 Phase 1: Creating Synthetic Test Data")
print("-" * 40)

synthetic_genome = """
>synthetic_pleiotropic_gene_1 Multi-trait gene with strong carbon/stress bias
ATGCTGCTGCTGGAAGAAGAAGAAAAAGAAAAAGAAAAAATGGAAGAAGAAGAACTGCTGCTGGAT
CGTCGTCGTGAAGAAGAAGAAAAAAAAACGTAAACGTCTGCTGCTGGAAGTTTTTTTTTCCC
AAACGTAAACGTATGCTGCTGCTGGAAGAAGAAGAACTGCTGCTGGATCGTCGTCGTAAACGTAAA
CGTTAAATGCTGCTGCTGTTCTTCTTCTTCGAAGAAGAAAAAATGAAACGTAAACGTAAACGTTAG

>synthetic_pleiotropic_gene_2 Regulatory and motility traits
ATGCGTCGTCGTAACAACAACTTCTTCTTCGGTGGTGGTCAACAACAACACGTCGTCGTTTCTTC
TTCCGTCGTCGTGGTGGTGGTAACAACAACCAACAACAATTCTTCTTCCGTCGTCGTAACAACAAC
GGTGGTGGTCGTCGTCGTTTCTTCTTCAACAACAACGGTGGTGGTCAACAACAACCGTCGTCGT
AACAACAACTTCTTCTTCGGTGGTGGTCGTCGTCGTAACAACAACTTCTTCTTCTAG

>synthetic_pleiotropic_gene_3 DNA processing and structural
ATGAACAACAACGCTGCTGCTTCTTCTTCTACCACCACCAACAACAACGCTGCTGCTTCTTCTTCT
GCTGCTGCTACCACCACCAACAACAACTCTTCTTCTGCTGCTGCTACCACCACCGCTGCTGCT
AACAACAACTCTTCTTCTACCACCACCGCTGCTGCTAACAACAACTCTTCTTCTGCTGCTGCT
TCTTCTTCTACCACCACCAACAACAACGCTGCTGCTTAA

>control_non_pleiotropic Random sequence with minimal bias
ATGGCATCGGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
GCATCGGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGTAG
"""

synthetic_fasta = trial_dir / "synthetic_test.fasta"
with open(synthetic_fasta, 'w') as f:
    f.write(synthetic_genome.strip())
print(f"✅ Created synthetic genome: {synthetic_fasta}")

# Enhanced trait definitions
trait_definitions = {
    "traits": [
        {
            "name": "carbon_metabolism",
            "description": "Carbon source utilization and metabolism",
            "associated_genes": ["crp", "cyaA", "aceE", "aceF"],
            "codon_patterns": ["CTG", "GAA", "AAA", "CGT"],
            "known_sequences": []
        },
        {
            "name": "stress_response", 
            "description": "Response to environmental stress conditions",
            "associated_genes": ["rpoS", "hns", "dnaK", "groEL"],
            "codon_patterns": ["GAA", "GCT", "ATT", "CTG"],
            "known_sequences": []
        },
        {
            "name": "regulatory",
            "description": "Gene expression regulation and control",
            "associated_genes": ["crp", "fis", "ihfA", "fnr"],
            "codon_patterns": ["CGT", "AAC", "TTC", "GAA"],
            "known_sequences": []
        },
        {
            "name": "motility",
            "description": "Flagellar synthesis and chemotaxis",
            "associated_genes": ["fliA", "flhDC", "cheA", "cheB"],
            "codon_patterns": ["TTC", "GGT", "CAA", "CGT"],
            "known_sequences": []
        },
        {
            "name": "dna_processing",
            "description": "DNA replication, repair, and recombination",
            "associated_genes": ["recA", "mutS", "dnaE", "polA"],
            "codon_patterns": ["AAC", "GCT", "TCT", "ACC"],
            "known_sequences": []
        },
        {
            "name": "structural",
            "description": "Structural proteins and cell components",
            "associated_genes": ["ompA", "ftsZ", "murA", "lpxA"],
            "codon_patterns": ["GCT", "ACC", "AAC", "TCT"],
            "known_sequences": []
        }
    ]
}

traits_file = trial_dir / "trait_definitions.json"
with open(traits_file, 'w') as f:
    json.dump(trait_definitions, f, indent=2)
print(f"✅ Created trait definitions: {traits_file}")

# 2. Run analysis on synthetic data
print("\n🔬 Phase 2: Analyzing Synthetic Data")
print("-" * 40)

output_dir = trial_dir / "synthetic_results"
output_dir.mkdir(exist_ok=True)

cmd = [
    "~/.cargo/bin/cargo", "run", "--manifest-path", "rust_impl/Cargo.toml", "--",
    "--input", str(synthetic_fasta),
    "--traits", str(traits_file),
    "--output", str(output_dir),
    "--min-traits", "2",
    "--verbose"
]

print(f"Running: {' '.join(cmd)}")
start_time = time.time()

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    analysis_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        
        # Read results
        results_file = output_dir / "pleiotropic_genes.json"
        if results_file.exists():
            with open(results_file) as f:
                synthetic_results = json.load(f)
            print(f"📊 Found {len(synthetic_results)} pleiotropic genes in synthetic data")
            
            trial_config["analyses"].append({
                "type": "synthetic",
                "file": str(synthetic_fasta),
                "time": analysis_time,
                "genes_found": len(synthetic_results),
                "results": synthetic_results
            })
    else:
        print(f"❌ Analysis failed: {result.stderr}")
except Exception as e:
    print(f"❌ Error running analysis: {e}")

# 3. Run analysis on E. coli genome
print("\n🧬 Phase 3: Analyzing E. coli Genome")
print("-" * 40)

ecoli_genome = Path("genome_research/public_ecoli_genome.fasta")
if ecoli_genome.exists():
    ecoli_output = trial_dir / "ecoli_results"
    ecoli_output.mkdir(exist_ok=True)
    
    cmd = [
        "~/.cargo/bin/cargo", "run", "--manifest-path", "rust_impl/Cargo.toml", "--",
        "--input", str(ecoli_genome),
        "--traits", str(traits_file),
        "--output", str(ecoli_output),
        "--min-traits", "2"
    ]
    
    print(f"Analyzing full E. coli genome...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        analysis_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ E. coli analysis completed in {analysis_time:.2f} seconds")
            
            # Read results
            results_file = ecoli_output / "pleiotropic_genes.json"
            if results_file.exists():
                with open(results_file) as f:
                    ecoli_results = json.load(f)
                print(f"📊 Found {len(ecoli_results)} pleiotropic regions in E. coli")
                
                trial_config["analyses"].append({
                    "type": "ecoli",
                    "file": str(ecoli_genome),
                    "time": analysis_time,
                    "genes_found": len(ecoli_results),
                    "results": ecoli_results[:5]  # First 5 for summary
                })
    except Exception as e:
        print(f"❌ Error analyzing E. coli: {e}")
else:
    print("⚠️  E. coli genome not found, skipping")

# 4. Generate visualizations
print("\n📈 Phase 4: Generating Visualizations")
print("-" * 40)

try:
    # Create visualization script
    viz_script = trial_dir / "generate_viz.py"
    with open(viz_script, 'w') as f:
        f.write('''
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
trial_dir = Path(".")
results = []

for result_dir in ["synthetic_results", "ecoli_results"]:
    genes_file = trial_dir / result_dir / "pleiotropic_genes.json"
    if genes_file.exists():
        with open(genes_file) as f:
            results.extend(json.load(f))

if results:
    # Create confidence distribution plot
    confidences = [gene["confidence"] for gene in results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Genes")
    plt.title("Distribution of Pleiotropic Gene Confidence Scores")
    plt.savefig("confidence_distribution.png", dpi=150, bbox_inches='tight')
    print("✅ Created confidence distribution plot")
    
    # Create trait frequency plot
    trait_counts = {}
    for gene in results:
        for trait in gene["traits"]:
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
    
    if trait_counts:
        plt.figure(figsize=(10, 6))
        traits = list(trait_counts.keys())
        counts = list(trait_counts.values())
        plt.bar(traits, counts, color='skyblue', edgecolor='navy')
        plt.xlabel("Trait")
        plt.ylabel("Number of Genes")
        plt.title("Trait Frequency in Pleiotropic Genes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("trait_frequency.png", dpi=150, bbox_inches='tight')
        print("✅ Created trait frequency plot")
''')
    
    # Run visualization
    subprocess.run(["python", str(viz_script)], cwd=trial_dir)
    print("✅ Visualizations generated")
    
except Exception as e:
    print(f"⚠️  Visualization generation skipped: {e}")

# 5. Generate comprehensive report
print("\n📄 Phase 5: Generating Trial Report")
print("-" * 40)

# Calculate summary statistics
total_genes = sum(analysis["genes_found"] for analysis in trial_config["analyses"])
total_time = sum(analysis["time"] for analysis in trial_config["analyses"])

trial_config["summary"] = {
    "total_analyses": len(trial_config["analyses"]),
    "total_genes_found": total_genes,
    "total_analysis_time": total_time,
    "average_confidence": 0.0
}

# Calculate average confidence
all_confidences = []
for analysis in trial_config["analyses"]:
    if "results" in analysis:
        all_confidences.extend([g["confidence"] for g in analysis["results"]])

if all_confidences:
    trial_config["summary"]["average_confidence"] = sum(all_confidences) / len(all_confidences)

# Save trial configuration
config_file = trial_dir / "trial_config.json"
with open(config_file, 'w') as f:
    json.dump(trial_config, f, indent=2)

# Generate markdown report
report = f"""# Pleiotropy Analysis Trial Report

**Trial ID**: {trial_id}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Analyses**: {trial_config['summary']['total_analyses']}
- **Total Pleiotropic Genes Found**: {trial_config['summary']['total_genes_found']}
- **Total Analysis Time**: {trial_config['summary']['total_analysis_time']:.2f} seconds
- **Average Confidence**: {trial_config['summary']['average_confidence']:.3f}

## Analysis Details

"""

for i, analysis in enumerate(trial_config["analyses"], 1):
    report += f"""
### Analysis {i}: {analysis['type'].title()}

- **File**: `{Path(analysis['file']).name}`
- **Genes Found**: {analysis['genes_found']}
- **Analysis Time**: {analysis['time']:.2f} seconds
"""
    
    if analysis.get("results"):
        report += "\n**Top Results**:\n"
        for j, gene in enumerate(analysis["results"][:3], 1):
            report += f"\n{j}. **{gene['gene_id']}**\n"
            report += f"   - Traits: {', '.join(gene['traits'])}\n"
            report += f"   - Confidence: {gene['confidence']:.3f}\n"

report += """
## Visualizations

- `confidence_distribution.png` - Distribution of confidence scores
- `trait_frequency.png` - Frequency of traits across genes

## Files Generated

- `synthetic_test.fasta` - Synthetic test genome
- `trait_definitions.json` - Trait pattern definitions
- `synthetic_results/` - Synthetic analysis results
- `ecoli_results/` - E. coli analysis results (if available)
- `trial_config.json` - Complete trial configuration

## System Information

- **Platform**: Linux/Docker
- **Analysis Engine**: NeuroDNA v0.0.2
- **Confidence Threshold**: 0.4
- **Minimum Traits**: 2

---

*Generated by Genomic Pleiotropy Cryptanalysis System*
"""

report_file = trial_dir / "trial_report.md"
with open(report_file, 'w') as f:
    f.write(report)

print(f"✅ Trial report saved: {report_file}")

# Final summary
print("\n" + "=" * 60)
print(f"🎉 Trial {trial_id} Complete!")
print(f"📊 Found {total_genes} pleiotropic genes in {total_time:.2f} seconds")
print(f"📁 Results saved in: {trial_dir}/")
print("=" * 60)