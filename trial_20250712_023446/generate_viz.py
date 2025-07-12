
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
