#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load all results
all_results = []
for result_dir in ["synthetic_results", "ecoli_results"]:
    genes_file = Path(result_dir) / "pleiotropic_genes.json"
    if genes_file.exists():
        with open(genes_file) as f:
            data = json.load(f)
            for gene in data:
                gene['source'] = 'Synthetic' if 'synthetic' in result_dir else 'E. coli'
            all_results.extend(data)

print(f"Loaded {len(all_results)} pleiotropic genes")

# 1. Confidence Score Distribution
plt.figure(figsize=(10, 6))
confidences = [gene["confidence"] for gene in all_results]
sources = [gene["source"] for gene in all_results]

# Create histogram with source separation
synthetic_conf = [g["confidence"] for g in all_results if g["source"] == "Synthetic"]
ecoli_conf = [g["confidence"] for g in all_results if g["source"] == "E. coli"]

plt.hist([synthetic_conf, ecoli_conf], bins=15, label=['Synthetic', 'E. coli'], 
         color=['skyblue', 'lightcoral'], edgecolor='black', alpha=0.7)
plt.xlabel("Confidence Score")
plt.ylabel("Number of Genes")
plt.title("Distribution of Pleiotropic Gene Confidence Scores")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("confidence_distribution.png", dpi=150, bbox_inches='tight')
print("✅ Created confidence distribution plot")

# 2. Trait Frequency Analysis
plt.figure(figsize=(12, 6))
trait_counts = {}
for gene in all_results:
    for trait in gene["traits"]:
        trait_counts[trait] = trait_counts.get(trait, 0) + 1

traits = list(trait_counts.keys())
counts = list(trait_counts.values())

# Sort by frequency
sorted_items = sorted(zip(traits, counts), key=lambda x: x[1], reverse=True)
traits, counts = zip(*sorted_items)

bars = plt.bar(traits, counts, color='darkblue', alpha=0.7, edgecolor='black')
plt.xlabel("Trait")
plt.ylabel("Number of Genes")
plt.title("Trait Frequency in Pleiotropic Genes")
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("trait_frequency.png", dpi=150, bbox_inches='tight')
print("✅ Created trait frequency plot")

# 3. Trait Co-occurrence Heatmap
from collections import defaultdict
import numpy as np

# Build co-occurrence matrix
trait_list = sorted(set(trait for gene in all_results for trait in gene["traits"]))
co_occurrence = defaultdict(int)

for gene in all_results:
    traits = gene["traits"]
    for i, trait1 in enumerate(traits):
        for trait2 in traits[i:]:
            co_occurrence[(trait1, trait2)] += 1
            if trait1 != trait2:
                co_occurrence[(trait2, trait1)] += 1

# Convert to matrix
matrix = np.zeros((len(trait_list), len(trait_list)))
for i, trait1 in enumerate(trait_list):
    for j, trait2 in enumerate(trait_list):
        matrix[i, j] = co_occurrence.get((trait1, trait2), 0)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, xticklabels=trait_list, yticklabels=trait_list, 
            annot=True, fmt='.0f', cmap='YlOrRd', square=True,
            cbar_kws={'label': 'Co-occurrence Count'})
plt.title("Trait Co-occurrence in Pleiotropic Genes")
plt.tight_layout()
plt.savefig("trait_cooccurrence.png", dpi=150, bbox_inches='tight')
print("✅ Created trait co-occurrence heatmap")

# 4. Gene Complexity (Number of Traits)
plt.figure(figsize=(10, 6))
trait_counts_per_gene = [len(gene["traits"]) for gene in all_results]
gene_names = [gene["gene_id"][:20] + "..." if len(gene["gene_id"]) > 20 else gene["gene_id"] 
              for gene in all_results]

y_pos = range(len(gene_names))
colors = ['skyblue' if gene["source"] == "Synthetic" else 'lightcoral' for gene in all_results]

plt.barh(y_pos, trait_counts_per_gene, color=colors, edgecolor='black', alpha=0.7)
plt.yticks(y_pos, gene_names)
plt.xlabel("Number of Traits")
plt.ylabel("Gene ID")
plt.title("Pleiotropic Complexity by Gene")
plt.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='skyblue', edgecolor='black', label='Synthetic'),
                  Patch(facecolor='lightcoral', edgecolor='black', label='E. coli')]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig("gene_complexity.png", dpi=150, bbox_inches='tight')
print("✅ Created gene complexity plot")

# Create summary statistics
stats = {
    "total_genes": len(all_results),
    "synthetic_genes": len([g for g in all_results if g["source"] == "Synthetic"]),
    "ecoli_genes": len([g for g in all_results if g["source"] == "E. coli"]),
    "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
    "average_traits_per_gene": sum(trait_counts_per_gene) / len(trait_counts_per_gene) if trait_counts_per_gene else 0,
    "most_common_trait": traits[0] if traits else "None",
    "unique_traits": len(trait_list)
}

with open("analysis_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n📊 Summary Statistics:")
print(f"  Total genes: {stats['total_genes']}")
print(f"  Average confidence: {stats['average_confidence']:.3f}")
print(f"  Average traits per gene: {stats['average_traits_per_gene']:.1f}")
print(f"  Most common trait: {stats['most_common_trait']}")
print(f"  Unique traits detected: {stats['unique_traits']}")