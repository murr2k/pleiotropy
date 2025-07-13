#!/usr/bin/env python3
"""Download Pseudomonas aeruginosa PAO1 genome from NCBI"""

import urllib.request
import gzip
import shutil
import os

def download_pseudomonas_genome():
    """Download Pseudomonas aeruginosa PAO1 reference genome"""
    # NCBI FTP URL for Pseudomonas aeruginosa PAO1
    # PAO1 is the most studied reference strain
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765/GCF_000006765.1_ASM676v1/GCF_000006765.1_ASM676v1_genomic.fna.gz"
    
    print("Downloading Pseudomonas aeruginosa PAO1 genome...")
    print("PAO1 characteristics:")
    print("- Opportunistic pathogen")
    print("- Metabolically versatile")
    print("- High intrinsic antibiotic resistance")
    print("- Large genome (~6.3 Mb)")
    
    try:
        # Download compressed file
        print("\nDownloading from NCBI...")
        urllib.request.urlretrieve(url, "pseudomonas_genome.fna.gz")
        
        # Decompress
        print("Decompressing genome file...")
        with gzip.open("pseudomonas_genome.fna.gz", 'rb') as f_in:
            with open("pseudomonas_genome.fasta", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Clean up compressed file
        os.remove("pseudomonas_genome.fna.gz")
        
        # Get file info and preview
        size = os.path.getsize("pseudomonas_genome.fasta")
        print(f"\n✓ Pseudomonas genome downloaded successfully ({size:,} bytes)")
        print(f"✓ Saved as: pseudomonas_genome.fasta")
        
        # Show genome information
        print("\nGenome preview:")
        with open("pseudomonas_genome.fasta", 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    if len(line) > 80:
                        print(line[:80] + "...")
                    else:
                        print(line.strip())
                else:
                    break
        
        print("\nPseudomonas aeruginosa PAO1 features:")
        print("- Single circular chromosome")
        print("- 6,264,404 base pairs")
        print("- 5,572 predicted genes")
        print("- GC content: 66.6%")
        print("- Known for biofilm formation and quorum sensing")
        
    except Exception as e:
        print(f"Error downloading genome: {e}")

if __name__ == "__main__":
    download_pseudomonas_genome()