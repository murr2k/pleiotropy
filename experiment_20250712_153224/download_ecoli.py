#!/usr/bin/env python3
"""Download E. coli K-12 genome from NCBI"""

import urllib.request
import gzip
import shutil
import os

def download_ecoli_genome():
    """Download E. coli K-12 MG1655 genome"""
    # NCBI FTP URL for E. coli K-12 MG1655
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    
    print("Downloading E. coli K-12 genome...")
    
    # Download compressed file
    urllib.request.urlretrieve(url, "ecoli_genome.fna.gz")
    
    # Decompress
    print("Decompressing genome file...")
    with gzip.open("ecoli_genome.fna.gz", 'rb') as f_in:
        with open("ecoli_genome.fasta", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up compressed file
    os.remove("ecoli_genome.fna.gz")
    
    # Get file size
    size = os.path.getsize("ecoli_genome.fasta")
    print(f"✓ E. coli genome downloaded successfully ({size:,} bytes)")
    print(f"✓ Saved as: ecoli_genome.fasta")
    
    # Show first few lines
    print("\nFirst 5 lines of genome file:")
    with open("ecoli_genome.fasta", 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(line.strip())
            else:
                break

if __name__ == "__main__":
    download_ecoli_genome()