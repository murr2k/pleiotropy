#!/usr/bin/env python3
"""Download Salmonella enterica serovar Typhimurium genome from NCBI"""

import urllib.request
import gzip
import shutil
import os

def download_salmonella_genome():
    """Download Salmonella enterica serovar Typhimurium LT2 genome"""
    # NCBI FTP URL for Salmonella enterica serovar Typhimurium LT2
    # This is a well-studied pathogenic strain
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/GCF_000006945.2_ASM694v2/GCF_000006945.2_ASM694v2_genomic.fna.gz"
    
    print("Downloading Salmonella enterica serovar Typhimurium LT2 genome...")
    print("This strain is a model organism for studying bacterial pathogenesis")
    
    try:
        # Download compressed file
        urllib.request.urlretrieve(url, "salmonella_genome.fna.gz")
        
        # Decompress
        print("Decompressing genome file...")
        with gzip.open("salmonella_genome.fna.gz", 'rb') as f_in:
            with open("salmonella_genome.fasta", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Clean up compressed file
        os.remove("salmonella_genome.fna.gz")
        
        # Get file info
        size = os.path.getsize("salmonella_genome.fasta")
        print(f"✓ Salmonella genome downloaded successfully ({size:,} bytes)")
        print(f"✓ Saved as: salmonella_genome.fasta")
        
        # Count sequences and show preview
        print("\nGenome information:")
        seq_count = 0
        total_length = 0
        with open("salmonella_genome.fasta", 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('>'):
                    seq_count += 1
                    if seq_count <= 3:
                        print(f"  Sequence {seq_count}: {line.strip()}")
                elif seq_count == 1 and i < 10:
                    total_length += len(line.strip())
        
        print(f"\nTotal sequences: {seq_count}")
        print("Main chromosome + plasmids")
        
    except Exception as e:
        print(f"Error downloading genome: {e}")
        print("Trying alternative approach...")
        # Alternative: Use a direct link to a Salmonella genome
        alt_url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/GCF_000006945.2_ASM694v2/GCF_000006945.2_ASM694v2_genomic.fna.gz"
        urllib.request.urlretrieve(alt_url, "salmonella_genome.fna.gz")

if __name__ == "__main__":
    download_salmonella_genome()