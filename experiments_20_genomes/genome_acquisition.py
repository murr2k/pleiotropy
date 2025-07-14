#!/usr/bin/env python3
"""
Genome Acquisition Script for Bacterial Genomes from NCBI
Data Librarian Agent - Downloads complete bacterial genomes with metadata tracking
"""

import os
import json
import time
import logging
from datetime import datetime
from urllib.parse import quote
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
import gzip
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genome_acquisition.log'),
        logging.StreamHandler()
    ]
)

class NCBIGenomeDownloader:
    """Download bacterial genomes from NCBI with metadata tracking"""
    
    def __init__(self, output_dir: str = "genomes", email: str = "your_email@example.com"):
        self.output_dir = output_dir
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.ftp_base = "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
        self.metadata_file = os.path.join(output_dir, "genome_metadata.json")
        self.metadata = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load existing metadata if available
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
    
    def search_genome(self, organism: str, strain: str = "") -> Optional[str]:
        """Search for genome accession number using NCBI E-utilities"""
        query = f"{organism}[Organism] AND complete genome[Title]"
        if strain:
            query += f" AND {strain}[Strain]"
        
        search_url = f"{self.base_url}esearch.fcgi?db=nuccore&term={quote(query)}&retmode=json&retmax=5"
        
        try:
            with urllib.request.urlopen(search_url) as response:
                data = json.loads(response.read().decode())
                
            if data['esearchresult']['count'] == '0':
                logging.warning(f"No results found for {organism} {strain}")
                return None
                
            # Get the first ID
            id_list = data['esearchresult']['idlist']
            if id_list:
                return id_list[0]
            
        except Exception as e:
            logging.error(f"Error searching for {organism}: {e}")
            return None
    
    def get_genome_info(self, uid: str) -> Optional[Dict]:
        """Fetch detailed genome information"""
        fetch_url = f"{self.base_url}efetch.fcgi?db=nuccore&id={uid}&rettype=gb&retmode=text"
        
        try:
            with urllib.request.urlopen(fetch_url) as response:
                gb_data = response.read().decode()
            
            # Parse key information from GenBank format
            info = {
                'uid': uid,
                'accession': '',
                'version': '',
                'organism': '',
                'strain': '',
                'size': 0,
                'download_date': datetime.now().isoformat()
            }
            
            for line in gb_data.split('\n'):
                if line.startswith('ACCESSION'):
                    info['accession'] = line.split()[1]
                elif line.startswith('VERSION'):
                    parts = line.split()
                    info['version'] = parts[1] if len(parts) > 1 else ''
                elif line.startswith('  ORGANISM'):
                    info['organism'] = line.replace('  ORGANISM', '').strip()
                elif line.startswith('SOURCE'):
                    source = line.replace('SOURCE', '').strip()
                    if 'strain' in source.lower():
                        info['strain'] = source
                elif line.startswith('ORIGIN'):
                    # Count sequence length
                    seq_lines = []
                    idx = gb_data.find('ORIGIN')
                    if idx != -1:
                        seq_data = gb_data[idx:].split('\n')[1:]
                        for seq_line in seq_data:
                            if seq_line.startswith('//'):
                                break
                            # Remove line numbers and spaces
                            seq_line = ''.join(seq_line.split()[1:])
                            seq_lines.append(seq_line)
                    info['size'] = len(''.join(seq_lines))
            
            return info
            
        except Exception as e:
            logging.error(f"Error fetching genome info for {uid}: {e}")
            return None
    
    def download_fasta(self, uid: str, accession: str, filename: str) -> bool:
        """Download genome in FASTA format"""
        fasta_url = f"{self.base_url}efetch.fcgi?db=nuccore&id={uid}&rettype=fasta&retmode=text"
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            logging.info(f"Downloading {filename} from NCBI...")
            
            # Download with progress indication
            with urllib.request.urlopen(fasta_url) as response:
                with open(output_path, 'wb') as out_file:
                    # Read in chunks
                    chunk_size = 8192
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (chunk_size * 100) == 0:
                            logging.info(f"Downloaded {downloaded / 1024 / 1024:.1f} MB...")
            
            logging.info(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error downloading {filename}: {e}")
            return False
    
    def download_genome(self, organism: str, strain: str, safe_name: str) -> Optional[Dict]:
        """Complete genome download workflow"""
        logging.info(f"Processing {organism} {strain}")
        
        # Search for genome
        uid = self.search_genome(organism, strain)
        if not uid:
            logging.error(f"Could not find genome for {organism} {strain}")
            return None
        
        # Get genome information
        info = self.get_genome_info(uid)
        if not info:
            return None
        
        # Create filename
        filename = f"{safe_name}.fasta"
        
        # Download FASTA file
        if self.download_fasta(uid, info['accession'], filename):
            info['filename'] = filename
            info['file_path'] = os.path.join(self.output_dir, filename)
            
            # Save metadata
            self.metadata[safe_name] = info
            self.save_metadata()
            
            # Respect NCBI rate limits
            time.sleep(0.34)  # Max 3 requests per second
            
            return info
        
        return None
    
    def save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logging.info(f"Metadata saved to {self.metadata_file}")
    
    def create_summary_report(self):
        """Create a summary report of downloaded genomes"""
        report_path = os.path.join(self.output_dir, "acquisition_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Genome Acquisition Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Downloaded Genomes\n\n")
            f.write("| Organism | Strain | Accession | Size (bp) | File |\n")
            f.write("|----------|--------|-----------|-----------|------|\n")
            
            total_size = 0
            for name, info in sorted(self.metadata.items()):
                f.write(f"| {info['organism']} | {info['strain']} | {info['accession']} | ")
                f.write(f"{info['size']:,} | {info['filename']} |\n")
                total_size += info['size']
            
            f.write(f"\n**Total genomes**: {len(self.metadata)}\n")
            f.write(f"**Total size**: {total_size:,} bp ({total_size/1e6:.1f} Mb)\n")
        
        logging.info(f"Summary report saved to {report_path}")


def main():
    """Main execution function"""
    
    # First batch of 5 genomes as requested
    genomes_batch1 = [
        ("Mycobacterium tuberculosis", "H37Rv", "mycobacterium_tuberculosis_h37rv"),
        ("Lactobacillus acidophilus", "NCFM", "lactobacillus_acidophilus_ncfm"),
        ("Streptococcus pneumoniae", "TIGR4", "streptococcus_pneumoniae_tigr4"),
        ("Bacillus subtilis", "168", "bacillus_subtilis_168"),
        ("Helicobacter pylori", "26695", "helicobacter_pylori_26695")
    ]
    
    # Remaining 15 genomes for later
    genomes_batch2 = [
        ("Rhizobium leguminosarum", "bv. viciae 3841", "rhizobium_leguminosarum"),
        ("Vibrio cholerae", "O1 biovar El Tor str. N16961", "vibrio_cholerae"),
        ("Clostridium difficile", "630", "clostridium_difficile_630"),
        ("Caulobacter crescentus", "CB15", "caulobacter_crescentus_cb15"),
        ("Synechocystis", "PCC 6803", "synechocystis_pcc6803"),
        ("Mycoplasma genitalium", "G37", "mycoplasma_genitalium_g37"),
        ("Listeria monocytogenes", "EGD-e", "listeria_monocytogenes_egde"),
        ("Borrelia burgdorferi", "B31", "borrelia_burgdorferi_b31"),
        ("Neisseria gonorrhoeae", "FA 1090", "neisseria_gonorrhoeae_fa1090"),
        ("Campylobacter jejuni", "NCTC 11168", "campylobacter_jejuni"),
        ("Haemophilus influenzae", "Rd KW20", "haemophilus_influenzae_rd"),
        ("Legionella pneumophila", "Philadelphia 1", "legionella_pneumophila"),
        ("Corynebacterium glutamicum", "ATCC 13032", "corynebacterium_glutamicum"),
        ("Enterococcus faecalis", "V583", "enterococcus_faecalis_v583"),
        ("Thermotoga maritima", "MSB8", "thermotoga_maritima_msb8")
    ]
    
    # Initialize downloader
    downloader = NCBIGenomeDownloader()
    
    # Download first batch
    logging.info("Starting genome acquisition - Batch 1")
    successful = 0
    failed = []
    
    for organism, strain, safe_name in genomes_batch1:
        result = downloader.download_genome(organism, strain, safe_name)
        if result:
            successful += 1
            logging.info(f"✓ Successfully downloaded {organism} {strain}")
        else:
            failed.append((organism, strain))
            logging.error(f"✗ Failed to download {organism} {strain}")
    
    # Create summary report
    downloader.create_summary_report()
    
    # Final summary
    logging.info(f"\n{'='*50}")
    logging.info(f"Batch 1 Complete: {successful}/{len(genomes_batch1)} genomes downloaded")
    if failed:
        logging.info(f"Failed downloads: {failed}")
    logging.info(f"{'='*50}\n")
    
    # Save full genome list for future reference
    all_genomes = {
        "batch1": genomes_batch1,
        "batch2": genomes_batch2,
        "download_status": {
            "batch1_complete": successful == len(genomes_batch1),
            "batch1_successful": successful,
            "batch1_failed": len(failed)
        }
    }
    
    with open(os.path.join(downloader.output_dir, "genome_list.json"), 'w') as f:
        json.dump(all_genomes, f, indent=2)


if __name__ == "__main__":
    main()