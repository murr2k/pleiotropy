#!/usr/bin/env python3
"""
Batch Download Script with Resume Capability
Run this to download all 20 genomes with automatic retry and resume
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genome_acquisition import NCBIGenomeDownloader
from download_tracker import DownloadTracker
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # All 20 genomes
    all_genomes = [
        # Batch 1
        ("Mycobacterium tuberculosis", "H37Rv", "mycobacterium_tuberculosis_h37rv"),
        ("Lactobacillus acidophilus", "NCFM", "lactobacillus_acidophilus_ncfm"),
        ("Streptococcus pneumoniae", "TIGR4", "streptococcus_pneumoniae_tigr4"),
        ("Bacillus subtilis", "168", "bacillus_subtilis_168"),
        ("Helicobacter pylori", "26695", "helicobacter_pylori_26695"),
        # Batch 2
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
    
    # Initialize components
    downloader = NCBIGenomeDownloader()
    tracker = DownloadTracker()
    
    # Get pending genomes
    pending = tracker.get_pending_genomes(all_genomes)
    
    if not pending:
        logging.info("All genomes have been successfully downloaded!")
        tracker.print_summary()
        return
    
    logging.info(f"Found {len(pending)} genomes to download")
    
    # Download pending genomes
    for organism, strain, genome_id in pending:
        tracker.mark_started(genome_id, organism, strain)
        
        try:
            result = downloader.download_genome(organism, strain, genome_id)
            
            if result:
                # Get file size
                file_path = result['file_path']
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                tracker.mark_completed(genome_id, file_size)
                logging.info(f"✓ Successfully downloaded {organism} {strain}")
            else:
                tracker.mark_failed(genome_id, "Download failed")
                logging.error(f"✗ Failed to download {organism} {strain}")
                
        except Exception as e:
            tracker.mark_failed(genome_id, str(e))
            logging.error(f"✗ Error downloading {organism} {strain}: {e}")
        
        # Add delay between downloads
        time.sleep(1)
    
    # Print final summary
    tracker.print_summary()
    
    # Create summary report
    downloader.create_summary_report()


if __name__ == "__main__":
    main()
