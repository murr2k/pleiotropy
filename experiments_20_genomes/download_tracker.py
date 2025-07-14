#!/usr/bin/env python3
"""
Download Progress Tracker for Genome Acquisition
Monitors progress and allows resuming interrupted downloads
"""

import os
import json
from datetime import datetime
from typing import Dict, List

class DownloadTracker:
    """Track genome download progress and status"""
    
    def __init__(self, tracker_file: str = "download_progress.json"):
        self.tracker_file = tracker_file
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load existing progress or initialize new"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        
        # Initialize progress structure
        return {
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'total_genomes': 20,
            'completed': 0,
            'failed': 0,
            'genomes': {}
        }
    
    def save_progress(self):
        """Save current progress to file"""
        self.progress['last_update'] = datetime.now().isoformat()
        with open(self.tracker_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def mark_started(self, genome_id: str, organism: str, strain: str):
        """Mark a genome download as started"""
        self.progress['genomes'][genome_id] = {
            'organism': organism,
            'strain': strain,
            'status': 'downloading',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'file_size': None,
            'error': None
        }
        self.save_progress()
    
    def mark_completed(self, genome_id: str, file_size: int):
        """Mark a genome download as completed"""
        if genome_id in self.progress['genomes']:
            self.progress['genomes'][genome_id]['status'] = 'completed'
            self.progress['genomes'][genome_id]['end_time'] = datetime.now().isoformat()
            self.progress['genomes'][genome_id]['file_size'] = file_size
            self.progress['completed'] += 1
            self.save_progress()
    
    def mark_failed(self, genome_id: str, error: str):
        """Mark a genome download as failed"""
        if genome_id in self.progress['genomes']:
            self.progress['genomes'][genome_id]['status'] = 'failed'
            self.progress['genomes'][genome_id]['end_time'] = datetime.now().isoformat()
            self.progress['genomes'][genome_id]['error'] = error
            self.progress['failed'] += 1
            self.save_progress()
    
    def get_pending_genomes(self, all_genomes: List[tuple]) -> List[tuple]:
        """Get list of genomes that haven't been successfully downloaded"""
        pending = []
        
        for organism, strain, genome_id in all_genomes:
            if genome_id not in self.progress['genomes']:
                pending.append((organism, strain, genome_id))
            elif self.progress['genomes'][genome_id]['status'] != 'completed':
                pending.append((organism, strain, genome_id))
        
        return pending
    
    def print_summary(self):
        """Print download progress summary"""
        print("\n" + "="*60)
        print("GENOME DOWNLOAD PROGRESS SUMMARY")
        print("="*60)
        print(f"Total genomes: {self.progress['total_genomes']}")
        print(f"Completed: {self.progress['completed']}")
        print(f"Failed: {self.progress['failed']}")
        print(f"In progress/Pending: {self.progress['total_genomes'] - self.progress['completed'] - self.progress['failed']}")
        print(f"Start time: {self.progress['start_time']}")
        print(f"Last update: {self.progress['last_update']}")
        print("\nDetailed Status:")
        print("-"*60)
        
        for genome_id, info in sorted(self.progress['genomes'].items()):
            status_icon = {
                'completed': '✓',
                'failed': '✗',
                'downloading': '⟳'
            }.get(info['status'], '?')
            
            print(f"{status_icon} {info['organism']} {info['strain']}")
            print(f"  Status: {info['status']}")
            if info['file_size']:
                print(f"  Size: {info['file_size']:,} bytes")
            if info['error']:
                print(f"  Error: {info['error']}")
            print()


def create_download_script():
    """Create a batch download script with resume capability"""
    script_content = '''#!/usr/bin/env python3
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
'''
    
    with open('batch_download.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('batch_download.py', 0o755)
    print("Created batch_download.py - Run this to download all genomes with resume capability")


if __name__ == "__main__":
    # Create the batch download script
    create_download_script()
    
    # Show current progress if any
    tracker = DownloadTracker()
    if tracker.progress['genomes']:
        tracker.print_summary()