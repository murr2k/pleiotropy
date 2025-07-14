# Genome Acquisition System

This directory contains the genome acquisition system for downloading and managing bacterial genomes from NCBI for the 20 real genome experiment.

## Components

### 1. `genome_acquisition.py`
Main download script that:
- Uses NCBI E-utilities API to search and download genomes
- Downloads complete genome sequences in FASTA format
- Tracks metadata (accession, size, organism info)
- Respects NCBI rate limits (3 requests/second)
- Creates acquisition reports

### 2. `genome_utilities.py`
Utility functions for:
- FASTA file validation
- GC content calculation
- Trait definition generation based on organism characteristics
- Experiment configuration creation

### 3. `download_tracker.py`
Progress tracking system that:
- Monitors download status for all genomes
- Allows resuming interrupted downloads
- Provides progress summaries
- Creates batch download scripts

## Usage

### Quick Start (First 5 Genomes)

```bash
# Download the first batch of 5 genomes
python genome_acquisition.py
```

This will download:
1. Mycobacterium tuberculosis H37Rv
2. Lactobacillus acidophilus NCFM
3. Streptococcus pneumoniae TIGR4
4. Bacillus subtilis 168
5. Helicobacter pylori 26695

### Download All 20 Genomes

```bash
# Create and run the batch download script
python download_tracker.py  # Creates batch_download.py
python batch_download.py    # Downloads all 20 genomes with resume capability
```

### Validate and Generate Trait Definitions

```bash
# After downloading genomes, validate and create trait definitions
python genome_utilities.py
```

## Output Structure

```
genomes/
├── genome_metadata.json          # Metadata for all downloaded genomes
├── acquisition_report.md         # Summary report
├── genome_list.json             # Complete list of all 20 genomes
├── download_progress.json       # Download tracking information
├── mycobacterium_tuberculosis_h37rv.fasta
├── mycobacterium_tuberculosis_h37rv_traits.json
├── mycobacterium_tuberculosis_h37rv_config.json
└── ... (repeated for each genome)
```

## Genome List

### Batch 1 (Ready to download)
1. **Mycobacterium tuberculosis H37Rv** - Pathogen causing tuberculosis
2. **Lactobacillus acidophilus NCFM** - Probiotic bacterium
3. **Streptococcus pneumoniae TIGR4** - Respiratory pathogen
4. **Bacillus subtilis 168** - Model soil bacterium
5. **Helicobacter pylori 26695** - Gastric pathogen

### Batch 2 (Configured for later)
6. Rhizobium leguminosarum - Nitrogen-fixing symbiont
7. Vibrio cholerae - Waterborne pathogen
8. Clostridium difficile 630 - Nosocomial pathogen
9. Caulobacter crescentus CB15 - Aquatic bacterium
10. Synechocystis PCC 6803 - Cyanobacterium
11. Mycoplasma genitalium G37 - Minimal genome
12. Listeria monocytogenes EGD-e - Foodborne pathogen
13. Borrelia burgdorferi B31 - Lyme disease agent
14. Neisseria gonorrhoeae FA 1090 - STI pathogen
15. Campylobacter jejuni NCTC 11168 - Foodborne pathogen
16. Haemophilus influenzae Rd KW20 - Respiratory pathogen
17. Legionella pneumophila Philadelphia 1 - Environmental pathogen
18. Corynebacterium glutamicum ATCC 13032 - Industrial bacterium
19. Enterococcus faecalis V583 - Opportunistic pathogen
20. Thermotoga maritima MSB8 - Hyperthermophile

## Metadata Format

Each genome's metadata includes:
```json
{
  "organism": "Mycobacterium tuberculosis",
  "strain": "H37Rv",
  "accession": "NC_000962.3",
  "version": "NC_000962.3",
  "size": 4411532,
  "filename": "mycobacterium_tuberculosis_h37rv.fasta",
  "file_path": "/path/to/genomes/mycobacterium_tuberculosis_h37rv.fasta",
  "download_date": "2025-01-12T10:30:00"
}
```

## Trait Definitions

Generated trait definitions are organism-specific and include:
- Core metabolic traits (all organisms)
- Organism-specific traits (e.g., sporulation for Bacillus, urease for H. pylori)
- Codon usage bias patterns
- Detection keywords

## Requirements

- Python 3.6+
- Internet connection for NCBI access
- ~100 MB disk space for all 20 genomes

## Notes

- The system respects NCBI's rate limits (max 3 requests/second)
- Downloads can be resumed if interrupted
- All data is sourced from NCBI's public genome database
- Checksums are calculated for data integrity