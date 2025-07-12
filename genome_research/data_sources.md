# Public Genomic Data Sources for E. coli K-12 Pleiotropic Gene Analysis

## Primary Databases

### 1. NCBI GenBank
- **Resource**: E. coli K-12 MG1655 Complete Genome
- **Accession**: NC_000913.3
- **URL**: https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3
- **Description**: Complete genome sequence of E. coli K-12 substr. MG1655
- **File Format**: FASTA, GenBank
- **Last Updated**: 2023

### 2. RegulonDB
- **Resource**: E. coli Transcriptional Regulatory Network
- **URL**: http://regulondb.ccg.unam.mx/
- **Description**: Database of transcriptional regulation in E. coli K-12
- **Data Types**: 
  - Promoters
  - Transcription factors
  - Regulatory interactions
  - Operons
- **Access**: Direct download or API

### 3. EcoCyc
- **Resource**: E. coli Encyclopedia
- **URL**: https://ecocyc.org/
- **Description**: Comprehensive database of E. coli biology
- **Data Types**:
  - Gene-trait associations
  - Metabolic pathways
  - Protein functions
  - Regulatory information
- **Access**: Web interface, downloadable files

### 4. KEGG (Kyoto Encyclopedia of Genes and Genomes)
- **Resource**: Metabolic Pathway Database
- **URL**: https://www.genome.jp/kegg/
- **E. coli Entry**: eco (E. coli K-12 MG1655)
- **Data Types**:
  - Metabolic pathways
  - Gene-pathway mappings
  - Enzyme functions
- **Access**: REST API, FTP download

### 5. UniProt
- **Resource**: Protein Sequence and Function Database
- **URL**: https://www.uniprot.org/
- **E. coli Proteome**: UP000000625
- **Data Types**:
  - Protein sequences
  - Functional annotations
  - Gene ontology terms
  - Protein-protein interactions
- **Access**: REST API, bulk download

### 6. Codon Usage Database
- **Resource**: Kazusa Codon Usage Database
- **URL**: http://www.kazusa.or.jp/codon/
- **E. coli Entry**: Escherichia coli K12
- **Data Format**: Frequency tables for all codons
- **Access**: Web interface, downloadable tables

## Target Pleiotropic Genes

### Core Set for Validation
1. **crp (b3357)** - cAMP receptor protein
   - RegulonDB ID: ECK120000050
   - Functions: Global transcriptional regulator
   - Affects: Carbon metabolism, motility, virulence

2. **fis (b3261)** - Factor for inversion stimulation
   - RegulonDB ID: ECK120000321
   - Functions: Nucleoid structuring, transcription regulation
   - Affects: Growth phase transitions, DNA topology

3. **rpoS (b2741)** - RNA polymerase sigma S
   - RegulonDB ID: ECK120000770
   - Functions: Stationary phase sigma factor
   - Affects: Stress response, stationary phase genes

4. **hns (b1237)** - Histone-like nucleoid structuring protein
   - RegulonDB ID: ECK120000434
   - Functions: DNA binding, transcriptional silencing
   - Affects: Temperature response, osmotic regulation

5. **ihfA (b1712)** - Integration host factor alpha
   - RegulonDB ID: ECK120000444
   - Functions: DNA bending, transcriptional regulation
   - Affects: Phage integration, gene expression

## Data Retrieval Methods

### NCBI Entrez Direct
```bash
# Install E-utilities
sh -c "$(curl -fsSL ftp://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

# Download E. coli genome
efetch -db nuccore -id NC_000913.3 -format fasta > ecoli_k12_genome.fasta
```

### RegulonDB Downloads
- Bulk download: http://regulondb.ccg.unam.mx/menu/download/datasets/
- Files needed:
  - PromoterSet.txt
  - GeneProductSet.txt
  - regulatory_interactions.txt

### EcoCyc API
```python
# Example API call
import requests
response = requests.get('https://websvc.biocyc.org/getxml?id=ECOLI:GENE-NAME')
```

## Data Quality Notes

1. **Version Control**: Always record the version/date of downloaded data
2. **Cross-Validation**: Verify gene annotations across multiple databases
3. **Coordinate Systems**: E. coli K-12 uses 1-based coordinates
4. **Gene Names**: Use b-numbers for consistency (e.g., b3357 for crp)

## Citation Requirements

When using these datasets, cite:
- NCBI GenBank: Benson DA, et al. Nucleic Acids Res. 2023
- RegulonDB: Tierrafría VH, et al. Nucleic Acids Res. 2022
- EcoCyc: Keseler IM, et al. Nucleic Acids Res. 2021
- KEGG: Kanehisa M, et al. Nucleic Acids Res. 2023
- UniProt: UniProt Consortium. Nucleic Acids Res. 2023