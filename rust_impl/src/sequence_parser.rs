use crate::types::Sequence;
use anyhow::{Context, Result};
use bio::io::{fasta, fastq};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct SequenceParser {
    min_length: usize,
    max_length: Option<usize>,
}

impl SequenceParser {
    pub fn new() -> Self {
        Self {
            min_length: 100,
            max_length: None,
        }
    }

    pub fn with_length_filter(mut self, min: usize, max: Option<usize>) -> Self {
        self.min_length = min;
        self.max_length = max;
        self
    }

    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Sequence>> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension {
            "fasta" | "fa" | "fna" => self.parse_fasta(path),
            "fastq" | "fq" => self.parse_fastq(path),
            "gb" | "genbank" => self.parse_genbank(path),
            _ => self.parse_fasta(path), // Default to FASTA
        }
    }

    fn parse_fasta(&self, path: &Path) -> Result<Vec<Sequence>> {
        let file = File::open(path).context("Failed to open FASTA file")?;
        let reader = fasta::Reader::new(BufReader::new(file));
        
        let mut sequences = Vec::new();
        
        for result in reader.records() {
            let record = result.context("Failed to parse FASTA record")?;
            let seq_len = record.seq().len();
            
            // Apply length filters
            if seq_len < self.min_length {
                continue;
            }
            if let Some(max) = self.max_length {
                if seq_len > max {
                    continue;
                }
            }
            
            sequences.push(Sequence {
                id: record.id().to_string(),
                name: record.desc().unwrap_or("").to_string(),
                sequence: String::from_utf8_lossy(record.seq()).to_string(),
                annotations: std::collections::HashMap::new(),
            });
        }
        
        Ok(sequences)
    }

    fn parse_fastq(&self, path: &Path) -> Result<Vec<Sequence>> {
        let file = File::open(path).context("Failed to open FASTQ file")?;
        let reader = fastq::Reader::new(BufReader::new(file));
        
        let mut sequences = Vec::new();
        
        for result in reader.records() {
            let record = result.context("Failed to parse FASTQ record")?;
            let seq_len = record.seq().len();
            
            // Apply length filters
            if seq_len < self.min_length {
                continue;
            }
            if let Some(max) = self.max_length {
                if seq_len > max {
                    continue;
                }
            }
            
            sequences.push(Sequence {
                id: record.id().to_string(),
                name: record.desc().unwrap_or("").to_string(),
                sequence: String::from_utf8_lossy(record.seq()).to_string(),
                annotations: std::collections::HashMap::new(),
            });
        }
        
        Ok(sequences)
    }

    fn parse_genbank(&self, _path: &Path) -> Result<Vec<Sequence>> {
        // For now, return empty vec - would implement GenBank parser
        Ok(Vec::new())
    }

    /// Extract coding sequences from a larger sequence
    pub fn extract_cds(&self, sequence: &Sequence, start: usize, end: usize) -> Result<Sequence> {
        if end > sequence.sequence.len() || start >= end {
            anyhow::bail!("Invalid CDS coordinates");
        }

        let cds_seq = &sequence.sequence[start..end];
        
        Ok(Sequence {
            id: format!("{}_{}-{}", sequence.id, start, end),
            name: format!("{} CDS", sequence.name),
            sequence: cds_seq.to_string(),
            annotations: sequence.annotations.clone(),
        })
    }

    /// Convert DNA to RNA (T -> U)
    pub fn dna_to_rna(&self, dna: &str) -> String {
        dna.replace('T', "U").replace('t', "u")
    }

    /// Get reverse complement
    pub fn reverse_complement(&self, dna: &str) -> String {
        dna.chars()
            .rev()
            .map(|c| match c {
                'A' | 'a' => 'T',
                'T' | 't' => 'A',
                'G' | 'g' => 'C',
                'C' | 'c' => 'G',
                _ => c,
            })
            .collect()
    }
}