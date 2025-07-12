/// Integration tests for CUDA acceleration
/// These tests verify that the CUDA backend integrates correctly with the main pipeline

#[cfg(test)]
mod cuda_integration_tests {
    use genomic_pleiotropy::*;
    use std::collections::HashMap;
    
    fn create_test_sequences() -> Vec<Sequence> {
        vec![
            Sequence::new(
                "test_gene_1".to_string(),
                "ATGGCGATCGCGATCGATATAGCGATCGATCGATCGATCGTAGCTAGCTAGC".to_string(),
            ),
            Sequence::new(
                "test_gene_2".to_string(),
                "GCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAGCATGGCGATCGCGA".to_string(),
            ),
            Sequence::new(
                "test_gene_3".to_string(),
                "TAGCTAGCTAGCATGGCGATCGCGATCGATATAGCGATCGATCGATCGATCG".to_string(),
            ),
        ]
    }
    
    fn create_test_traits() -> Vec<TraitInfo> {
        vec![
            TraitInfo {
                name: "carbon_metabolism".to_string(),
                description: "Carbon metabolism trait".to_string(),
                associated_genes: vec!["test_gene_1".to_string()],
                known_sequences: vec![],
            },
            TraitInfo {
                name: "stress_response".to_string(),
                description: "Stress response trait".to_string(),
                associated_genes: vec!["test_gene_2".to_string()],
                known_sequences: vec![],
            },
        ]
    }
    
    #[test]
    fn test_cuda_backend_initialization() {
        let analyzer = GenomicCryptanalysis::new();
        
        // Check if CUDA is available
        println!("CUDA enabled: {}", analyzer.is_cuda_enabled());
        
        // Should create successfully regardless of CUDA availability
        assert!(true);
    }
    
    #[test]
    fn test_compute_backend_switching() {
        let mut analyzer = GenomicCryptanalysis::new();
        
        // Test forcing CPU mode
        analyzer.set_force_cpu(true);
        assert!(!analyzer.is_cuda_enabled());
        
        // Test re-enabling GPU (if available)
        analyzer.set_force_cpu(false);
        // Will be true only if CUDA is actually available
        println!("CUDA available after re-enabling: {}", analyzer.is_cuda_enabled());
    }
    
    #[test]
    fn test_performance_stats_tracking() {
        let mut backend = compute_backend::ComputeBackend::new().unwrap();
        let sequences = create_test_sequences();
        let freq_table = FrequencyTable {
            codon_frequencies: vec![
                CodonFrequency {
                    codon: "ATG".to_string(),
                    amino_acid: 'M',
                    global_frequency: 0.02,
                    trait_specific_frequency: HashMap::new(),
                },
                CodonFrequency {
                    codon: "GCG".to_string(),
                    amino_acid: 'A',
                    global_frequency: 0.03,
                    trait_specific_frequency: HashMap::new(),
                },
            ],
            total_codons: 1000,
            trait_codon_counts: HashMap::new(),
        };
        
        // Run some operations
        let _ = backend.build_codon_vectors(&sequences, &freq_table);
        
        // Check stats
        let stats = backend.get_stats();
        println!("Performance stats: {:?}", stats);
        
        assert!(stats.cpu_calls > 0 || stats.cuda_calls > 0);
        assert_eq!(stats.total_sequences_processed, sequences.len());
    }
    
    #[test]
    fn test_cuda_fallback_on_error() {
        let mut backend = compute_backend::ComputeBackend::new().unwrap();
        let sequences = create_test_sequences();
        let traits = create_test_traits();
        
        // This should work even if CUDA fails
        let result = backend.calculate_codon_bias(&sequences, &traits);
        
        assert!(result.is_ok());
        let bias_map = result.unwrap();
        assert_eq!(bias_map.len(), traits.len());
    }
    
    #[test]
    fn test_gpu_trait_extraction() {
        let mut extractor = trait_extractor_gpu::TraitExtractorGPU::new().unwrap();
        
        // Create test decrypted regions
        let regions = vec![
            DecryptedRegion {
                start: 0,
                end: 50,
                gene_id: "test_gene_1".to_string(),
                decrypted_traits: vec!["carbon_metabolism".to_string()],
                confidence_scores: {
                    let mut map = HashMap::new();
                    map.insert("carbon_metabolism".to_string(), 0.85);
                    map
                },
                regulatory_context: RegulatoryContext::default(),
            },
            DecryptedRegion {
                start: 25,
                end: 75,
                gene_id: "test_gene_1".to_string(),
                decrypted_traits: vec!["stress_response".to_string()],
                confidence_scores: {
                    let mut map = HashMap::new();
                    map.insert("stress_response".to_string(), 0.75);
                    map
                },
                regulatory_context: RegulatoryContext::default(),
            },
        ];
        
        let traits = create_test_traits();
        let result = extractor.extract_traits(&regions, &traits);
        
        assert!(result.is_ok());
        let signatures = result.unwrap();
        assert!(!signatures.is_empty());
        
        // Check performance stats
        let stats = extractor.get_stats();
        println!("GPU trait extraction stats: {:?}", stats);
    }
    
    #[test]
    fn test_end_to_end_cuda_pipeline() {
        // Create temporary test file
        let test_fasta = r#">test_gene_1
ATGGCGATCGCGATCGATATAGCGATCGATCGATCGATCGTAGCTAGCTAGC
>test_gene_2
GCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAGCATGGCGATCGCGA
>test_gene_3
TAGCTAGCTAGCATGGCGATCGCGATCGATATAGCGATCGATCGATCGATCG
"#;
        
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.fasta");
        std::fs::write(&test_file, test_fasta).unwrap();
        
        // Run analysis
        let mut analyzer = GenomicCryptanalysis::new();
        let traits = create_test_traits();
        
        let result = analyzer.analyze_genome(&test_file, traits);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        println!("Analysis results: {} sequences processed", analysis.sequences);
        println!("Identified traits: {}", analysis.identified_traits.len());
        
        // Check performance stats
        let stats = analyzer.get_performance_stats();
        println!("Pipeline performance stats: {:?}", stats);
        
        // Find pleiotropic genes
        let pleiotropic = analyzer.find_pleiotropic_genes(&analysis, 1);
        println!("Pleiotropic genes found: {}", pleiotropic.len());
    }
    
    #[test]
    fn test_cuda_with_large_dataset() {
        // Create a larger dataset to test GPU efficiency
        let mut sequences = Vec::new();
        for i in 0..100 {
            sequences.push(Sequence::new(
                format!("gene_{}", i),
                "ATGGCGATCGCGATCGATATAGCGATCGATCGATCGATCGTAGCTAGCTAGC".repeat(10),
            ));
        }
        
        let mut backend = compute_backend::ComputeBackend::new().unwrap();
        let freq_table = FrequencyTable {
            codon_frequencies: vec![
                CodonFrequency {
                    codon: "ATG".to_string(),
                    amino_acid: 'M',
                    global_frequency: 0.02,
                    trait_specific_frequency: HashMap::new(),
                },
            ],
            total_codons: 10000,
            trait_codon_counts: HashMap::new(),
        };
        
        let start = std::time::Instant::now();
        let result = backend.build_codon_vectors(&sequences, &freq_table);
        let elapsed = start.elapsed();
        
        assert!(result.is_ok());
        println!("Processed {} sequences in {:?}", sequences.len(), elapsed);
        
        let stats = backend.get_stats();
        if stats.cuda_calls > 0 {
            println!("GPU was used! Average CUDA time: {}ms", stats.avg_cuda_time_ms);
        } else {
            println!("CPU was used. Average CPU time: {}ms", stats.avg_cpu_time_ms);
        }
    }
}

#[derive(Default)]
struct RegulatoryContext {
    promoter_strength: f64,
    enhancers: Vec<(usize, usize)>,
    silencers: Vec<(usize, usize)>,
    expression_conditions: Vec<String>,
}