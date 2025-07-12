/// Full pipeline integration tests for CUDA implementation
/// Tests the complete workflow from sequence parsing to trait detection

use crate::cuda::CudaAccelerator;
use crate::compute_backend::ComputeBackend;
use crate::types::*;
use crate::sequence_parser::SequenceParser;
use crate::neurodna_trait_detector::NeuroDnaTraitDetector;
use crate::trait_extractor::TraitExtractor;
use std::path::PathBuf;
use std::collections::HashMap;
use anyhow::Result;

/// Create test genome data
fn create_test_genome() -> String {
    // Synthetic genome with known pleiotropic regions
    let mut genome = String::new();
    
    // Region 1: Carbon metabolism + stress response (positions 0-999)
    genome.push_str(&"ATGATGATG".repeat(30)); // High ATG frequency
    genome.push_str(&"GCGGCGGCG".repeat(30)); // High GCG frequency
    genome.push_str(&"AAATTTCCC".repeat(11)); // Mixed codons
    
    // Region 2: Motility genes (positions 1000-1999)
    genome.push_str(&"TTTTTTAAA".repeat(50)); // AT-rich region
    genome.push_str(&"ATCGATCGA".repeat(50)); // Balanced region
    
    // Region 3: Non-coding/low complexity (positions 2000-2999)
    genome.push_str(&"AAAAAAAAAA".repeat(100)); // Low complexity
    
    // Region 4: Regulatory + metabolism (positions 3000-3999)
    genome.push_str(&"GAAGAAGAA".repeat(30)); // High GAA
    genome.push_str(&"ATGATGATG".repeat(30)); // High ATG again
    genome.push_str(&"CGTCGTCGT".repeat(13)); // Mixed
    
    genome
}

fn create_test_trait_definitions() -> Vec<TraitInfo> {
    vec![
        TraitInfo {
            name: "carbon_metabolism".to_string(),
            description: "Carbon metabolism pathways".to_string(),
            associated_genes: vec!["metA".to_string(), "metB".to_string()],
            known_sequences: vec!["ATGATGATG".to_string()],
        },
        TraitInfo {
            name: "stress_response".to_string(),
            description: "Stress response mechanisms".to_string(),
            associated_genes: vec!["stressA".to_string()],
            known_sequences: vec!["GCGGCGGCG".to_string()],
        },
        TraitInfo {
            name: "motility".to_string(),
            description: "Cell motility and flagellar assembly".to_string(),
            associated_genes: vec!["flaA".to_string(), "motB".to_string()],
            known_sequences: vec!["TTTTTTAAA".to_string()],
        },
        TraitInfo {
            name: "regulatory".to_string(),
            description: "Gene regulation and transcription control".to_string(),
            associated_genes: vec!["regX".to_string()],
            known_sequences: vec!["GAAGAAGAA".to_string()],
        },
    ]
}

#[test]
fn test_full_pipeline_gpu_vs_cpu() -> Result<()> {
    // Create test data
    let genome = create_test_genome();
    let traits = create_test_trait_definitions();
    
    // Parse sequences
    let parser = SequenceParser::new();
    let sequences = parser.parse_string(&genome, "test_genome", 300, 150)?;
    
    println!("Parsed {} sequence windows", sequences.len());
    
    // Initialize both backends
    let mut gpu_backend = ComputeBackend::new()?;
    let mut cpu_backend = ComputeBackend::new()?;
    cpu_backend.set_force_cpu(true);
    
    // Build frequency table
    let frequency_table = FrequencyTable {
        codon_frequencies: vec![
            CodonFrequency { codon: "ATG".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
            CodonFrequency { codon: "GCG".to_string(), global_frequency: 0.015, trait_frequencies: HashMap::new() },
            CodonFrequency { codon: "AAA".to_string(), global_frequency: 0.025, trait_frequencies: HashMap::new() },
            CodonFrequency { codon: "TTT".to_string(), global_frequency: 0.025, trait_frequencies: HashMap::new() },
            CodonFrequency { codon: "GAA".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
            // Add more codons as needed
        ],
        trait_info: traits.clone(),
    };
    
    // Process with GPU backend
    let gpu_start = std::time::Instant::now();
    let gpu_regions = gpu_backend.decrypt_sequences(&sequences, &frequency_table)?;
    let gpu_time = gpu_start.elapsed();
    
    // Process with CPU backend
    let cpu_start = std::time::Instant::now();
    let cpu_regions = cpu_backend.decrypt_sequences(&sequences, &frequency_table)?;
    let cpu_time = cpu_start.elapsed();
    
    // Compare results
    println!("\n=== Pipeline Results ===");
    println!("GPU found {} regions in {:?}", gpu_regions.len(), gpu_time);
    println!("CPU found {} regions in {:?}", cpu_regions.len(), cpu_time);
    println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    
    // Verify similar results (may not be exact due to floating point)
    assert!(
        (gpu_regions.len() as i32 - cpu_regions.len() as i32).abs() <= 2,
        "GPU and CPU should find similar number of regions"
    );
    
    // Check performance statistics
    let gpu_stats = gpu_backend.get_stats();
    let cpu_stats = cpu_backend.get_stats();
    
    println!("\n=== Backend Statistics ===");
    println!("GPU calls: {}, failures: {}", gpu_stats.cuda_calls, gpu_stats.cuda_failures);
    println!("CPU calls: {}", cpu_stats.cpu_calls);
    println!("Avg GPU time: {:.2} ms", gpu_stats.avg_cuda_time_ms);
    println!("Avg CPU time: {:.2} ms", cpu_stats.avg_cpu_time_ms);
    
    Ok(())
}

#[test]
fn test_neurodna_integration_with_cuda() -> Result<()> {
    let genome = create_test_genome();
    let traits = create_test_trait_definitions();
    
    // Parse genome
    let parser = SequenceParser::new();
    let sequences = parser.parse_string(&genome, "test_genome", 300, 150)?;
    
    // Initialize components
    let mut compute_backend = ComputeBackend::new()?;
    let neurodna_detector = NeuroDnaTraitDetector::new(0.7);
    let trait_extractor = TraitExtractor::new();
    
    // Create frequency table
    let frequency_table = FrequencyTable {
        codon_frequencies: vec![
            CodonFrequency { codon: "ATG".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
            CodonFrequency { codon: "GCG".to_string(), global_frequency: 0.015, trait_frequencies: HashMap::new() },
            // ... more codons
        ],
        trait_info: traits.clone(),
    };
    
    // Process sequences
    let start = std::time::Instant::now();
    
    // Build codon vectors using GPU
    let codon_vectors = compute_backend.build_codon_vectors(&sequences, &frequency_table)?;
    
    // Detect traits with NeuroDNA
    let mut all_detections = Vec::new();
    for (seq, vector) in sequences.iter().zip(codon_vectors.iter()) {
        let detections = neurodna_detector.detect_traits(seq, vector)?;
        all_detections.extend(detections);
    }
    
    // Extract pleiotropic genes
    let genes = trait_extractor.extract_pleiotropic_genes(&all_detections)?;
    
    let elapsed = start.elapsed();
    
    println!("\n=== NeuroDNA + CUDA Integration Test ===");
    println!("Processing time: {:?}", elapsed);
    println!("Total detections: {}", all_detections.len());
    println!("Pleiotropic genes found: {}", genes.len());
    
    // Verify results
    assert!(!genes.is_empty(), "Should find pleiotropic genes");
    
    // Check for expected patterns
    let has_carbon_stress = genes.iter().any(|g| 
        g.traits.contains(&"carbon_metabolism".to_string()) &&
        g.traits.contains(&"stress_response".to_string())
    );
    
    let has_regulatory_metabolism = genes.iter().any(|g|
        g.traits.contains(&"regulatory".to_string()) &&
        g.traits.contains(&"carbon_metabolism".to_string())
    );
    
    println!("\nFound carbon+stress gene: {}", has_carbon_stress);
    println!("Found regulatory+metabolism gene: {}", has_regulatory_metabolism);
    
    Ok(())
}

#[test]
fn test_error_handling_and_recovery() -> Result<()> {
    let mut backend = ComputeBackend::new()?;
    
    // Test 1: Empty sequences
    let empty_sequences = vec![];
    let empty_freq_table = FrequencyTable {
        codon_frequencies: vec![],
        trait_info: vec![],
    };
    
    let result = backend.decrypt_sequences(&empty_sequences, &empty_freq_table);
    assert!(result.is_ok(), "Should handle empty input gracefully");
    
    // Test 2: Invalid sequences
    let invalid_sequences = vec![
        Sequence {
            id: "invalid".to_string(),
            sequence: "XXXNNNZZZ".to_string(),
            position: 0,
        },
    ];
    
    let result = backend.build_codon_vectors(&invalid_sequences, &empty_freq_table);
    assert!(result.is_ok(), "Should handle invalid sequences");
    
    // Test 3: Very large sequence
    let huge_sequence = vec![
        Sequence {
            id: "huge".to_string(),
            sequence: "ATCG".repeat(10_000_000), // 40MB sequence
            position: 0,
        },
    ];
    
    let freq_table = FrequencyTable {
        codon_frequencies: vec![
            CodonFrequency { codon: "ATG".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
        ],
        trait_info: vec![],
    };
    
    // This should either succeed or fail gracefully
    match backend.build_codon_vectors(&huge_sequence, &freq_table) {
        Ok(vectors) => {
            println!("Successfully processed 40MB sequence, vectors: {}", vectors.len());
        }
        Err(e) => {
            println!("Large sequence processing failed gracefully: {}", e);
        }
    }
    
    Ok(())
}

#[test]
fn test_streaming_large_genome() -> Result<()> {
    // Simulate streaming processing of a large genome
    let chunk_size = 100_000;
    let num_chunks = 50; // 5MB total
    
    let mut backend = ComputeBackend::new()?;
    let mut total_regions = 0;
    let mut total_time = std::time::Duration::ZERO;
    
    let freq_table = FrequencyTable {
        codon_frequencies: vec![
            CodonFrequency { codon: "ATG".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
            CodonFrequency { codon: "GCG".to_string(), global_frequency: 0.015, trait_frequencies: HashMap::new() },
        ],
        trait_info: create_test_trait_definitions(),
    };
    
    println!("\n=== Streaming Large Genome Test ===");
    
    for chunk_idx in 0..num_chunks {
        // Generate chunk
        let chunk = create_test_genome();
        let parser = SequenceParser::new();
        let sequences = parser.parse_string(&chunk, &format!("chunk_{}", chunk_idx), 300, 150)?;
        
        // Process chunk
        let start = std::time::Instant::now();
        let regions = backend.decrypt_sequences(&sequences, &freq_table)?;
        let chunk_time = start.elapsed();
        
        total_regions += regions.len();
        total_time += chunk_time;
        
        if chunk_idx % 10 == 0 {
            println!("Processed chunk {}/{}: {} regions in {:?}", 
                chunk_idx + 1, num_chunks, regions.len(), chunk_time);
        }
    }
    
    println!("\nTotal regions found: {}", total_regions);
    println!("Total processing time: {:?}", total_time);
    println!("Average time per chunk: {:?}", total_time / num_chunks);
    
    let stats = backend.get_stats();
    println!("\nBackend statistics:");
    println!("CUDA usage: {}%", 
        (stats.cuda_calls as f64 / (stats.cuda_calls + stats.cpu_calls) as f64) * 100.0);
    println!("CUDA failures: {}", stats.cuda_failures);
    
    Ok(())
}

#[test]
fn test_concurrent_multi_gpu_simulation() -> Result<()> {
    use std::thread;
    use std::sync::{Arc, Mutex};
    
    // Simulate multiple GPU processing (even if we only have one GPU)
    let num_workers = 4;
    let sequences_per_worker = 250;
    
    let freq_table = Arc::new(FrequencyTable {
        codon_frequencies: vec![
            CodonFrequency { codon: "ATG".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
        ],
        trait_info: create_test_trait_definitions(),
    });
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];
    
    println!("\n=== Concurrent Processing Test ===");
    
    for worker_id in 0..num_workers {
        let freq_table_clone = Arc::clone(&freq_table);
        let results_clone = Arc::clone(&results);
        
        let handle = thread::spawn(move || {
            let mut backend = ComputeBackend::new()
                .expect("Failed to create backend");
            
            // Generate worker-specific sequences
            let parser = SequenceParser::new();
            let genome = create_test_genome();
            let sequences = parser.parse_string(
                &genome, 
                &format!("worker_{}", worker_id), 
                300, 
                150
            ).expect("Failed to parse");
            
            let start = std::time::Instant::now();
            let regions = backend.decrypt_sequences(&sequences, &freq_table_clone)
                .expect("Processing failed");
            let elapsed = start.elapsed();
            
            let mut results = results_clone.lock().unwrap();
            results.push((worker_id, regions.len(), elapsed));
            
            println!("Worker {} completed: {} regions in {:?}", 
                worker_id, regions.len(), elapsed);
        });
        
        handles.push(handle);
    }
    
    // Wait for all workers
    for handle in handles {
        handle.join().expect("Worker thread panicked");
    }
    
    let results = results.lock().unwrap();
    let total_regions: usize = results.iter().map(|(_, count, _)| count).sum();
    let avg_time = results.iter()
        .map(|(_, _, time)| time.as_secs_f64())
        .sum::<f64>() / num_workers as f64;
    
    println!("\nTotal regions from all workers: {}", total_regions);
    println!("Average processing time: {:.3} seconds", avg_time);
    
    Ok(())
}

#[test]
fn test_performance_degradation_detection() -> Result<()> {
    let mut backend = ComputeBackend::new()?;
    let parser = SequenceParser::new();
    let genome = create_test_genome();
    
    let freq_table = FrequencyTable {
        codon_frequencies: vec![
            CodonFrequency { codon: "ATG".to_string(), global_frequency: 0.02, trait_frequencies: HashMap::new() },
        ],
        trait_info: create_test_trait_definitions(),
    };
    
    println!("\n=== Performance Degradation Test ===");
    
    let mut timings = Vec::new();
    
    // Run multiple iterations to detect performance degradation
    for i in 0..10 {
        let sequences = parser.parse_string(&genome, &format!("iteration_{}", i), 300, 150)?;
        
        let start = std::time::Instant::now();
        let _ = backend.decrypt_sequences(&sequences, &freq_table)?;
        let elapsed = start.elapsed();
        
        timings.push(elapsed);
        println!("Iteration {}: {:?}", i + 1, elapsed);
    }
    
    // Check for performance degradation
    let first_half_avg = timings[..5].iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>() / 5.0;
        
    let second_half_avg = timings[5..].iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>() / 5.0;
    
    let degradation = (second_half_avg - first_half_avg) / first_half_avg * 100.0;
    
    println!("\nFirst half average: {:.3} seconds", first_half_avg);
    println!("Second half average: {:.3} seconds", second_half_avg);
    println!("Performance degradation: {:.1}%", degradation);
    
    // Performance should not degrade by more than 10%
    assert!(
        degradation < 10.0,
        "Performance degradation too high: {:.1}%",
        degradation
    );
    
    Ok(())
}