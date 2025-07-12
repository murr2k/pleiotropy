use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use genomic_cryptanalysis::{
    cuda::{CudaAccelerator, cuda_available},
    types::{DnaSequence, TraitInfo},
    FrequencyAnalyzer,
};
use std::time::Duration;

fn generate_test_sequence(size: usize) -> DnaSequence {
    let bases = ['A', 'C', 'G', 'T'];
    let sequence: String = (0..size)
        .map(|i| bases[i % 4])
        .collect();
    
    DnaSequence::new(format!("seq_{}", size), sequence)
}

fn generate_test_sequences(count: usize, size: usize) -> Vec<DnaSequence> {
    (0..count)
        .map(|i| {
            let mut seq = generate_test_sequence(size);
            seq.id = format!("seq_{}", i);
            seq
        })
        .collect()
}

fn bench_codon_counting(c: &mut Criterion) {
    if !cuda_available() {
        println!("CUDA not available, skipping GPU benchmarks");
        return;
    }
    
    let mut group = c.benchmark_group("codon_counting");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [1000, 10000, 100000, 1000000].iter() {
        let sequences = generate_test_sequences(10, *size);
        
        // CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("CPU", size),
            &sequences,
            |b, seqs| {
                let mut analyzer = FrequencyAnalyzer::new();
                b.iter(|| {
                    analyzer.count_codons_cpu(black_box(seqs))
                });
            },
        );
        
        // GPU benchmark
        group.bench_with_input(
            BenchmarkId::new("GPU", size),
            &sequences,
            |b, seqs| {
                let mut accelerator = CudaAccelerator::new()
                    .expect("Failed to create CUDA accelerator");
                b.iter(|| {
                    accelerator.count_codons(black_box(seqs))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_frequency_calculation(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    
    let mut group = c.benchmark_group("frequency_calculation");
    
    // Pre-generate codon counts
    let sequences = generate_test_sequences(100, 10000);
    let analyzer = FrequencyAnalyzer::new();
    let codon_counts = analyzer.count_codons_cpu(&sequences)
        .expect("Failed to count codons");
    
    let traits = vec![
        TraitInfo {
            name: "trait1".to_string(),
            description: "Test trait 1".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "trait2".to_string(),
            description: "Test trait 2".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
    ];
    
    // CPU benchmark
    group.bench_function("CPU", |b| {
        let mut analyzer = FrequencyAnalyzer::new();
        b.iter(|| {
            analyzer.calculate_frequencies_cpu(
                black_box(&codon_counts),
                black_box(&traits)
            )
        });
    });
    
    // GPU benchmark
    group.bench_function("GPU", |b| {
        let mut accelerator = CudaAccelerator::new()
            .expect("Failed to create CUDA accelerator");
        b.iter(|| {
            accelerator.calculate_frequencies(
                black_box(&codon_counts),
                black_box(&traits)
            )
        });
    });
    
    group.finish();
}

fn bench_pattern_matching(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    
    let mut group = c.benchmark_group("pattern_matching");
    group.measurement_time(Duration::from_secs(15));
    
    // Prepare test data
    let sequences = generate_test_sequences(50, 50000);
    let mut accelerator = CudaAccelerator::new()
        .expect("Failed to create CUDA accelerator");
    
    let codon_counts = accelerator.count_codons(&sequences)
        .expect("Failed to count codons");
    
    let traits = vec![
        TraitInfo {
            name: "complex_trait".to_string(),
            description: "Complex metabolic trait".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
    ];
    
    let freq_table = accelerator.calculate_frequencies(&codon_counts, &traits)
        .expect("Failed to calculate frequencies");
    
    // Create trait patterns
    use crate::types::TraitPattern;
    let patterns: Vec<TraitPattern> = (0..10)
        .map(|i| TraitPattern {
            trait_name: format!("pattern_{}", i),
            preferred_codons: vec!["ATG".to_string(), "GCG".to_string()],
            avoided_codons: vec!["TAA".to_string(), "TAG".to_string()],
            motifs: vec![],
            weight: 1.0,
            codon_preferences: Default::default(),
            regulatory_patterns: vec![],
        })
        .collect();
    
    // GPU benchmark
    group.bench_function("GPU_pattern_match", |b| {
        b.iter(|| {
            accelerator.match_patterns(
                black_box(&freq_table),
                black_box(&patterns)
            )
        });
    });
    
    group.finish();
}

fn bench_matrix_operations(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    
    let mut group = c.benchmark_group("matrix_operations");
    
    for size in [64, 128, 256, 512].iter() {
        // Create correlation matrix
        let matrix: Vec<f32> = (0..size * size)
            .map(|i| (i as f32).sin())
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("eigenanalysis", size),
            &matrix,
            |b, mat| {
                let mut accelerator = CudaAccelerator::new()
                    .expect("Failed to create CUDA accelerator");
                b.iter(|| {
                    accelerator.eigenanalysis(black_box(mat), *size)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    
    let mut group = c.benchmark_group("end_to_end");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    // E. coli-sized genome simulation
    let sequences = generate_test_sequences(4000, 1000); // ~4MB genome
    
    let traits = vec![
        TraitInfo {
            name: "metabolism".to_string(),
            description: "Metabolic functions".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "motility".to_string(),
            description: "Cell motility".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
        TraitInfo {
            name: "stress_response".to_string(),
            description: "Stress response".to_string(),
            associated_genes: vec![],
            known_sequences: vec![],
        },
    ];
    
    group.bench_function("GPU_full_analysis", |b| {
        b.iter(|| {
            let mut accelerator = CudaAccelerator::new()
                .expect("Failed to create CUDA accelerator");
            
            // Full pipeline
            let codon_counts = accelerator.count_codons(&sequences)
                .expect("Failed to count codons");
            
            let _freq_table = accelerator.calculate_frequencies(&codon_counts, &traits)
                .expect("Failed to calculate frequencies");
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_codon_counting,
    bench_frequency_calculation,
    bench_pattern_matching,
    bench_matrix_operations,
    bench_end_to_end
);
criterion_main!(benches);