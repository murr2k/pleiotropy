use super::*;
use crate::cuda::kernels::matrix_processor::MatrixProcessor;
use crate::cuda::device::CudaDevice;

#[test]
fn test_eigendecomposition() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create a simple symmetric matrix
    let size = 4;
    let matrix = vec![
        4.0, 1.0, 2.0, 1.0,
        1.0, 3.0, 1.0, 2.0,
        2.0, 1.0, 4.0, 1.0,
        1.0, 2.0, 1.0, 3.0,
    ];
    
    // Compute eigendecomposition
    let (eigenvalues, eigenvectors) = processor.eigendecompose(&matrix, size)
        .expect("Eigendecomposition failed");
    
    // Verify results
    assert_eq!(eigenvalues.len(), size);
    assert_eq!(eigenvectors.len(), size * size);
    
    // Check that eigenvalues are sorted in descending order
    for i in 1..eigenvalues.len() {
        assert!(eigenvalues[i-1] >= eigenvalues[i] - 1e-5, 
                "Eigenvalues not properly sorted");
    }
}

#[test]
fn test_power_iteration() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create a simple matrix with known dominant eigenvalue
    let size = 3;
    let matrix = vec![
        2.0, 1.0, 0.0,
        1.0, 2.0, 1.0,
        0.0, 1.0, 2.0,
    ];
    
    // Compute dominant eigenvalue/eigenvector
    let (eigenvalue, eigenvector) = processor.dominant_eigen(&matrix, size)
        .expect("Power iteration failed");
    
    // The dominant eigenvalue should be close to 2 + sqrt(2)
    let expected = 2.0 + (2.0f32).sqrt();
    assert!((eigenvalue - expected).abs() < 0.01, 
            "Dominant eigenvalue incorrect: got {}, expected {}", eigenvalue, expected);
    
    assert_eq!(eigenvector.len(), size);
}

#[test]
fn test_correlation_matrix() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create test data matrix (sequences x features)
    let rows = 100;
    let cols = 5;
    let mut data = vec![0.0f32; rows * cols];
    
    // Fill with some pattern
    for i in 0..rows {
        for j in 0..cols {
            data[i * cols + j] = (i as f32 * 0.1 + j as f32) % 10.0;
        }
    }
    
    // Compute correlation matrix
    let correlation = processor.compute_correlation_matrix(&data, rows, cols)
        .expect("Correlation matrix computation failed");
    
    // Verify results
    assert_eq!(correlation.len(), cols * cols);
    
    // Check diagonal elements are 1.0 (self-correlation)
    for i in 0..cols {
        let diag = correlation[i * cols + i];
        assert!((diag - 1.0).abs() < 0.01, 
                "Diagonal element {} should be 1.0, got {}", i, diag);
    }
    
    // Check symmetry
    for i in 0..cols {
        for j in i+1..cols {
            let upper = correlation[i * cols + j];
            let lower = correlation[j * cols + i];
            assert!((upper - lower).abs() < 1e-5, 
                    "Matrix not symmetric at ({},{})", i, j);
        }
    }
}

#[test]
fn test_pca_trait_separation() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create codon frequency data
    let num_sequences = 50;
    let num_codons = 64;
    let num_components = 5;
    
    // Generate synthetic data with pattern
    let mut codon_frequencies = vec![0.0f32; num_sequences * num_codons];
    for i in 0..num_sequences {
        for j in 0..num_codons {
            // Create patterns in the data
            if i < 25 && j < 32 {
                codon_frequencies[i * num_codons + j] = 0.02 + 0.01 * (i % 5) as f32;
            } else if i >= 25 && j >= 32 {
                codon_frequencies[i * num_codons + j] = 0.015 + 0.005 * (j % 8) as f32;
            } else {
                codon_frequencies[i * num_codons + j] = 0.015625; // baseline
            }
        }
    }
    
    // Perform PCA
    let (principal_components, loadings) = processor.pca_trait_separation(
        &codon_frequencies,
        num_sequences,
        num_codons,
        num_components
    ).expect("PCA failed");
    
    // Verify results
    assert_eq!(principal_components.len(), num_sequences * num_components);
    assert_eq!(loadings.len(), num_components * num_codons);
}

#[test]
fn test_svd_trait_factorization() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create trait-codon preference matrix
    let num_traits = 5;
    let num_codons = 64;
    
    // Generate synthetic preference matrix
    let mut trait_codon_matrix = vec![0.0f32; num_traits * num_codons];
    for i in 0..num_traits {
        for j in 0..num_codons {
            // Each trait prefers different codons
            if j % num_traits == i {
                trait_codon_matrix[i * num_codons + j] = 0.8;
            } else {
                trait_codon_matrix[i * num_codons + j] = 0.2;
            }
        }
    }
    
    // Perform SVD
    let (u, s, v) = processor.svd_trait_factorization(
        &trait_codon_matrix,
        num_traits,
        num_codons
    ).expect("SVD failed");
    
    // Verify dimensions
    let k = num_traits.min(num_codons);
    assert_eq!(u.len(), num_traits * k);
    assert_eq!(s.len(), k);
    assert_eq!(v.len(), num_codons * k);
    
    // Singular values should be non-negative and decreasing
    for i in 1..s.len() {
        assert!(s[i-1] >= s[i] - 1e-5, "Singular values not properly ordered");
        assert!(s[i] >= 0.0, "Negative singular value");
    }
}

#[test]
fn test_identify_trait_components() {
    if !CudaDevice::is_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }
    
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    let processor = MatrixProcessor::new(&device).expect("Failed to create matrix processor");
    
    // Create codon frequency data with clear components
    let num_sequences = 100;
    let num_codons = 64;
    
    // Generate data with 3 main components
    let mut codon_frequencies = vec![0.0f32; num_sequences * num_codons];
    
    for i in 0..num_sequences {
        for j in 0..num_codons {
            // Component 1: affects first 20 codons
            if j < 20 {
                codon_frequencies[i * num_codons + j] += 0.01 * (i as f32).sin();
            }
            // Component 2: affects codons 20-40
            if j >= 20 && j < 40 {
                codon_frequencies[i * num_codons + j] += 0.01 * (i as f32 * 2.0).cos();
            }
            // Component 3: affects codons 40-60
            if j >= 40 && j < 60 {
                codon_frequencies[i * num_codons + j] += 0.01 * (i as f32 * 3.0).sin();
            }
            // Add baseline
            codon_frequencies[i * num_codons + j] += 0.015625;
        }
    }
    
    // Identify components that explain 90% of variance
    let components = processor.identify_trait_components(
        &codon_frequencies,
        num_sequences,
        num_codons,
        0.9
    ).expect("Component identification failed");
    
    // Should identify at least 3 main components
    assert!(components.len() >= 3, "Expected at least 3 components, got {}", components.len());
    
    // Check that variance explained adds up
    let total_variance: f32 = components.iter().map(|(_, var, _)| var).sum();
    assert!(total_variance >= 0.89, "Total variance explained should be at least 89%");
    
    // Check eigenvector dimensions
    for (idx, var_explained, eigenvector) in &components {
        assert_eq!(eigenvector.len(), num_codons, "Eigenvector dimension mismatch");
        assert!(*var_explained > 0.0 && *var_explained <= 1.0, 
                "Invalid variance explained: {}", var_explained);
    }
}