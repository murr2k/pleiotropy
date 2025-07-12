#[cfg(all(test, feature = "cuda"))]
mod test_codon_counter;

#[cfg(all(test, feature = "cuda"))]
mod test_frequency_calculator;

#[cfg(all(test, feature = "cuda"))]
mod test_pattern_matcher;

#[cfg(all(test, feature = "cuda"))]
mod test_matrix_processor;

#[cfg(all(test, feature = "cuda"))]
mod test_memory;

#[cfg(all(test, feature = "cuda"))]
mod test_integration;

#[cfg(all(test, feature = "cuda"))]
mod test_kernel_correctness;

#[cfg(all(test, feature = "cuda"))]
mod test_performance_benchmarks;

#[cfg(all(test, feature = "cuda"))]
mod test_full_pipeline_integration;