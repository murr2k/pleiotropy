// CUDA kernel implementations

#[cfg(feature = "cuda")]
pub mod codon_counter;
#[cfg(feature = "cuda")]
pub mod frequency_calculator;
#[cfg(feature = "cuda")]
pub mod pattern_matcher;
#[cfg(feature = "cuda")]
pub mod matrix_processor;
#[cfg(feature = "cuda")]
pub mod prime_factorizer;
#[cfg(feature = "cuda")]
pub mod large_factorization;

#[cfg(feature = "cuda")]
pub use codon_counter::CodonCounter;
#[cfg(feature = "cuda")]
pub use frequency_calculator::FrequencyCalculator;
#[cfg(feature = "cuda")]
pub use pattern_matcher::PatternMatcher;
#[cfg(feature = "cuda")]
pub use matrix_processor::MatrixProcessor;
#[cfg(feature = "cuda")]
pub use prime_factorizer::PrimeFactorizer;
#[cfg(feature = "cuda")]
pub use large_factorization::LargeFactorizer;

// Stub implementations for non-CUDA builds
#[cfg(not(feature = "cuda"))]
pub struct CodonCounter;
#[cfg(not(feature = "cuda"))]
pub struct FrequencyCalculator;
#[cfg(not(feature = "cuda"))]
pub struct PatternMatcher;
#[cfg(not(feature = "cuda"))]
pub struct MatrixProcessor;
#[cfg(not(feature = "cuda"))]
pub struct PrimeFactorizer;
#[cfg(not(feature = "cuda"))]
pub struct LargeFactorizer;