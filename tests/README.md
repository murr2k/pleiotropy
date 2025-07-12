# Testing Documentation

## Overview

This directory contains comprehensive test suites for the Genomic Pleiotropy Cryptanalysis project. The testing infrastructure ensures code quality, performance, and reliability across all components.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_statistical_analyzer.py
│   ├── test_trait_visualizer.py
│   └── test_rust_components.rs
├── integration/            # Integration tests
│   └── test_integration.py
├── performance/           # Performance and benchmark tests
│   └── test_performance.py
├── e2e/                   # End-to-end tests
├── fixtures/              # Test data generators and fixtures
│   └── test_data_generator.py
├── pytest.ini            # Pytest configuration
└── requirements-test.txt # Test dependencies
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all Python tests
pytest

# Run specific test category
pytest tests/unit -v
pytest tests/integration -v
pytest tests/performance -v --benchmark-only

# Run with coverage
pytest --cov=python_analysis --cov-report=html
```

### Rust Tests

```bash
cd rust_impl

# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_crypto_engine

# Run benchmarks
cargo bench
```

### UI Tests

```bash
cd trial_database/ui

# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watchAll
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)
- High code coverage target (> 80%)

### Integration Tests
- Test interaction between components
- Verify Rust-Python interface
- Test database operations
- API endpoint testing (when implemented)

### Performance Tests
- Benchmark critical operations
- Memory profiling
- Scalability testing
- Load testing with large datasets

### End-to-End Tests
- Complete workflow testing
- User scenario simulation
- Full system validation

## Writing Tests

### Python Test Example

```python
import pytest
from python_analysis.statistical_analyzer import StatisticalAnalyzer

class TestStatisticalAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()
    
    def test_correlation_calculation(self, analyzer):
        # Test implementation
        pass
```

### Rust Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decrypt_sequence() {
        // Test implementation
    }
}
```

## Test Data Generation

Use the test data generator for consistent test data:

```python
from tests.fixtures.test_data_generator import TestDataGenerator

generator = TestDataGenerator(seed=42)
genome = generator.generate_genome(n_genes=100)
trials = generator.generate_trial_data(n_trials=1000)
```

## Continuous Integration

Tests run automatically via GitHub Actions on:
- Every push to main/develop branches
- All pull requests
- Manual workflow dispatch

See `.github/workflows/test.yml` for CI configuration.

## Coverage Requirements

- **Overall**: > 80% coverage
- **Core modules**: > 90% coverage
- **Critical paths**: 100% coverage

Generate coverage reports:

```bash
# Python coverage
pytest --cov=python_analysis --cov-report=html
open htmlcov/index.html

# Rust coverage
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

## Performance Benchmarks

Run performance benchmarks:

```bash
# Python benchmarks
pytest tests/performance -v --benchmark-only

# Save benchmark results
pytest tests/performance --benchmark-save=baseline

# Compare with baseline
pytest tests/performance --benchmark-compare=baseline
```

## Test Markers

Available pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow tests (> 5 seconds)
- `@pytest.mark.security` - Security tests

Run specific markers:

```bash
pytest -m "unit"
pytest -m "not slow"
```

## Debugging Tests

### Python Debugging

```bash
# Run with verbose output
pytest -vv

# Show print statements
pytest -s

# Drop to debugger on failure
pytest --pdb

# Run specific test
pytest tests/unit/test_statistical_analyzer.py::test_correlation -v
```

### Rust Debugging

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run single test with output
cargo test test_name -- --nocapture
```

## Security Testing

Security tests check for:
- Input validation
- SQL injection prevention
- XSS protection
- Authentication/authorization
- Dependency vulnerabilities

Run security audit:

```bash
# Python dependencies
safety check -r requirements.txt
bandit -r python_analysis/

# Rust dependencies
cargo audit

# npm dependencies
cd trial_database/ui && npm audit
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Fixtures**: Use fixtures for common test data
3. **Mocking**: Mock external dependencies and I/O operations
4. **Assertions**: Use specific assertions with clear error messages
5. **Performance**: Keep unit tests fast (< 1 second)
6. **Coverage**: Aim for high coverage but focus on critical paths
7. **Documentation**: Document complex test scenarios

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in PYTHONPATH
2. **Rust Build Errors**: Run `cargo clean` and rebuild
3. **Flaky Tests**: Use proper test isolation and cleanup
4. **Performance Tests**: Run on consistent hardware/environment

### Getting Help

- Check test output for detailed error messages
- Run tests in verbose mode for more information
- Review CI logs for environment-specific issues
- Consult team members for complex test scenarios