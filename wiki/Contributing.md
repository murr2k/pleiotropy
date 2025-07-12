# Contributing to Genomic Pleiotropy Cryptanalysis

Thank you for your interest in contributing! This project thrives on community contributions, from bug fixes to new features to documentation improvements.

## 🤝 Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct:

- **Be respectful** of differing viewpoints and experiences
- **Be constructive** with criticism and feedback
- **Be inclusive** and welcoming to newcomers
- **Be collaborative** and help others learn and grow

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/pleiotropy.git
cd pleiotropy
git remote add upstream https://github.com/murr2k/pleiotropy.git
```

### 2. Set Up Development Environment

```bash
# Install development dependencies
./scripts/setup-dev.sh

# Or manually:
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup component add clippy rustfmt

# Python
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Node.js
nvm install 18
npm install -g pnpm
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

## 📝 Contribution Types

### 🐛 Bug Reports

Found a bug? Please help us fix it!

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error messages/logs

**Bug Report Template:**
```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Run command X
2. Upload file Y
3. Click button Z

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Ubuntu 22.04
- Docker version: 24.0.7
- Browser: Chrome 120

## Logs
```
Error messages here
```
```

### ✨ Feature Requests

Have an idea for a new feature?

1. **Check the roadmap** to see if it's planned
2. **Discuss in GitHub Discussions** first
3. **Create a feature request** with:
   - Use case and motivation
   - Proposed solution
   - Alternative approaches
   - Implementation considerations

### 🔧 Code Contributions

#### Rust Code

```rust
// Follow Rust conventions
// Use rustfmt and clippy
cargo fmt
cargo clippy -- -D warnings

// Write tests for new code
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_feature() {
        // Test implementation
    }
}
```

#### Python Code

```python
# Follow PEP 8
# Use Black formatter
black python_analysis/

# Type hints required
def analyze_trait(sequence: str, confidence: float = 0.5) -> List[Trait]:
    """
    Analyze traits in a sequence.
    
    Args:
        sequence: DNA sequence string
        confidence: Minimum confidence threshold
        
    Returns:
        List of detected traits
    """
    pass

# Write tests
def test_analyze_trait():
    result = analyze_trait("ATCG", 0.5)
    assert len(result) > 0
```

#### TypeScript/React Code

```typescript
// Use TypeScript strict mode
// Follow React best practices

interface Props {
  trialId: string;
  onComplete: (result: AnalysisResult) => void;
}

export const AnalysisComponent: React.FC<Props> = ({ trialId, onComplete }) => {
  // Component implementation
};

// Write tests
describe('AnalysisComponent', () => {
  it('should render correctly', () => {
    const { getByText } = render(<AnalysisComponent {...props} />);
    expect(getByText('Analyze')).toBeInTheDocument();
  });
});
```

### 📚 Documentation

Documentation improvements are always welcome!

- **Fix typos** and grammatical errors
- **Clarify unclear sections**
- **Add examples** and use cases
- **Update outdated information**
- **Translate** to other languages

## 🔄 Development Workflow

### 1. Make Changes

```bash
# Keep your branch updated
git fetch upstream
git rebase upstream/main

# Make your changes
# Write/update tests
# Update documentation
```

### 2. Test Your Changes

```bash
# Run all tests
./scripts/test-all.sh

# Or individually:
# Rust tests
cd rust_impl && cargo test

# Python tests
pytest python_analysis/tests/

# JavaScript tests
cd trial_database/ui && npm test

# Integration tests
./scripts/integration-test.sh
```

### 3. Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <subject>

# Types:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# style: Code style changes (formatting, etc)
# refactor: Code refactoring
# perf: Performance improvements
# test: Test additions/changes
# chore: Maintenance tasks

# Examples:
git commit -m "feat(rust): add GPU acceleration for frequency analysis"
git commit -m "fix(api): handle timeout errors in analysis endpoint"
git commit -m "docs(wiki): update installation guide for macOS"
```

### 4. Submit Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title following commit convention
   - Description of changes
   - Link to related issues
   - Screenshots (if UI changes)
   - Test results

**PR Template:**
```markdown
## Description
Brief description of the changes

## Related Issues
Fixes #123
Related to #456

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🧪 Testing Guidelines

### Test Coverage Requirements

- **New code**: Minimum 80% coverage
- **Critical paths**: 95% coverage
- **UI components**: Snapshot + behavior tests

### Writing Tests

#### Rust Tests
```rust
#[test]
fn test_codon_frequency_calculation() {
    let sequence = "ATGATGATG";
    let frequencies = calculate_frequencies(sequence);
    
    assert_eq!(frequencies.get("ATG"), Some(&1.0));
    assert_eq!(frequencies.len(), 1);
}

#[test]
#[should_panic(expected = "Invalid sequence")]
fn test_invalid_sequence() {
    calculate_frequencies("XYZ");
}
```

#### Python Tests
```python
import pytest
from statistical_analyzer import StatisticalAnalyzer

@pytest.fixture
def analyzer():
    return StatisticalAnalyzer()

def test_chi_squared_calculation(analyzer):
    observed = [10, 20, 30]
    expected = [15, 15, 30]
    
    chi2, p_value = analyzer.chi_squared_test(observed, expected)
    assert chi2 > 0
    assert 0 <= p_value <= 1

@pytest.mark.parametrize("data,expected", [
    ([1, 2, 3], 2.0),
    ([5, 5, 5], 5.0),
])
def test_mean_calculation(analyzer, data, expected):
    assert analyzer.mean(data) == expected
```

#### React Tests
```typescript
import { render, fireEvent, waitFor } from '@testing-library/react';
import { TrialList } from './TrialList';

describe('TrialList', () => {
  it('should filter trials by status', async () => {
    const { getByRole, queryByText } = render(<TrialList trials={mockTrials} />);
    
    const filterSelect = getByRole('combobox', { name: /status filter/i });
    fireEvent.change(filterSelect, { target: { value: 'completed' } });
    
    await waitFor(() => {
      expect(queryByText('Processing')).not.toBeInTheDocument();
      expect(queryByText('Completed')).toBeInTheDocument();
    });
  });
});
```

## 🎨 Style Guidelines

### General Principles

1. **Consistency** over personal preference
2. **Readability** over cleverness
3. **Maintainability** over optimization
4. **Simplicity** over complexity

### Language-Specific Guidelines

#### Rust
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` and `cargo clippy`
- Prefer `Result<T, E>` over panics
- Document public APIs

#### Python
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) formatter
- Type hints for all functions
- Docstrings for public functions

#### TypeScript/JavaScript
- Use [ESLint](https://eslint.org/) configuration
- Prefer functional components in React
- Use TypeScript strict mode
- Document complex logic

## 🚀 Performance Considerations

When contributing performance-critical code:

1. **Profile before optimizing**
   ```bash
   cargo bench
   pytest --benchmark-only
   ```

2. **Document complexity**
   ```rust
   // O(n log n) time complexity, O(n) space
   fn sort_sequences(sequences: &mut [Sequence]) {
       sequences.sort_by_key(|s| s.id.clone());
   }
   ```

3. **Consider memory usage**
   - Use iterators over collections
   - Implement streaming for large files
   - Pool resources when possible

## 🔐 Security Guidelines

1. **Never commit secrets** (API keys, passwords)
2. **Validate all inputs**
3. **Use parameterized queries**
4. **Follow OWASP guidelines**
5. **Report security issues privately**

## 📋 Review Process

### What to Expect

1. **Automated checks** run immediately
2. **Code review** within 48 hours
3. **Feedback** may request changes
4. **Approval** from at least one maintainer
5. **Merge** when all checks pass

### Review Criteria

- **Functionality**: Does it work as intended?
- **Tests**: Are tests comprehensive?
- **Documentation**: Is it well-documented?
- **Performance**: No significant regressions?
- **Security**: No vulnerabilities introduced?
- **Style**: Follows project conventions?

## 🏆 Recognition

We value all contributions! Contributors are:

- Listed in [CONTRIBUTORS.md](../CONTRIBUTORS.md)
- Mentioned in release notes
- Eligible for contributor badges
- Invited to contributor meetings

## 📞 Getting Help

Need help with your contribution?

- **Discord**: Join our [developer chat](https://discord.gg/pleiotropy)
- **Discussions**: Ask in [GitHub Discussions](https://github.com/murr2k/pleiotropy/discussions)
- **Office Hours**: Thursdays 16:00 UTC (check calendar)
- **Email**: contributors@pleiotropy.dev

## 🎯 First-Time Contributors

New to open source? Start here:

1. Look for issues labeled [`good first issue`](https://github.com/murr2k/pleiotropy/labels/good%20first%20issue)
2. Read through the codebase
3. Set up your development environment
4. Make a small change (typo fix, etc.)
5. Submit your first PR!

### Beginner-Friendly Tasks

- Fix typos in documentation
- Add code comments
- Improve error messages
- Write missing tests
- Update examples

## 🌟 Advanced Contributions

For experienced contributors:

### Architecture Changes

1. Discuss in an issue first
2. Write an RFC (Request for Comments)
3. Get consensus from maintainers
4. Implement in phases

### New Algorithms

1. Provide mathematical foundation
2. Include benchmarks
3. Compare with existing methods
4. Add comprehensive tests

### Major Features

1. Break into smaller PRs
2. Use feature flags
3. Maintain backward compatibility
4. Update all documentation

---

*Thank you for contributing to Genomic Pleiotropy Cryptanalysis! Your efforts help advance our understanding of genomics.* 🧬✨