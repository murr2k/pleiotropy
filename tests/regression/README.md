# API Regression Test Suite

**Memory Namespace:** `swarm-regression-1752301224`

A comprehensive API regression testing framework for the Genomic Pleiotropy Trial Tracking System. This suite provides exhaustive testing of all REST endpoints, WebSocket functionality, security vulnerabilities, and performance benchmarks.

## 📋 Overview

This regression test suite is designed to:
- ✅ Test all API endpoints with various payloads
- 🔒 Verify authentication and authorization mechanisms
- 🌐 Validate WebSocket functionality under load
- 🛡️ Check for security vulnerabilities (SQL injection, XSS, etc.)
- ⚡ Measure performance and identify bottlenecks
- 📊 Generate comprehensive reports with actionable insights

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure API server is available
cd ../../trial_database/api
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run All Tests
```bash
# Complete regression test suite
python run_tests.py --all

# Individual test categories
python run_tests.py --api          # API functionality only
python run_tests.py --security     # Security vulnerabilities only
python run_tests.py --performance  # Performance benchmarks only
```

### Quick Validation
```bash
# Run core API tests without coverage (faster)
python run_tests.py --api --no-coverage --no-verbose
```

## 📁 Test Suite Structure

```
tests/regression/
├── conftest.py                           # Pytest configuration and fixtures
├── test_api.py                          # Main API endpoint tests
├── test_security_vulnerabilities.py     # Security vulnerability tests
├── test_performance_benchmarks.py       # Performance and load tests
├── run_tests.py                         # Test runner script
├── requirements.txt                     # Test dependencies
├── API_BUG_REPORT.md                   # Bug findings and reproduction steps
└── README.md                           # This file
```

## 🧪 Test Categories

### 1. API Functionality Tests (`test_api.py`)
- **Agents API:** Registration, login, authentication, role-based access
- **Trials API:** CRUD operations, validation, status transitions
- **Results API:** Creation, validation, filtering, batch operations
- **Progress API:** Updates, tracking, estimation calculations
- **Batch Operations:** Performance testing of bulk operations
- **Error Handling:** Consistent error response formats

### 2. Security Vulnerability Tests (`test_security_vulnerabilities.py`)
- **Injection Attacks:** SQL, NoSQL, Command injection protection
- **Authentication Security:** JWT manipulation, weak passwords, brute force
- **Authorization Testing:** Horizontal/vertical privilege escalation
- **Information Disclosure:** Sensitive data in responses, error messages
- **Input Validation:** XSS, path traversal, malformed requests

### 3. Performance Benchmark Tests (`test_performance_benchmarks.py`)
- **Response Times:** Endpoint latency measurements
- **Throughput Testing:** Concurrent request handling
- **Load Testing:** Sustained load over time
- **WebSocket Performance:** Connection establishment, message latency
- **Resource Usage:** Memory, CPU, database performance monitoring

## 🔍 Test Execution Options

### Command Line Interface
```bash
# Run all tests with full reporting
python run_tests.py --all

# Run specific test categories
python run_tests.py --security
python run_tests.py --performance  
python run_tests.py --api

# Control output and coverage
python run_tests.py --all --no-verbose --no-coverage
```

### Direct Pytest Execution
```bash
# Run specific test files
pytest test_api.py -v
pytest test_security_vulnerabilities.py -v -m security
pytest test_performance_benchmarks.py -v -m performance

# Run with coverage
pytest test_api.py --cov=app --cov-report=html

# Run specific test classes or methods
pytest test_api.py::TestAgentsAPI::test_agent_registration_valid -v
```

### Integration with CI/CD
```yaml
# GitHub Actions example
- name: Run API Regression Tests
  run: |
    cd tests/regression
    python run_tests.py --all
    
# Jenkins pipeline example
stage('API Regression Tests') {
    steps {
        sh 'cd tests/regression && python run_tests.py --all'
        publishHTML([
            allowMissing: false,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'htmlcov',
            reportFiles: 'index.html',
            reportName: 'Coverage Report'
        ])
    }
}
```

## 📊 Report Generation

The test suite automatically generates multiple types of reports:

### 1. Consolidated Test Report
- **File:** `consolidated_regression_report_YYYYMMDD_HHMMSS.json`
- **Content:** Overall test execution summary, timing, success rates

### 2. Security Vulnerability Report
- **File:** `security_vulnerability_report_YYYYMMDD_HHMMSS.json`
- **Content:** Detailed security findings, risk scores, recommendations

### 3. Performance Benchmark Report
- **File:** `performance_benchmark_report_YYYYMMDD_HHMMSS.json`
- **Content:** Response times, throughput metrics, resource usage

### 4. Coverage Report
- **Files:** `htmlcov/index.html`, `coverage.xml`
- **Content:** Code coverage analysis with line-by-line details

## 🔧 Configuration

### Test Configuration
Edit `conftest.py` to modify:
- Database connection settings
- Test data generation
- Performance thresholds
- Security test vectors

### Performance Thresholds
```python
performance_thresholds = {
    "response_time": {
        "fast": 0.1,      # 100ms
        "acceptable": 0.5, # 500ms
        "slow": 2.0       # 2 seconds
    },
    "throughput": {
        "min_requests_per_second": 10,
        "target_requests_per_second": 100
    }
}
```

### Security Test Vectors
```python
security_test_vectors = {
    "sql_injection": [
        "'; DROP TABLE agents; --",
        "' OR '1'='1",
        # ... more vectors
    ],
    "xss": [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        # ... more vectors
    ]
}
```

## 🐛 Bug Reporting and Reproduction

### Found Issues
All discovered bugs are documented in [`API_BUG_REPORT.md`](API_BUG_REPORT.md) with:
- Detailed reproduction steps
- Expected vs actual behavior
- Impact assessment
- Recommended fixes

### Bug Categories
- **High Priority:** Security vulnerabilities, data corruption risks
- **Medium Priority:** Performance issues, API inconsistencies  
- **Low Priority:** Minor usability issues, documentation gaps

### Reproduction Examples
```bash
# Example: Test input validation bug
curl -X POST http://localhost:8000/api/v1/trials/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test",
    "organism": "E. coli", 
    "genome_file": "/test.fasta",
    "parameters": {
      "window_size": -1000,  # Invalid negative value
      "min_confidence": 2.5,  # Invalid > 1.0
      "trait_count": -5       # Invalid negative
    }
  }'
```

## 🔐 Security Testing

### Test Categories
1. **Authentication & Authorization**
   - JWT token manipulation
   - Role-based access control
   - Session management

2. **Input Validation**
   - SQL injection protection
   - XSS prevention
   - Command injection

3. **Information Disclosure**
   - Error message analysis
   - Sensitive data exposure
   - Debug information leaks

### Security Findings Format
```json
{
  "severity": "high",
  "type": "SQL Injection",
  "endpoint": "/api/v1/trials/",
  "payload": "'; DROP TABLE agents; --",
  "description": "SQL injection possible in name field",
  "reproduction_steps": ["Step 1...", "Step 2..."],
  "recommendations": ["Use parameterized queries", "Add input sanitization"]
}
```

## ⚡ Performance Testing

### Metrics Collected
- **Response Times:** min, max, average, 95th percentile
- **Throughput:** requests per second under various loads
- **Resource Usage:** CPU, memory, database connections
- **Scalability:** Performance under concurrent load

### Load Testing Scenarios
1. **Burst Load:** High concurrent requests for short duration
2. **Sustained Load:** Moderate load over extended period
3. **Stress Testing:** Beyond normal capacity limits
4. **WebSocket Load:** Multiple concurrent connections

### Performance Benchmarks
```json
{
  "endpoint": "GET /api/v1/trials/",
  "metrics": {
    "avg_response_time": 0.156,
    "p95_response_time": 0.298,
    "max_throughput": 145.2,
    "error_rate": 0.001
  },
  "sla_compliance": {
    "response_time_sla": true,
    "throughput_sla": true,
    "error_rate_sla": true
  }
}
```

## 🛠️ Extending the Test Suite

### Adding New Tests
1. **Create new test method** in appropriate test class
2. **Use existing fixtures** for authentication and data setup
3. **Follow naming convention:** `test_[feature]_[scenario]`
4. **Add appropriate markers:** `@pytest.mark.security`, `@pytest.mark.performance`

### Example New Test
```python
@pytest.mark.asyncio
async def test_new_endpoint_functionality(self, api_client: AsyncClient, admin_agent: dict):
    """Test new endpoint functionality"""
    response = await api_client.get(
        "/api/v1/new-endpoint/", 
        headers=admin_agent["headers"]
    )
    assert response.status_code == 200
    # Add specific assertions for new functionality
```

### Adding Security Tests
```python
@pytest.mark.security
@pytest.mark.asyncio
async def test_new_security_vulnerability(self, api_client: AsyncClient):
    """Test for new security vulnerability"""
    malicious_payload = "test_payload"
    response = await api_client.post("/api/v1/endpoint/", json={"data": malicious_payload})
    
    # Check for vulnerability indicators
    if response.status_code == 200:
        SecurityTestFramework.log_vulnerability(
            severity="high",
            vuln_type="New Vulnerability Type",
            endpoint="/api/v1/endpoint/",
            payload=malicious_payload,
            description="Description of the vulnerability"
        )
```

## 📈 Monitoring and Alerting

### Continuous Monitoring
- **Automated Execution:** Run tests on every deployment
- **Performance Regression:** Alert on response time increases
- **Security Monitoring:** Alert on new vulnerabilities
- **Coverage Tracking:** Maintain minimum coverage thresholds

### Integration Points
- **CI/CD Pipelines:** Fail builds on critical test failures
- **Monitoring Systems:** Export metrics to Prometheus/Grafana
- **Alert Channels:** Slack/email notifications for failures
- **Dashboard Integration:** Real-time test status displays

## 🤝 Contributing

### Development Workflow
1. **Clone Repository:** `git clone <repo-url>`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Tests:** `python run_tests.py --all`
4. **Add New Tests:** Follow existing patterns and conventions
5. **Submit PR:** Include test results and coverage reports

### Code Standards
- **PEP 8 Compliance:** Use black formatter
- **Type Hints:** Add type annotations where helpful
- **Documentation:** Document complex test scenarios
- **Error Handling:** Graceful handling of test failures

## 📞 Support and Troubleshooting

### Common Issues
1. **Database Connection:** Ensure test database is accessible
2. **Port Conflicts:** Check API server is running on expected port
3. **Dependency Issues:** Verify all requirements are installed
4. **Permission Errors:** Check file system permissions for report generation

### Debug Mode
```bash
# Run with detailed output
pytest test_api.py -v -s --tb=long

# Run single test for debugging
pytest test_api.py::TestAgentsAPI::test_agent_registration_valid -v -s
```

### Memory Namespace Storage
All test findings are stored in: `swarm-regression-1752301224`
- Test execution results
- Security vulnerability findings
- Performance benchmark data
- Bug reproduction steps

---

**Framework Version:** 1.0.0  
**Last Updated:** 2025-07-12  
**Maintainer:** API Test Automation Team