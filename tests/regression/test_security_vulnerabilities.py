"""
Security Vulnerability Testing Suite
Comprehensive security testing for the Genomic Pleiotropy API

Memory Namespace: swarm-regression-1752301224/security
"""

import pytest
import asyncio
import json
import base64
import time
from typing import Dict, List, Any
from httpx import AsyncClient
from datetime import datetime, timedelta
import jwt
from faker import Faker

fake = Faker()

# Security findings storage
SECURITY_FINDINGS = {
    "critical": [],
    "high": [],
    "medium": [],
    "low": [],
    "info": []
}


class SecurityTestFramework:
    """Framework for organizing security tests"""
    
    @staticmethod
    def log_vulnerability(severity: str, vuln_type: str, endpoint: str, 
                         payload: Any, description: str, **kwargs):
        """Log a security vulnerability finding"""
        finding = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "type": vuln_type,
            "endpoint": endpoint,
            "payload": payload,
            "description": description,
            **kwargs
        }
        SECURITY_FINDINGS[severity].append(finding)
    
    @staticmethod
    def check_response_for_leaks(response_data: Dict, sensitive_patterns: List[str]) -> List[str]:
        """Check response for potential information leaks"""
        leaks = []
        response_str = json.dumps(response_data).lower()
        
        for pattern in sensitive_patterns:
            if pattern.lower() in response_str:
                leaks.append(pattern)
        
        return leaks


@pytest.mark.security
class TestAuthenticationSecurity:
    """Test authentication security vulnerabilities"""
    
    @pytest.mark.asyncio
    async def test_weak_password_policy(self, api_client: AsyncClient):
        """Test weak password acceptance"""
        weak_passwords = [
            "123",
            "password",
            "admin",
            "test",
            "12345678",
            "qwerty123",
            "password123"
        ]
        
        for weak_password in weak_passwords:
            agent_data = {
                "name": f"weak_pass_test_{fake.uuid4()[:8]}",
                "password": weak_password,
                "role": "analyzer"
            }
            
            response = await api_client.post("/api/v1/agents/register", json=agent_data)
            
            if response.status_code == 201:
                SecurityTestFramework.log_vulnerability(
                    severity="medium",
                    vuln_type="Weak Password Policy",
                    endpoint="/api/v1/agents/register",
                    payload={"password": weak_password},
                    description=f"System accepts weak password: {weak_password}"
                )
    
    @pytest.mark.asyncio
    async def test_jwt_token_manipulation(self, api_client: AsyncClient):
        """Test JWT token manipulation vulnerabilities"""
        # Create a test agent first
        agent_data = {
            "name": f"jwt_test_{fake.uuid4()[:8]}",
            "password": "secure_password_123",
            "role": "analyzer"
        }
        
        reg_response = await api_client.post("/api/v1/agents/register", json=agent_data)
        assert reg_response.status_code == 201
        
        # Login to get valid token
        login_response = await api_client.post(
            "/api/v1/agents/login",
            data={"username": agent_data["name"], "password": agent_data["password"]}
        )
        assert login_response.status_code == 200
        valid_token = login_response.json()["access_token"]
        
        # Test various JWT manipulation attacks
        jwt_attacks = [
            # Algorithm confusion attack
            self._create_malicious_jwt("none", {"sub": "1", "role": "coordinator"}),
            
            # Modified claims
            self._modify_jwt_claims(valid_token, {"role": "coordinator", "sub": "999"}),
            
            # Expired token (if not properly validated)
            self._create_expired_jwt(),
            
            # Invalid signature
            valid_token[:-10] + "malicious",
            
            # Null byte injection
            valid_token + "\x00admin",
        ]
        
        for malicious_token in jwt_attacks:
            if malicious_token:
                headers = {"Authorization": f"Bearer {malicious_token}"}
                response = await api_client.get("/api/v1/agents/me", headers=headers)
                
                if response.status_code == 200:
                    SecurityTestFramework.log_vulnerability(
                        severity="critical",
                        vuln_type="JWT Token Manipulation",
                        endpoint="/api/v1/agents/me",
                        payload=malicious_token,
                        description="Malicious JWT token accepted",
                        attack_type="JWT manipulation"
                    )
    
    def _create_malicious_jwt(self, algorithm: str, payload: Dict) -> str:
        """Create malicious JWT with specified algorithm"""
        try:
            if algorithm == "none":
                header = {"alg": "none", "typ": "JWT"}
                header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
                payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
                return f"{header_b64}.{payload_b64}."
            return None
        except:
            return None
    
    def _modify_jwt_claims(self, token: str, new_claims: Dict) -> str:
        """Attempt to modify JWT claims (will have invalid signature)"""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            # Decode payload
            payload_data = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
            
            # Modify claims
            payload_data.update(new_claims)
            
            # Re-encode payload
            new_payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip('=')
            
            return f"{parts[0]}.{new_payload}.{parts[2]}"
        except:
            return None
    
    def _create_expired_jwt(self) -> str:
        """Create an expired JWT token"""
        try:
            expired_payload = {
                "sub": "1",
                "exp": int((datetime.utcnow() - timedelta(hours=1)).timestamp())
            }
            # This will have wrong signature but tests if expiry is checked
            return jwt.encode(expired_payload, "wrong_secret", algorithm="HS256")
        except:
            return None
    
    @pytest.mark.asyncio
    async def test_session_fixation(self, api_client: AsyncClient):
        """Test session fixation vulnerabilities"""
        # Create test agent
        agent_data = {
            "name": f"session_test_{fake.uuid4()[:8]}",
            "password": "secure_password_123",
            "role": "analyzer"
        }
        
        reg_response = await api_client.post("/api/v1/agents/register", json=agent_data)
        assert reg_response.status_code == 201
        
        # Login multiple times and check if tokens are different
        tokens = []
        for _ in range(5):
            login_response = await api_client.post(
                "/api/v1/agents/login",
                data={"username": agent_data["name"], "password": agent_data["password"]}
            )
            if login_response.status_code == 200:
                tokens.append(login_response.json()["access_token"])
        
        # Check if all tokens are unique
        if len(set(tokens)) != len(tokens):
            SecurityTestFramework.log_vulnerability(
                severity="medium",
                vuln_type="Session Fixation",
                endpoint="/api/v1/agents/login",
                payload=None,
                description="Same token returned for multiple logins",
                tokens_reused=len(tokens) - len(set(tokens))
            )
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self, api_client: AsyncClient):
        """Test brute force attack protection"""
        # Create test agent
        agent_data = {
            "name": f"brute_test_{fake.uuid4()[:8]}",
            "password": "secure_password_123",
            "role": "analyzer"
        }
        
        reg_response = await api_client.post("/api/v1/agents/register", json=agent_data)
        assert reg_response.status_code == 201
        
        # Attempt multiple failed logins
        failed_attempts = 0
        blocked = False
        
        for i in range(20):  # Try 20 failed login attempts
            response = await api_client.post(
                "/api/v1/agents/login",
                data={"username": agent_data["name"], "password": f"wrong_password_{i}"}
            )
            
            if response.status_code == 429:  # Rate limited
                blocked = True
                break
            elif response.status_code == 401:
                failed_attempts += 1
            
            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
        
        if not blocked and failed_attempts >= 15:
            SecurityTestFramework.log_vulnerability(
                severity="high",
                vuln_type="Missing Brute Force Protection",
                endpoint="/api/v1/agents/login",
                payload=None,
                description=f"No brute force protection after {failed_attempts} failed attempts",
                failed_attempts=failed_attempts
            )


@pytest.mark.security
class TestInjectionVulnerabilities:
    """Test various injection vulnerabilities"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_comprehensive(self, api_client: AsyncClient, admin_agent: dict, security_test_vectors: Dict):
        """Comprehensive SQL injection testing"""
        sql_payloads = security_test_vectors["sql_injection"]
        
        # Test SQL injection in various endpoints
        test_cases = [
            {
                "endpoint": "/api/v1/trials/",
                "method": "POST",
                "data_template": {
                    "name": "PAYLOAD",
                    "organism": "E. coli",
                    "genome_file": "/test",
                    "parameters": {"window_size": 1000, "min_confidence": 0.7, "trait_count": 2},
                    "created_by": admin_agent["agent"]["id"]
                },
                "inject_fields": ["name", "organism", "genome_file"]
            },
            {
                "endpoint": "/api/v1/results/",
                "method": "POST", 
                "data_template": {
                    "trial_id": 1,
                    "gene_id": "PAYLOAD",
                    "traits": ["metabolism"],
                    "confidence_scores": {"metabolism": 0.8},
                    "codon_usage_bias": {"TTT": 0.5, "TTC": 0.5}
                },
                "inject_fields": ["gene_id"]
            }
        ]
        
        for test_case in test_cases:
            for payload in sql_payloads:
                for field in test_case["inject_fields"]:
                    data = test_case["data_template"].copy()
                    data[field] = payload
                    
                    if test_case["method"] == "POST":
                        response = await api_client.post(
                            test_case["endpoint"],
                            json=data,
                            headers=admin_agent["headers"]
                        )
                    
                    # Check for SQL injection success indicators
                    if response.status_code == 201:
                        # Check response for SQL error messages
                        response_text = str(response.json())
                        sql_error_indicators = [
                            "sql", "syntax error", "mysql", "postgresql", "sqlite",
                            "ora-", "microsoft", "driver", "table", "column"
                        ]
                        
                        for indicator in sql_error_indicators:
                            if indicator in response_text.lower():
                                SecurityTestFramework.log_vulnerability(
                                    severity="critical",
                                    vuln_type="SQL Injection",
                                    endpoint=test_case["endpoint"],
                                    payload=payload,
                                    description=f"SQL injection possible in field '{field}'",
                                    field=field,
                                    sql_error_found=indicator
                                )
                                break
    
    @pytest.mark.asyncio
    async def test_nosql_injection(self, api_client: AsyncClient, admin_agent: dict, security_test_vectors: Dict):
        """Test NoSQL injection vulnerabilities"""
        nosql_payloads = security_test_vectors["nosql_injection"]
        
        # Test in JSON fields that might be passed to NoSQL queries
        for payload in nosql_payloads:
            trial_data = {
                "name": "NoSQL Test",
                "organism": "E. coli", 
                "genome_file": "/test",
                "parameters": {
                    "window_size": payload,  # Inject into parameters
                    "min_confidence": 0.7,
                    "trait_count": 2,
                    "custom_query": payload
                },
                "created_by": admin_agent["agent"]["id"]
            }
            
            response = await api_client.post(
                "/api/v1/trials/",
                json=trial_data,
                headers=admin_agent["headers"]
            )
            
            # Check for successful injection
            if response.status_code == 201:
                SecurityTestFramework.log_vulnerability(
                    severity="high",
                    vuln_type="NoSQL Injection",
                    endpoint="/api/v1/trials/",
                    payload=payload,
                    description="NoSQL injection payload accepted in parameters"
                )
    
    @pytest.mark.asyncio
    async def test_command_injection(self, api_client: AsyncClient, admin_agent: dict, security_test_vectors: Dict):
        """Test command injection vulnerabilities"""
        command_payloads = security_test_vectors["command_injection"]
        
        for payload in command_payloads:
            # Test in file path fields
            trial_data = {
                "name": "Command Injection Test",
                "organism": "E. coli",
                "genome_file": payload,  # Inject into file path
                "parameters": {
                    "window_size": 1000,
                    "min_confidence": 0.7,
                    "trait_count": 2
                },
                "created_by": admin_agent["agent"]["id"]
            }
            
            response = await api_client.post(
                "/api/v1/trials/",
                json=trial_data,
                headers=admin_agent["headers"]
            )
            
            # Look for command execution indicators in response
            if response.status_code == 201:
                response_data = response.json()
                command_indicators = ["uid=", "gid=", "total", "etc/passwd", "bin/"]
                
                for indicator in command_indicators:
                    if indicator in str(response_data):
                        SecurityTestFramework.log_vulnerability(
                            severity="critical",
                            vuln_type="Command Injection",
                            endpoint="/api/v1/trials/",
                            payload=payload,
                            description="Command injection possible in file path",
                            indicator_found=indicator
                        )


@pytest.mark.security
class TestInformationDisclosure:
    """Test information disclosure vulnerabilities"""
    
    @pytest.mark.asyncio
    async def test_sensitive_data_exposure(self, api_client: AsyncClient, admin_agent: dict):
        """Test for sensitive data exposure in API responses"""
        sensitive_patterns = [
            "password", "secret", "key", "token", "hash", "salt",
            "private", "confidential", "database_url", "connection_string"
        ]
        
        # Test various endpoints for sensitive data leaks
        endpoints_to_check = [
            "/api/v1/agents/me",
            "/api/v1/agents/",
            "/",
            "/health"
        ]
        
        for endpoint in endpoints_to_check:
            response = await api_client.get(endpoint, headers=admin_agent["headers"])
            
            if response.status_code == 200:
                leaks = SecurityTestFramework.check_response_for_leaks(
                    response.json() if response.headers.get("content-type", "").startswith("application/json") else {"content": response.text},
                    sensitive_patterns
                )
                
                if leaks:
                    SecurityTestFramework.log_vulnerability(
                        severity="medium",
                        vuln_type="Information Disclosure",
                        endpoint=endpoint,
                        payload=None,
                        description=f"Sensitive data patterns found: {leaks}",
                        patterns_found=leaks
                    )
    
    @pytest.mark.asyncio
    async def test_error_message_disclosure(self, api_client: AsyncClient, admin_agent: dict):
        """Test for information disclosure in error messages"""
        # Send malformed requests to trigger errors
        malformed_requests = [
            {
                "endpoint": "/api/v1/trials/",
                "data": {"invalid": "data structure"},
                "description": "Invalid JSON structure"
            },
            {
                "endpoint": "/api/v1/trials/99999",
                "method": "GET",
                "description": "Non-existent resource"
            },
            {
                "endpoint": "/api/v1/results/",
                "data": {
                    "trial_id": "not_an_integer",
                    "gene_id": "test",
                    "traits": [],
                    "confidence_scores": {},
                    "codon_usage_bias": {}
                },
                "description": "Type mismatch"
            }
        ]
        
        for req in malformed_requests:
            if req.get("method") == "GET":
                response = await api_client.get(req["endpoint"], headers=admin_agent["headers"])
            else:
                response = await api_client.post(
                    req["endpoint"],
                    json=req.get("data", {}),
                    headers=admin_agent["headers"]
                )
            
            if response.status_code >= 400:
                error_response = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"text": response.text}
                
                # Check for sensitive information in error messages
                sensitive_error_patterns = [
                    "database", "sql", "file path", "internal", "stack trace",
                    "line ", "function", "class", "module", "traceback"
                ]
                
                leaks = SecurityTestFramework.check_response_for_leaks(
                    error_response,
                    sensitive_error_patterns
                )
                
                if leaks:
                    SecurityTestFramework.log_vulnerability(
                        severity="low",
                        vuln_type="Information Disclosure - Error Messages",
                        endpoint=req["endpoint"],
                        payload=req.get("data"),
                        description=f"Error message contains sensitive info: {leaks}",
                        error_type=req["description"],
                        patterns_found=leaks
                    )
    
    @pytest.mark.asyncio
    async def test_directory_traversal(self, api_client: AsyncClient, admin_agent: dict, security_test_vectors: Dict):
        """Test directory traversal vulnerabilities"""
        path_traversal_payloads = security_test_vectors["path_traversal"]
        
        for payload in path_traversal_payloads:
            trial_data = {
                "name": "Directory Traversal Test",
                "organism": "E. coli",
                "genome_file": payload,  # Inject path traversal in file path
                "parameters": {
                    "window_size": 1000,
                    "min_confidence": 0.7,
                    "trait_count": 2
                },
                "created_by": admin_agent["agent"]["id"]
            }
            
            response = await api_client.post(
                "/api/v1/trials/",
                json=trial_data,
                headers=admin_agent["headers"]
            )
            
            # Check if path traversal was processed
            if response.status_code == 201:
                response_data = response.json()
                
                # Look for indicators of successful path traversal
                traversal_indicators = [
                    "root:", "bin/bash", "etc/passwd", "windows", "system32"
                ]
                
                for indicator in traversal_indicators:
                    if indicator in str(response_data):
                        SecurityTestFramework.log_vulnerability(
                            severity="high",
                            vuln_type="Directory Traversal",
                            endpoint="/api/v1/trials/",
                            payload=payload,
                            description="Directory traversal possible",
                            indicator_found=indicator
                        )


@pytest.mark.security
class TestAccessControlVulnerabilities:
    """Test access control and authorization vulnerabilities"""
    
    @pytest.mark.asyncio
    async def test_horizontal_privilege_escalation(self, api_client: AsyncClient):
        """Test horizontal privilege escalation"""
        # Create two test agents
        agent1_data = {
            "name": f"agent1_{fake.uuid4()[:8]}",
            "password": "password123",
            "role": "analyzer"
        }
        
        agent2_data = {
            "name": f"agent2_{fake.uuid4()[:8]}",
            "password": "password123", 
            "role": "analyzer"
        }
        
        # Register both agents
        reg1 = await api_client.post("/api/v1/agents/register", json=agent1_data)
        reg2 = await api_client.post("/api/v1/agents/register", json=agent2_data)
        
        assert reg1.status_code == 201
        assert reg2.status_code == 201
        
        agent1_id = reg1.json()["id"]
        agent2_id = reg2.json()["id"]
        
        # Login as agent1
        login1 = await api_client.post(
            "/api/v1/agents/login",
            data={"username": agent1_data["name"], "password": agent1_data["password"]}
        )
        assert login1.status_code == 200
        agent1_token = login1.json()["access_token"]
        agent1_headers = {"Authorization": f"Bearer {agent1_token}"}
        
        # Try to access agent2's information using agent1's token
        response = await api_client.get(f"/api/v1/agents/{agent2_id}", headers=agent1_headers)
        
        if response.status_code == 200:
            SecurityTestFramework.log_vulnerability(
                severity="high",
                vuln_type="Horizontal Privilege Escalation",
                endpoint=f"/api/v1/agents/{agent2_id}",
                payload=None,
                description="Agent can access other agent's information",
                accessing_agent_id=agent1_id,
                target_agent_id=agent2_id
            )
    
    @pytest.mark.asyncio
    async def test_vertical_privilege_escalation(self, api_client: AsyncClient):
        """Test vertical privilege escalation"""
        # Create a regular agent
        regular_agent_data = {
            "name": f"regular_{fake.uuid4()[:8]}",
            "password": "password123",
            "role": "analyzer"
        }
        
        reg_response = await api_client.post("/api/v1/agents/register", json=regular_agent_data)
        assert reg_response.status_code == 201
        
        # Login as regular agent
        login_response = await api_client.post(
            "/api/v1/agents/login",
            data={"username": regular_agent_data["name"], "password": regular_agent_data["password"]}
        )
        assert login_response.status_code == 200
        
        regular_token = login_response.json()["access_token"]
        regular_headers = {"Authorization": f"Bearer {regular_token}"}
        
        # Try to access coordinator-only endpoints
        coordinator_endpoints = [
            "/api/v1/agents/",  # List all agents
            "/api/v1/agents/1",  # Delete agent (if implemented)
        ]
        
        for endpoint in coordinator_endpoints:
            response = await api_client.get(endpoint, headers=regular_headers)
            
            if response.status_code == 200:
                SecurityTestFramework.log_vulnerability(
                    severity="critical",
                    vuln_type="Vertical Privilege Escalation",
                    endpoint=endpoint,
                    payload=None,
                    description="Regular agent can access coordinator-only endpoints",
                    agent_role="analyzer",
                    required_role="coordinator"
                )
    
    @pytest.mark.asyncio
    async def test_missing_authorization_checks(self, api_client: AsyncClient, admin_agent: dict):
        """Test for missing authorization checks"""
        # Create a trial with admin agent
        trial_data = {
            "name": "Authorization Test Trial",
            "organism": "E. coli",
            "genome_file": "/test",
            "parameters": {
                "window_size": 1000,
                "min_confidence": 0.7,
                "trait_count": 2
            },
            "created_by": admin_agent["agent"]["id"]
        }
        
        trial_response = await api_client.post(
            "/api/v1/trials/",
            json=trial_data,
            headers=admin_agent["headers"]
        )
        assert trial_response.status_code == 201
        trial_id = trial_response.json()["id"]
        
        # Try to access trial without authentication
        unauth_response = await api_client.get(f"/api/v1/trials/{trial_id}")
        
        if unauth_response.status_code == 200:
            SecurityTestFramework.log_vulnerability(
                severity="medium",
                vuln_type="Missing Authorization",
                endpoint=f"/api/v1/trials/{trial_id}",
                payload=None,
                description="Trial accessible without authentication",
                trial_id=trial_id
            )


def generate_security_report() -> Dict[str, Any]:
    """Generate comprehensive security vulnerability report"""
    total_findings = sum(len(findings) for findings in SECURITY_FINDINGS.values())
    
    severity_counts = {
        severity: len(findings)
        for severity, findings in SECURITY_FINDINGS.items()
    }
    
    report = {
        "security_test_timestamp": datetime.now().isoformat(),
        "memory_namespace": "swarm-regression-1752301224/security",
        "summary": {
            "total_vulnerabilities": total_findings,
            "severity_breakdown": severity_counts,
            "risk_score": _calculate_risk_score(severity_counts)
        },
        "vulnerabilities_by_category": _categorize_vulnerabilities(),
        "detailed_findings": SECURITY_FINDINGS,
        "recommendations": _generate_security_recommendations(),
        "compliance_status": _check_security_compliance()
    }
    
    return report


def _calculate_risk_score(severity_counts: Dict[str, int]) -> float:
    """Calculate overall risk score based on vulnerability counts"""
    weights = {"critical": 10, "high": 7, "medium": 4, "low": 2, "info": 1}
    
    total_score = sum(
        weights.get(severity, 0) * count
        for severity, count in severity_counts.items()
    )
    
    # Normalize to 0-100 scale
    max_possible = 100  # Assuming reasonable maximum
    return min(100, (total_score / max_possible) * 100)


def _categorize_vulnerabilities() -> Dict[str, int]:
    """Categorize vulnerabilities by type"""
    categories = {}
    
    for findings in SECURITY_FINDINGS.values():
        for finding in findings:
            vuln_type = finding.get("type", "Unknown")
            categories[vuln_type] = categories.get(vuln_type, 0) + 1
    
    return categories


def _generate_security_recommendations() -> List[str]:
    """Generate security recommendations based on findings"""
    recommendations = []
    
    # Check for critical vulnerabilities
    if SECURITY_FINDINGS["critical"]:
        recommendations.append("URGENT: Address critical vulnerabilities immediately")
        recommendations.append("Implement emergency security patches")
        
    # Check for injection vulnerabilities
    injection_types = ["SQL Injection", "NoSQL Injection", "Command Injection"]
    if any(finding["type"] in injection_types for findings in SECURITY_FINDINGS.values() for finding in findings):
        recommendations.append("Implement input sanitization and parameterized queries")
        recommendations.append("Add input validation middleware")
        
    # Check for authentication issues
    auth_types = ["JWT Token Manipulation", "Weak Password Policy", "Missing Brute Force Protection"]
    if any(finding["type"] in auth_types for findings in SECURITY_FINDINGS.values() for finding in findings):
        recommendations.append("Strengthen authentication mechanisms")
        recommendations.append("Implement account lockout policies")
        recommendations.append("Add multi-factor authentication")
        
    # Check for authorization issues
    authz_types = ["Horizontal Privilege Escalation", "Vertical Privilege Escalation", "Missing Authorization"]
    if any(finding["type"] in authz_types for findings in SECURITY_FINDINGS.values() for finding in findings):
        recommendations.append("Implement role-based access controls")
        recommendations.append("Add authorization checks to all endpoints")
        
    # General recommendations
    recommendations.extend([
        "Implement security headers (HSTS, CSP, etc.)",
        "Add comprehensive logging and monitoring",
        "Conduct regular security testing",
        "Implement rate limiting and DDoS protection",
        "Use HTTPS for all communications",
        "Regular security training for development team"
    ])
    
    return recommendations


def _check_security_compliance() -> Dict[str, bool]:
    """Check compliance with security standards"""
    return {
        "owasp_top_10_compliant": len(SECURITY_FINDINGS["critical"]) == 0,
        "pci_dss_compliant": False,  # Would need specific checks
        "hipaa_compliant": False,    # Would need specific checks
        "gdpr_compliant": False,     # Would need privacy checks
        "basic_security_practices": len(SECURITY_FINDINGS["high"]) < 3
    }


@pytest.fixture(scope="session", autouse=True)
def generate_security_report_fixture():
    """Generate security report after all tests complete"""
    yield
    
    # Generate and save report
    report = generate_security_report()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/home/murr2k/projects/agentic/pleiotropy/tests/regression/security_vulnerability_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n🔒 Security Vulnerability Report Generated: {report_file}")
    print(f"🚨 Total Vulnerabilities: {report['summary']['total_vulnerabilities']}")
    print(f"📊 Risk Score: {report['summary']['risk_score']:.1f}/100")
    
    if report['summary']['severity_breakdown']['critical'] > 0:
        print(f"⚠️  CRITICAL: {report['summary']['severity_breakdown']['critical']} critical vulnerabilities found!")