"""
Pytest configuration for API regression tests
"""

import pytest
import asyncio
import os
import sys
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

# Add the API app to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../trial_database/api'))

from app.main import app
from app.db.database import Base, get_db
from app.core.config import settings


# Test database URL (use SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
async def override_get_db(test_db_session: AsyncSession):
    """Override the get_db dependency"""
    def _override_get_db():
        return test_db_session
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def api_client(override_get_db) -> AsyncGenerator[AsyncClient, None]:
    """Create HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture(scope="session")
def security_test_vectors():
    """Security test vectors for various attack types"""
    return {
        "sql_injection": [
            "'; DROP TABLE agents; --",
            "' OR '1'='1",
            "'; INSERT INTO agents (name) VALUES ('hacker'); --",
            "' UNION SELECT * FROM agents --",
            "1'; UPDATE agents SET role='coordinator' WHERE id=1; --",
            "admin'/**/UNION/**/SELECT/**/password/**/FROM/**/agents--",
            "' AND (SELECT COUNT(*) FROM agents) > 0 --"
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'; alert('xss'); //",
            "<iframe src='javascript:alert(\"xss\")'></iframe>",
            "<svg onload=alert('xss')>",
            "<body onload=alert('xss')>",
            "{{constructor.constructor('alert(1)')()}}"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ],
        "command_injection": [
            "; ls -la",
            "| whoami",
            "& ping -c 1 127.0.0.1",
            "`id`",
            "$(whoami)",
            "&& cat /etc/passwd",
            "|| echo vulnerable"
        ],
        "ldap_injection": [
            "*)(uid=*",
            "*)(|(uid=*))",
            "admin)(|(password=*))",
            "*))%00",
            ")(cn=*"
        ],
        "nosql_injection": [
            "true, true",
            "', '$where': 'true",
            "1; return true",
            "{\"$ne\": null}",
            "{\"$regex\": \".*\"}"
        ]
    }


@pytest.fixture(scope="session")
def performance_thresholds():
    """Performance testing thresholds"""
    return {
        "response_time": {
            "fast": 0.1,      # 100ms
            "acceptable": 0.5, # 500ms
            "slow": 2.0       # 2 seconds
        },
        "throughput": {
            "min_requests_per_second": 10,
            "target_requests_per_second": 100
        },
        "concurrent_connections": {
            "websocket_max": 100,
            "http_max": 200
        },
        "memory_usage": {
            "max_mb_per_request": 50,
            "max_total_mb": 500
        }
    }


@pytest.fixture(scope="session")
def load_test_data():
    """Generate test data for load testing"""
    return {
        "agents": [
            {
                "name": f"load_test_agent_{i}",
                "password": f"password_{i}_123",
                "role": "analyzer" if i % 2 == 0 else "validator",
                "capabilities": ["analysis", "validation"]
            }
            for i in range(50)
        ],
        "trials": [
            {
                "name": f"Load Test Trial {i}",
                "organism": "E. coli" if i % 3 == 0 else "S. cerevisiae",
                "genome_file": f"/data/load_test_{i}.fasta",
                "parameters": {
                    "window_size": 1000 + (i * 100),
                    "min_confidence": 0.7 + (i % 10) / 100,
                    "trait_count": 3 + (i % 5)
                }
            }
            for i in range(100)
        ]
    }