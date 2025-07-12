"""
Performance Benchmarking Test Suite
Load testing and performance analysis for the Genomic Pleiotropy API

Memory Namespace: swarm-regression-1752301224/performance
"""

import pytest
import asyncio
import time
import statistics
import psutil
import json
from typing import Dict, List, Any, Tuple
from httpx import AsyncClient
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import websockets

# Performance metrics storage
PERFORMANCE_METRICS = {
    "response_times": {},
    "throughput": {},
    "resource_usage": {},
    "error_rates": {},
    "websocket_performance": {}
}


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    @staticmethod
    def measure_time(func):
        """Decorator to measure function execution time"""
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return result, execution_time
        return wrapper
    
    @staticmethod
    def calculate_percentiles(times: List[float]) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not times:
            return {}
        
        return {
            "p50": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
            "min": min(times),
            "max": max(times),
            "avg": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    @staticmethod
    def get_system_metrics() -> Dict[str, float]:
        """Get current system resource metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_io_read": psutil.disk_io_counters().read_bytes / 1024 / 1024,
            "disk_io_write": psutil.disk_io_counters().write_bytes / 1024 / 1024,
            "network_sent": psutil.net_io_counters().bytes_sent / 1024 / 1024,
            "network_recv": psutil.net_io_counters().bytes_recv / 1024 / 1024
        }


@pytest.mark.performance
class TestAPIResponseTimes:
    """Test API endpoint response times"""
    
    @pytest.mark.asyncio
    async def test_basic_endpoint_response_times(self, api_client: AsyncClient, admin_agent: dict):
        """Test response times for basic endpoints"""
        endpoints = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/api/v1/agents/me", "GET"),
            ("/api/v1/trials/", "GET"),
            ("/api/v1/results/", "GET"),
            ("/api/v1/progress/active", "GET")
        ]
        
        for endpoint, method in endpoints:
            times = []
            errors = 0
            
            # Run 10 requests to get average response time
            for _ in range(10):
                start_time = time.perf_counter()
                
                if method == "GET":
                    if endpoint in ["/api/v1/agents/me", "/api/v1/trials/", "/api/v1/results/"]:
                        response = await api_client.get(endpoint, headers=admin_agent["headers"])
                    else:
                        response = await api_client.get(endpoint)
                
                end_time = time.perf_counter()
                response_time = end_time - start_time
                
                if response.status_code < 400:
                    times.append(response_time)
                else:
                    errors += 1
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            if times:
                metrics = PerformanceBenchmark.calculate_percentiles(times)
                PERFORMANCE_METRICS["response_times"][f"{method} {endpoint}"] = {
                    "metrics": metrics,
                    "error_rate": errors / (len(times) + errors),
                    "total_requests": len(times) + errors
                }
    
    @pytest.mark.asyncio
    async def test_crud_operation_performance(self, api_client: AsyncClient, admin_agent: dict):
        """Test CRUD operation performance"""
        
        # Test trial CRUD operations
        crud_times = {
            "create": [],
            "read": [],
            "update": [],
            "delete": []
        }
        
        trial_ids = []
        
        # Create operations
        for i in range(5):
            trial_data = {
                "name": f"Performance Test Trial {i}",
                "organism": "E. coli",
                "genome_file": f"/data/perf_test_{i}.fasta",
                "parameters": {
                    "window_size": 1000,
                    "min_confidence": 0.7,
                    "trait_count": 3
                },
                "created_by": admin_agent["agent"]["id"]
            }
            
            start_time = time.perf_counter()
            response = await api_client.post(
                "/api/v1/trials/",
                json=trial_data,
                headers=admin_agent["headers"]
            )
            end_time = time.perf_counter()
            
            if response.status_code == 201:
                crud_times["create"].append(end_time - start_time)
                trial_ids.append(response.json()["id"])
        
        # Read operations
        for trial_id in trial_ids:
            start_time = time.perf_counter()
            response = await api_client.get(f"/api/v1/trials/{trial_id}")
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                crud_times["read"].append(end_time - start_time)
        
        # Update operations
        for trial_id in trial_ids:
            update_data = {"description": f"Updated description for performance test {trial_id}"}
            
            start_time = time.perf_counter()
            response = await api_client.patch(
                f"/api/v1/trials/{trial_id}",
                json=update_data,
                headers=admin_agent["headers"]
            )
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                crud_times["update"].append(end_time - start_time)
        
        # Delete operations (only for pending trials)
        for trial_id in trial_ids:
            start_time = time.perf_counter()
            response = await api_client.delete(
                f"/api/v1/trials/{trial_id}",
                headers=admin_agent["headers"]
            )
            end_time = time.perf_counter()
            
            if response.status_code == 204:
                crud_times["delete"].append(end_time - start_time)
        
        # Store CRUD performance metrics
        for operation, times in crud_times.items():
            if times:
                PERFORMANCE_METRICS["response_times"][f"trial_{operation}"] = {
                    "metrics": PerformanceBenchmark.calculate_percentiles(times),
                    "operation_type": "crud"
                }


@pytest.mark.performance 
class TestThroughputAndLoad:
    """Test API throughput and load handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, api_client: AsyncClient, admin_agent: dict):
        """Test concurrent request handling"""
        concurrent_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrent_levels:
            start_time = time.perf_counter()
            tasks = []
            
            # Create concurrent requests
            for i in range(concurrency):
                task = api_client.get("/health")
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            successful_responses = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
            throughput = successful_responses / total_time if total_time > 0 else 0
            
            PERFORMANCE_METRICS["throughput"][f"concurrent_{concurrency}"] = {
                "requests_per_second": throughput,
                "total_requests": concurrency,
                "successful_requests": successful_responses,
                "total_time": total_time,
                "error_rate": (concurrency - successful_responses) / concurrency
            }
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, api_client: AsyncClient, admin_agent: dict):
        """Test sustained load over time"""
        duration_seconds = 30  # 30 second test
        request_interval = 0.1  # 10 requests per second target
        
        start_time = time.perf_counter()
        request_times = []
        errors = 0
        total_requests = 0
        
        while time.perf_counter() - start_time < duration_seconds:
            request_start = time.perf_counter()
            
            try:
                response = await api_client.get("/health")
                request_end = time.perf_counter()
                
                request_times.append(request_end - request_start)
                
                if response.status_code != 200:
                    errors += 1
                    
                total_requests += 1
                
            except Exception:
                errors += 1
                total_requests += 1
            
            # Wait for next request (if needed)
            elapsed = time.perf_counter() - request_start
            if elapsed < request_interval:
                await asyncio.sleep(request_interval - elapsed)
        
        actual_duration = time.perf_counter() - start_time
        actual_throughput = total_requests / actual_duration
        
        PERFORMANCE_METRICS["throughput"]["sustained_load"] = {
            "duration_seconds": actual_duration,
            "total_requests": total_requests,
            "requests_per_second": actual_throughput,
            "error_rate": errors / total_requests if total_requests > 0 else 0,
            "response_time_metrics": PerformanceBenchmark.calculate_percentiles(request_times)
        }
    
    @pytest.mark.asyncio
    async def test_batch_operation_performance(self, api_client: AsyncClient, admin_agent: dict):
        """Test batch operation performance vs individual requests"""
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            # Test batch creation
            batch_data = {
                "operation": "create",
                "items": []
            }
            
            for i in range(batch_size):
                trial_item = {
                    "name": f"Batch Perf Test {i}",
                    "organism": "E. coli",
                    "genome_file": f"/data/batch_perf_{i}.fasta",
                    "parameters": {
                        "window_size": 1000,
                        "min_confidence": 0.7,
                        "trait_count": 2
                    },
                    "created_by": admin_agent["agent"]["id"]
                }
                batch_data["items"].append(trial_item)
            
            # Time batch operation
            start_time = time.perf_counter()
            response = await api_client.post(
                "/api/v1/trials/batch",
                json=batch_data,
                headers=admin_agent["headers"]
            )
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                throughput = result["success_count"] / batch_time if batch_time > 0 else 0
                
                PERFORMANCE_METRICS["throughput"][f"batch_size_{batch_size}"] = {
                    "batch_size": batch_size,
                    "execution_time": batch_time,
                    "items_per_second": throughput,
                    "success_count": result["success_count"],
                    "error_count": result["error_count"]
                }


@pytest.mark.performance
class TestWebSocketPerformance:
    """Test WebSocket performance and load"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_performance(self):
        """Test WebSocket connection establishment performance"""
        connection_times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            
            try:
                uri = f"ws://testserver/ws/connect?client_id=perf_test_{i}&agent_name=perf_agent"
                websocket = await websockets.connect(uri)
                
                # Send a ping to verify connection
                await websocket.send(json.dumps({"type": "ping"}))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                
                end_time = time.perf_counter()
                connection_times.append(end_time - start_time)
                
                await websocket.close()
                
            except Exception as e:
                # Connection failed
                pass
        
        if connection_times:
            PERFORMANCE_METRICS["websocket_performance"]["connection_establishment"] = {
                "metrics": PerformanceBenchmark.calculate_percentiles(connection_times),
                "successful_connections": len(connection_times)
            }
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(self):
        """Test WebSocket concurrent connection handling"""
        concurrent_levels = [5, 10, 20]
        
        for concurrency in concurrent_levels:
            connections = []
            connection_times = []
            successful_connections = 0
            
            # Establish concurrent connections
            start_time = time.perf_counter()
            
            for i in range(concurrency):
                try:
                    uri = f"ws://testserver/ws/connect?client_id=concurrent_{i}&agent_name=concurrent_agent_{i}"
                    websocket = await websockets.connect(uri)
                    connections.append(websocket)
                    successful_connections += 1
                except Exception:
                    pass
            
            connection_time = time.perf_counter() - start_time
            
            # Test message throughput
            message_times = []
            
            if connections:
                start_time = time.perf_counter()
                
                # Send messages from all connections
                tasks = []
                for i, websocket in enumerate(connections):
                    message = {"type": "ping", "client_id": f"concurrent_{i}"}
                    task = websocket.send(json.dumps(message))
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                message_time = time.perf_counter() - start_time
                message_throughput = len(connections) / message_time if message_time > 0 else 0
                
                PERFORMANCE_METRICS["websocket_performance"][f"concurrent_{concurrency}"] = {
                    "target_connections": concurrency,
                    "successful_connections": successful_connections,
                    "connection_establishment_time": connection_time,
                    "message_throughput": message_throughput,
                    "success_rate": successful_connections / concurrency
                }
            
            # Clean up connections
            for websocket in connections:
                try:
                    await websocket.close()
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_websocket_message_latency(self):
        """Test WebSocket message latency"""
        try:
            uri = "ws://testserver/ws/connect?client_id=latency_test&agent_name=latency_agent"
            websocket = await websockets.connect(uri)
            
            latencies = []
            
            # Test message round-trip times
            for i in range(20):
                start_time = time.perf_counter()
                
                await websocket.send(json.dumps({"type": "ping"}))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                
                end_time = time.perf_counter()
                latency = end_time - start_time
                latencies.append(latency)
                
                await asyncio.sleep(0.1)  # Small delay between messages
            
            await websocket.close()
            
            PERFORMANCE_METRICS["websocket_performance"]["message_latency"] = {
                "metrics": PerformanceBenchmark.calculate_percentiles(latencies),
                "message_count": len(latencies)
            }
            
        except Exception as e:
            PERFORMANCE_METRICS["websocket_performance"]["message_latency"] = {
                "error": str(e),
                "message_count": 0
            }


@pytest.mark.performance
class TestResourceUsage:
    """Test system resource usage under load"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, api_client: AsyncClient, admin_agent: dict):
        """Test memory usage during high load"""
        initial_metrics = PerformanceBenchmark.get_system_metrics()
        
        # Create load by making many requests
        tasks = []
        for i in range(100):
            task = api_client.get("/health")
            tasks.append(task)
        
        # Monitor resource usage during load
        load_start = time.perf_counter()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        load_end = time.perf_counter()
        
        final_metrics = PerformanceBenchmark.get_system_metrics()
        
        # Calculate resource usage delta
        memory_delta = final_metrics["memory_mb"] - initial_metrics["memory_mb"]
        cpu_avg = (initial_metrics["cpu_percent"] + final_metrics["cpu_percent"]) / 2
        
        successful_requests = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        
        PERFORMANCE_METRICS["resource_usage"]["load_test"] = {
            "initial_memory_mb": initial_metrics["memory_mb"],
            "final_memory_mb": final_metrics["memory_mb"],
            "memory_delta_mb": memory_delta,
            "cpu_usage_percent": cpu_avg,
            "load_duration": load_end - load_start,
            "requests_processed": successful_requests,
            "memory_per_request_mb": memory_delta / successful_requests if successful_requests > 0 else 0
        }
    
    @pytest.mark.asyncio
    async def test_database_performance(self, api_client: AsyncClient, admin_agent: dict):
        """Test database operation performance"""
        db_operation_times = {
            "create": [],
            "read": [],
            "query": []
        }
        
        # Test database create operations
        trial_ids = []
        for i in range(20):
            trial_data = {
                "name": f"DB Perf Test {i}",
                "organism": "E. coli",
                "genome_file": f"/data/db_test_{i}.fasta",
                "parameters": {
                    "window_size": 1000,
                    "min_confidence": 0.7,
                    "trait_count": 2
                },
                "created_by": admin_agent["agent"]["id"]
            }
            
            start_time = time.perf_counter()
            response = await api_client.post(
                "/api/v1/trials/",
                json=trial_data,
                headers=admin_agent["headers"]
            )
            end_time = time.perf_counter()
            
            if response.status_code == 201:
                db_operation_times["create"].append(end_time - start_time)
                trial_ids.append(response.json()["id"])
        
        # Test database read operations
        for trial_id in trial_ids:
            start_time = time.perf_counter()
            response = await api_client.get(f"/api/v1/trials/{trial_id}")
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                db_operation_times["read"].append(end_time - start_time)
        
        # Test database query operations (with filters)
        start_time = time.perf_counter()
        response = await api_client.get("/api/v1/trials/?organism=E.%20coli&page_size=50")
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            db_operation_times["query"].append(end_time - start_time)
        
        # Store database performance metrics
        for operation, times in db_operation_times.items():
            if times:
                PERFORMANCE_METRICS["resource_usage"][f"database_{operation}"] = {
                    "metrics": PerformanceBenchmark.calculate_percentiles(times),
                    "operation_count": len(times)
                }


def generate_performance_report() -> Dict[str, Any]:
    """Generate comprehensive performance benchmark report"""
    report = {
        "performance_test_timestamp": datetime.now().isoformat(),
        "memory_namespace": "swarm-regression-1752301224/performance",
        "summary": _generate_performance_summary(),
        "detailed_metrics": PERFORMANCE_METRICS,
        "performance_analysis": _analyze_performance(),
        "recommendations": _generate_performance_recommendations(),
        "sla_compliance": _check_sla_compliance()
    }
    
    return report


def _generate_performance_summary() -> Dict[str, Any]:
    """Generate performance summary statistics"""
    summary = {
        "total_tests": 0,
        "endpoints_tested": 0,
        "average_response_time": 0,
        "max_throughput": 0,
        "websocket_tests": 0
    }
    
    # Count tests and calculate averages
    response_times = PERFORMANCE_METRICS.get("response_times", {})
    throughput_metrics = PERFORMANCE_METRICS.get("throughput", {})
    websocket_metrics = PERFORMANCE_METRICS.get("websocket_performance", {})
    
    summary["endpoints_tested"] = len(response_times)
    summary["total_tests"] = len(response_times) + len(throughput_metrics) + len(websocket_metrics)
    summary["websocket_tests"] = len(websocket_metrics)
    
    # Calculate average response time
    all_avg_times = []
    for endpoint_data in response_times.values():
        if "metrics" in endpoint_data and "avg" in endpoint_data["metrics"]:
            all_avg_times.append(endpoint_data["metrics"]["avg"])
    
    if all_avg_times:
        summary["average_response_time"] = statistics.mean(all_avg_times)
    
    # Find max throughput
    max_throughput = 0
    for throughput_data in throughput_metrics.values():
        if "requests_per_second" in throughput_data:
            max_throughput = max(max_throughput, throughput_data["requests_per_second"])
    summary["max_throughput"] = max_throughput
    
    return summary


def _analyze_performance() -> Dict[str, Any]:
    """Analyze performance metrics and identify issues"""
    analysis = {
        "slow_endpoints": [],
        "high_error_rates": [],
        "resource_issues": [],
        "scaling_bottlenecks": []
    }
    
    # Identify slow endpoints (> 1 second average)
    for endpoint, data in PERFORMANCE_METRICS.get("response_times", {}).items():
        if "metrics" in data and data["metrics"].get("avg", 0) > 1.0:
            analysis["slow_endpoints"].append({
                "endpoint": endpoint,
                "avg_time": data["metrics"]["avg"],
                "p95_time": data["metrics"].get("p95", 0)
            })
    
    # Identify high error rates (> 5%)
    for endpoint, data in PERFORMANCE_METRICS.get("response_times", {}).items():
        if data.get("error_rate", 0) > 0.05:
            analysis["high_error_rates"].append({
                "endpoint": endpoint,
                "error_rate": data["error_rate"],
                "total_requests": data.get("total_requests", 0)
            })
    
    # Check resource usage
    resource_data = PERFORMANCE_METRICS.get("resource_usage", {})
    if resource_data:
        for test, data in resource_data.items():
            if data.get("memory_per_request_mb", 0) > 10:  # More than 10MB per request
                analysis["resource_issues"].append({
                    "test": test,
                    "memory_per_request": data["memory_per_request_mb"],
                    "issue": "High memory usage per request"
                })
    
    # Check scaling bottlenecks
    throughput_data = PERFORMANCE_METRICS.get("throughput", {})
    concurrent_results = {k: v for k, v in throughput_data.items() if "concurrent_" in k}
    
    if len(concurrent_results) >= 2:
        # Check if throughput scales linearly
        sorted_results = sorted(concurrent_results.items(), key=lambda x: int(x[0].split('_')[1]))
        
        for i in range(1, len(sorted_results)):
            prev_name, prev_data = sorted_results[i-1]
            curr_name, curr_data = sorted_results[i]
            
            prev_rps = prev_data.get("requests_per_second", 0)
            curr_rps = curr_data.get("requests_per_second", 0)
            
            # If throughput doesn't increase with concurrency, it's a bottleneck
            if curr_rps <= prev_rps * 1.1:  # Allow 10% variance
                analysis["scaling_bottlenecks"].append({
                    "concurrency_level": curr_name,
                    "throughput": curr_rps,
                    "previous_throughput": prev_rps,
                    "issue": "Throughput not scaling with concurrency"
                })
    
    return analysis


def _generate_performance_recommendations() -> List[str]:
    """Generate performance optimization recommendations"""
    recommendations = []
    analysis = _analyze_performance()
    
    if analysis["slow_endpoints"]:
        recommendations.append("Optimize slow endpoints with caching and database query optimization")
        recommendations.append("Consider implementing response compression")
        
    if analysis["high_error_rates"]:
        recommendations.append("Investigate and fix endpoints with high error rates")
        recommendations.append("Add better error handling and circuit breakers")
        
    if analysis["resource_issues"]:
        recommendations.append("Optimize memory usage - consider object pooling")
        recommendations.append("Implement garbage collection tuning")
        
    if analysis["scaling_bottlenecks"]:
        recommendations.append("Identify and resolve scaling bottlenecks")
        recommendations.append("Consider horizontal scaling or load balancing")
        
    # General recommendations
    recommendations.extend([
        "Implement response caching for frequently accessed data",
        "Add database connection pooling",
        "Consider using CDN for static assets",
        "Implement request rate limiting",
        "Add performance monitoring and alerting",
        "Regular performance testing in CI/CD pipeline"
    ])
    
    return recommendations


def _check_sla_compliance() -> Dict[str, bool]:
    """Check compliance with SLA requirements"""
    sla_requirements = {
        "response_time_95p_under_500ms": True,
        "throughput_above_100_rps": True,
        "error_rate_under_1_percent": True,
        "availability_above_99_percent": True
    }
    
    # Check 95th percentile response times
    response_times = PERFORMANCE_METRICS.get("response_times", {})
    for endpoint_data in response_times.values():
        if "metrics" in endpoint_data:
            p95_time = endpoint_data["metrics"].get("p95", 0)
            if p95_time > 0.5:  # 500ms
                sla_requirements["response_time_95p_under_500ms"] = False
                break
    
    # Check throughput
    throughput_metrics = PERFORMANCE_METRICS.get("throughput", {})
    max_throughput = 0
    for data in throughput_metrics.values():
        rps = data.get("requests_per_second", 0)
        max_throughput = max(max_throughput, rps)
    
    if max_throughput < 100:
        sla_requirements["throughput_above_100_rps"] = False
    
    # Check error rates
    for endpoint_data in response_times.values():
        error_rate = endpoint_data.get("error_rate", 0)
        if error_rate > 0.01:  # 1%
            sla_requirements["error_rate_under_1_percent"] = False
            break
    
    return sla_requirements


@pytest.fixture(scope="session", autouse=True)
def generate_performance_report_fixture():
    """Generate performance report after all tests complete"""
    yield
    
    # Generate and save report
    report = generate_performance_report()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/home/murr2k/projects/agentic/pleiotropy/tests/regression/performance_benchmark_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n⚡ Performance Benchmark Report Generated: {report_file}")
    print(f"📊 Tests Completed: {report['summary']['total_tests']}")
    print(f"🚀 Max Throughput: {report['summary']['max_throughput']:.1f} RPS")
    print(f"⏱️  Avg Response Time: {report['summary']['average_response_time']:.3f}s")
    
    # Check SLA compliance
    sla_failures = [k for k, v in report["sla_compliance"].items() if not v]
    if sla_failures:
        print(f"⚠️  SLA Violations: {', '.join(sla_failures)}")
    else:
        print("✅ All SLA requirements met")