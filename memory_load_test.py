#!/usr/bin/env python3
"""
Memory System Load Testing for Multi-Agent Coordination

Tests Redis-based memory system under concurrent agent loads.
"""

import asyncio
import json
import time
import random
import threading
import concurrent.futures
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import redis
import pandas as pd
import numpy as np

@dataclass
class AgentMemoryEntry:
    """Memory entry for agent coordination."""
    agent_id: str
    task_id: str
    data_type: str
    payload: Dict[str, Any]
    timestamp: float
    ttl: int = 3600

@dataclass
class LoadTestResult:
    """Results from memory load testing."""
    test_name: str
    concurrent_agents: int
    operations_per_agent: int
    total_operations: int
    execution_time: float
    operations_per_second: float
    success_rate: float
    average_latency: float
    max_latency: float
    memory_usage_mb: float
    errors: List[str]

class MemoryLoadTester:
    """Load tester for Redis-based memory system."""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        """Initialize the load tester."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.results = []
        
    def get_redis_connection(self):
        """Get a Redis connection."""
        return redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    
    def simulate_agent_memory_operations(self, agent_id: str, operations: int) -> Dict[str, Any]:
        """Simulate memory operations for a single agent."""
        r = self.get_redis_connection()
        
        operation_times = []
        errors = []
        successful_ops = 0
        
        for i in range(operations):
            start_time = time.time()
            
            try:
                # Create test data
                entry = AgentMemoryEntry(
                    agent_id=agent_id,
                    task_id=f"task_{i}",
                    data_type=random.choice(['genomic_analysis', 'trait_extraction', 'visualization', 'coordination']),
                    payload={
                        'gene_id': f"gene_{random.randint(1, 1000)}",
                        'traits': random.sample(['metabolism', 'stress', 'motility', 'regulation'], 
                                               random.randint(1, 3)),
                        'confidence': random.uniform(0.5, 1.0),
                        'processing_time': random.uniform(0.1, 5.0)
                    },
                    timestamp=time.time()
                )
                
                # Set operation
                key = f"agent:{agent_id}:task:{i}"
                r.setex(key, entry.ttl, json.dumps(asdict(entry)))
                
                # Get operation
                retrieved = r.get(key)
                if retrieved:
                    parsed = json.loads(retrieved)
                    assert parsed['agent_id'] == agent_id
                
                # Pub/Sub operation
                channel = f"agent_coordination:{agent_id}"
                r.publish(channel, json.dumps({
                    'type': 'task_update',
                    'agent_id': agent_id,
                    'task_id': f"task_{i}",
                    'status': 'completed'
                }))
                
                # List operations
                list_key = f"agent:{agent_id}:completed_tasks"
                r.lpush(list_key, f"task_{i}")
                r.expire(list_key, 3600)
                
                # Hash operations for agent status
                status_key = f"agent:{agent_id}:status"
                r.hset(status_key, mapping={
                    'last_seen': time.time(),
                    'tasks_completed': i + 1,
                    'current_load': random.uniform(0.1, 0.9)
                })
                r.expire(status_key, 3600)
                
                successful_ops += 1
                
            except Exception as e:
                errors.append(f"Operation {i}: {str(e)}")
            
            operation_times.append(time.time() - start_time)
        
        return {
            'agent_id': agent_id,
            'successful_operations': successful_ops,
            'total_operations': operations,
            'operation_times': operation_times,
            'errors': errors
        }
    
    def run_concurrent_load_test(self, num_agents: int, operations_per_agent: int) -> LoadTestResult:
        """Run concurrent load test with multiple agents."""
        print(f"🧪 Running concurrent load test: {num_agents} agents, {operations_per_agent} ops each")
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
            # Submit all agent tasks
            futures = []
            for i in range(num_agents):
                agent_id = f"test_agent_{i}"
                future = executor.submit(self.simulate_agent_memory_operations, agent_id, operations_per_agent)
                futures.append(future)
            
            # Collect results
            agent_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    agent_results.append(result)
                except Exception as e:
                    agent_results.append({
                        'agent_id': 'unknown',
                        'successful_operations': 0,
                        'total_operations': operations_per_agent,
                        'operation_times': [],
                        'errors': [f"Agent execution failed: {str(e)}"]
                    })
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Aggregate results
        total_successful = sum(r['successful_operations'] for r in agent_results)
        total_operations = num_agents * operations_per_agent
        all_operation_times = []
        all_errors = []
        
        for result in agent_results:
            all_operation_times.extend(result['operation_times'])
            all_errors.extend(result['errors'])
        
        # Calculate metrics
        success_rate = total_successful / total_operations if total_operations > 0 else 0
        operations_per_second = total_successful / execution_time if execution_time > 0 else 0
        average_latency = np.mean(all_operation_times) if all_operation_times else 0
        max_latency = max(all_operation_times) if all_operation_times else 0
        
        # Get Redis memory usage
        r = self.get_redis_connection()
        memory_info = r.info('memory')
        memory_usage_mb = memory_info.get('used_memory', 0) / (1024 * 1024)
        
        return LoadTestResult(
            test_name=f"concurrent_agents_{num_agents}",
            concurrent_agents=num_agents,
            operations_per_agent=operations_per_agent,
            total_operations=total_operations,
            execution_time=execution_time,
            operations_per_second=operations_per_second,
            success_rate=success_rate,
            average_latency=average_latency,
            max_latency=max_latency,
            memory_usage_mb=memory_usage_mb,
            errors=all_errors[:10]  # Keep only first 10 errors
        )
    
    def test_pubsub_coordination(self, num_publishers: int, num_subscribers: int, messages_per_publisher: int) -> Dict[str, Any]:
        """Test pub/sub coordination system."""
        print(f"📡 Testing pub/sub coordination: {num_publishers} publishers, {num_subscribers} subscribers")
        
        received_messages = {}
        
        def subscriber_worker(subscriber_id: str, channels: List[str]):
            """Worker function for subscribers."""
            r = self.get_redis_connection()
            pubsub = r.pubsub()
            
            for channel in channels:
                pubsub.subscribe(channel)
            
            received_messages[subscriber_id] = []
            
            # Listen for messages with timeout
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                try:
                    message = pubsub.get_message(timeout=1)
                    if message and message['type'] == 'message':
                        received_messages[subscriber_id].append({
                            'channel': message['channel'],
                            'data': message['data'],
                            'received_at': time.time()
                        })
                except Exception as e:
                    break
            
            pubsub.close()
        
        def publisher_worker(publisher_id: str, channel: str, num_messages: int):
            """Worker function for publishers."""
            r = self.get_redis_connection()
            
            for i in range(num_messages):
                message = {
                    'publisher_id': publisher_id,
                    'message_id': i,
                    'data': f"test_data_{i}",
                    'timestamp': time.time()
                }
                r.publish(channel, json.dumps(message))
                time.sleep(0.01)  # Small delay between messages
        
        # Start subscribers
        subscriber_threads = []
        for i in range(num_subscribers):
            subscriber_id = f"subscriber_{i}"
            channels = [f"test_channel_{j}" for j in range(num_publishers)]
            thread = threading.Thread(
                target=subscriber_worker,
                args=(subscriber_id, channels)
            )
            thread.start()
            subscriber_threads.append(thread)
        
        time.sleep(1)  # Give subscribers time to connect
        
        # Start publishers
        publisher_threads = []
        for i in range(num_publishers):
            publisher_id = f"publisher_{i}"
            channel = f"test_channel_{i}"
            thread = threading.Thread(
                target=publisher_worker,
                args=(publisher_id, channel, messages_per_publisher)
            )
            thread.start()
            publisher_threads.append(thread)
        
        # Wait for publishers to finish
        for thread in publisher_threads:
            thread.join()
        
        # Wait for subscribers (with timeout)
        for thread in subscriber_threads:
            thread.join(timeout=35)
        
        # Analyze results
        total_sent = num_publishers * messages_per_publisher
        total_received = sum(len(messages) for messages in received_messages.values())
        
        return {
            'publishers': num_publishers,
            'subscribers': num_subscribers,
            'messages_per_publisher': messages_per_publisher,
            'total_sent': total_sent,
            'total_received': total_received,
            'delivery_rate': total_received / total_sent if total_sent > 0 else 0,
            'received_by_subscriber': {k: len(v) for k, v in received_messages.items()}
        }
    
    def run_full_load_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive load test suite."""
        print("🚀 Starting Full Memory System Load Test Suite")
        print("=" * 60)
        
        # Test Redis connectivity
        try:
            r = self.get_redis_connection()
            r.ping()
            print("✅ Redis connectivity confirmed")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return {"error": "Redis not available"}
        
        # Clear any existing test data
        r.flushdb()
        print("✅ Test database cleared")
        
        results = {
            'test_suite': 'Memory System Load Testing',
            'timestamp': time.time(),
            'concurrent_load_tests': [],
            'pubsub_test': None,
            'summary': {}
        }
        
        # Test 1: Low concurrency
        result1 = self.run_concurrent_load_test(5, 100)
        results['concurrent_load_tests'].append(asdict(result1))
        
        # Test 2: Medium concurrency
        result2 = self.run_concurrent_load_test(10, 50)
        results['concurrent_load_tests'].append(asdict(result2))
        
        # Test 3: High concurrency
        result3 = self.run_concurrent_load_test(20, 25)
        results['concurrent_load_tests'].append(asdict(result3))
        
        # Test 4: Pub/Sub coordination
        pubsub_result = self.test_pubsub_coordination(5, 3, 20)
        results['pubsub_test'] = pubsub_result
        
        # Calculate summary metrics
        avg_ops_per_sec = np.mean([r['operations_per_second'] for r in results['concurrent_load_tests']])
        avg_success_rate = np.mean([r['success_rate'] for r in results['concurrent_load_tests']])
        max_memory_usage = max([r['memory_usage_mb'] for r in results['concurrent_load_tests']])
        
        results['summary'] = {
            'average_operations_per_second': avg_ops_per_sec,
            'average_success_rate': avg_success_rate,
            'maximum_memory_usage_mb': max_memory_usage,
            'pubsub_delivery_rate': pubsub_result['delivery_rate'],
            'overall_status': 'PASS' if avg_success_rate > 0.95 else 'FAIL'
        }
        
        print("\n" + "=" * 60)
        print("📊 Load Test Results Summary:")
        print(f"   Average Ops/Sec: {avg_ops_per_sec:.2f}")
        print(f"   Average Success Rate: {avg_success_rate:.2%}")
        print(f"   Max Memory Usage: {max_memory_usage:.2f} MB")
        print(f"   Pub/Sub Delivery Rate: {pubsub_result['delivery_rate']:.2%}")
        print(f"   Overall Status: {results['summary']['overall_status']}")
        
        return results

def main():
    """Main function to run memory load tests."""
    tester = MemoryLoadTester()
    
    # Run the complete test suite
    results = tester.run_full_load_test_suite()
    
    # Save results
    with open('test_output/memory_load_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Load test results saved to: test_output/memory_load_test_results.json")
    
    if results.get('summary', {}).get('overall_status') == 'PASS':
        print("🎉 Memory system load test PASSED!")
        return True
    else:
        print("❌ Memory system load test FAILED!")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)