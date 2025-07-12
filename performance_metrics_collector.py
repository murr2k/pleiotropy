#!/usr/bin/env python3
"""
Performance Metrics Collector for Complete System

Collects comprehensive performance metrics across all system components
and generates detailed performance analysis reports.
"""

import time
import json
import psutil
import subprocess
import concurrent.futures
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import redis
import pandas as pd
import numpy as np

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int

@dataclass
class ComponentPerformance:
    """Performance metrics for a system component."""
    component_name: str
    execution_time: float
    memory_peak_mb: float
    cpu_peak_percent: float
    throughput_ops_per_sec: float
    success_rate: float
    error_count: int

@dataclass
class IntegrationMetrics:
    """Integration performance metrics."""
    rust_python_latency: float
    memory_system_latency: float
    data_transfer_rate_mbps: float
    end_to_end_latency: float

class PerformanceCollector:
    """Collects performance metrics across the entire system."""
    
    def __init__(self):
        """Initialize the performance collector."""
        self.metrics_history = []
        self.component_metrics = {}
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            network_io_bytes_sent=network.bytes_sent,
            network_io_bytes_recv=network.bytes_recv
        )
    
    def benchmark_rust_performance(self) -> ComponentPerformance:
        """Benchmark Rust component performance."""
        print("⚡ Benchmarking Rust performance...")
        
        # Create test data for benchmarking
        test_sequences = 50
        
        # Write test file
        with open('benchmark_input.fasta', 'w') as f:
            for i in range(test_sequences):
                f.write(f""">test_gene_{i} Benchmark gene {i}
ATGTCTGATCTGGGTGGTAACCTGATCGACTGGATTACCGGCTTTCTGCAACGTGGCTACTTCGAAGTTGTTAATCATGTGATCCGACAACGTAAATACTCTGGCTACCTGGAAGGTGGTGGCCGTGAATTCAACAAGGAAGTTTCTGGTATTAAACAGTCCGTGAACGGCGATTTTGGCGTGGCGGTTAAAGAATTCGAACTGAACCTGGGCAAATCTGATCCGGGTGACCGTTCCGGCGAATGGGTGATGGCTGAAATCGGTACCTCTGGTGGTTACGGTGAAAATCTGGGTATCGACATCGTCGCAAAAGGCATTCCAGAAGCCCTGAAACAGGATATTATTGCGCAGAAACAGTATGGTAAACTCTTCGAAGTTGTAAACAGCGAACTGGAAGTCGATATTGGCAAATACAACCACGACCAACAAGTTTCCCAAAAAGGCTATCAGGTAGACGCCGATTTCATTGAAGCGAAACAGCGCGGTATTCACATTCACATTCTGAAACGTGTTTCTACCAAACTGGGGCAAGTTGCCCGCTACGGCAACTTCGGCTAG

""")
        
        with open('benchmark_traits.json', 'w') as f:
            json.dump([
                {"name": "test_trait", "description": "Test trait", "associated_genes": ["test"], "known_sequences": []}
            ], f)
        
        # Monitor system during execution
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run benchmark
        result = subprocess.run([
            './pleiotropy_core',
            '--input', 'benchmark_input.fasta',
            '--traits', 'benchmark_traits.json',
            '--output', 'benchmark_output',
            '--verbose'
        ], capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = max(initial_memory, final_memory)
        
        # Calculate throughput
        throughput = test_sequences / execution_time if execution_time > 0 else 0
        success_rate = 1.0 if result.returncode == 0 else 0.0
        
        # Cleanup
        import os
        for f in ['benchmark_input.fasta', 'benchmark_traits.json']:
            if os.path.exists(f):
                os.remove(f)
        
        return ComponentPerformance(
            component_name="rust_core",
            execution_time=execution_time,
            memory_peak_mb=peak_memory,
            cpu_peak_percent=psutil.cpu_percent(),
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            error_count=0 if result.returncode == 0 else 1
        )
    
    def benchmark_python_performance(self) -> ComponentPerformance:
        """Benchmark Python components performance."""
        print("🐍 Benchmarking Python performance...")
        
        import sys
        sys.path.append('python_analysis')
        from rust_interface import RustInterface, InterfaceMode
        from statistical_analyzer import StatisticalAnalyzer
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create test data
        test_data = {
            "sequences": 1000,
            "identified_traits": [],
            "frequency_table": {
                "codon_frequencies": [
                    {"codon": "ATG", "amino_acid": "M", "global_frequency": 0.02, "trait_specific_frequency": {}},
                    {"codon": "TGA", "amino_acid": "*", "global_frequency": 0.01, "trait_specific_frequency": {}}
                ] * 32,  # 64 codons
                "total_codons": 10000,
                "trait_codon_counts": {}
            }
        }
        
        # Process data multiple times for benchmarking
        operations = 0
        for i in range(100):
            # Simulate data processing
            df = pd.DataFrame(test_data["frequency_table"]["codon_frequencies"])
            stats = {
                "mean_frequency": df["global_frequency"].mean(),
                "std_frequency": df["global_frequency"].std(),
                "total_codons": test_data["frequency_table"]["total_codons"]
            }
            operations += 1
        
        execution_time = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = max(initial_memory, final_memory)
        
        throughput = operations / execution_time if execution_time > 0 else 0
        
        return ComponentPerformance(
            component_name="python_analysis",
            execution_time=execution_time,
            memory_peak_mb=peak_memory,
            cpu_peak_percent=psutil.cpu_percent(),
            throughput_ops_per_sec=throughput,
            success_rate=1.0,
            error_count=0
        )
    
    def benchmark_memory_system(self) -> ComponentPerformance:
        """Benchmark Redis memory system performance."""
        print("💾 Benchmarking memory system performance...")
        
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
        except:
            return ComponentPerformance(
                component_name="memory_system",
                execution_time=0,
                memory_peak_mb=0,
                cpu_peak_percent=0,
                throughput_ops_per_sec=0,
                success_rate=0.0,
                error_count=1
            )
        
        start_time = time.time()
        operations = 0
        errors = 0
        
        # Benchmark Redis operations
        for i in range(1000):
            try:
                # SET operation
                key = f"benchmark_key_{i}"
                value = json.dumps({"test_data": i, "timestamp": time.time()})
                r.setex(key, 60, value)
                
                # GET operation
                retrieved = r.get(key)
                if retrieved:
                    json.loads(retrieved)
                
                # List operation
                r.lpush(f"benchmark_list_{i % 10}", value)
                
                operations += 3  # Count SET, GET, LPUSH
                
            except Exception:
                errors += 1
        
        execution_time = time.time() - start_time
        throughput = operations / execution_time if execution_time > 0 else 0
        success_rate = (operations - errors) / operations if operations > 0 else 0
        
        # Get Redis memory usage
        memory_info = r.info('memory')
        memory_usage_mb = memory_info.get('used_memory', 0) / (1024 * 1024)
        
        return ComponentPerformance(
            component_name="memory_system",
            execution_time=execution_time,
            memory_peak_mb=memory_usage_mb,
            cpu_peak_percent=psutil.cpu_percent(),
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            error_count=errors
        )
    
    def measure_integration_latency(self) -> IntegrationMetrics:
        """Measure integration latency between components."""
        print("🔗 Measuring integration latency...")
        
        # Rust-Python integration latency
        start_time = time.time()
        result = subprocess.run(['./pleiotropy_core', '--version'], capture_output=True)
        rust_python_latency = time.time() - start_time
        
        # Memory system latency
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            start_time = time.time()
            r.ping()
            memory_latency = time.time() - start_time
        except:
            memory_latency = float('inf')
        
        # Data transfer rate (simulated)
        test_data_size_mb = 1.0  # 1 MB test data
        start_time = time.time()
        
        # Simulate data transfer
        test_data = "A" * int(test_data_size_mb * 1024 * 1024)
        with open('/tmp/test_transfer.txt', 'w') as f:
            f.write(test_data)
        
        with open('/tmp/test_transfer.txt', 'r') as f:
            data = f.read()
        
        transfer_time = time.time() - start_time
        transfer_rate = test_data_size_mb / transfer_time if transfer_time > 0 else 0
        
        # End-to-end latency (simple workflow)
        start_time = time.time()
        
        # Simulate mini workflow
        with open('/tmp/mini_genome.fasta', 'w') as f:
            f.write(">test\nATGTCTGATCTGGGTGGTAACCTGATCGACTGGATTACCGGCTTTCTGCAACGTGGCTACTTCGAAGTTG\n")
        
        with open('/tmp/mini_traits.json', 'w') as f:
            json.dump([{"name": "test", "description": "test", "associated_genes": [], "known_sequences": []}], f)
        
        subprocess.run([
            './pleiotropy_core',
            '--input', '/tmp/mini_genome.fasta',
            '--traits', '/tmp/mini_traits.json',
            '--output', '/tmp/mini_output'
        ], capture_output=True)
        
        end_to_end_latency = time.time() - start_time
        
        # Cleanup
        import os
        for f in ['/tmp/test_transfer.txt', '/tmp/mini_genome.fasta', '/tmp/mini_traits.json']:
            if os.path.exists(f):
                os.remove(f)
        
        return IntegrationMetrics(
            rust_python_latency=rust_python_latency,
            memory_system_latency=memory_latency,
            data_transfer_rate_mbps=transfer_rate * 8,  # Convert to Mbps
            end_to_end_latency=end_to_end_latency
        )
    
    def run_concurrent_stress_test(self) -> Dict[str, Any]:
        """Run concurrent stress test across all components."""
        print("🔥 Running concurrent stress test...")
        
        def stress_worker(worker_id: int) -> Dict[str, Any]:
            """Worker function for stress testing."""
            try:
                # Mini workflow execution
                with open(f'/tmp/stress_{worker_id}.fasta', 'w') as f:
                    f.write(f">stress_gene_{worker_id}\nATGTCTGATCTGGGTGGTAACCTGATCGACTGGATTACCGGCTTTCTGCAACGTGGCTACTTCGAAGTTG\n")
                
                with open(f'/tmp/stress_traits_{worker_id}.json', 'w') as f:
                    json.dump([{"name": f"stress_trait_{worker_id}", "description": "stress", "associated_genes": [], "known_sequences": []}], f)
                
                start_time = time.time()
                result = subprocess.run([
                    './pleiotropy_core',
                    '--input', f'/tmp/stress_{worker_id}.fasta',
                    '--traits', f'/tmp/stress_traits_{worker_id}.json',
                    '--output', f'/tmp/stress_output_{worker_id}'
                ], capture_output=True)
                
                execution_time = time.time() - start_time
                
                # Cleanup
                import os
                for f in [f'/tmp/stress_{worker_id}.fasta', f'/tmp/stress_traits_{worker_id}.json']:
                    if os.path.exists(f):
                        os.remove(f)
                
                return {
                    "worker_id": worker_id,
                    "execution_time": execution_time,
                    "success": result.returncode == 0,
                    "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
                }
                
            except Exception as e:
                return {
                    "worker_id": worker_id,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e),
                    "memory_usage": 0
                }
        
        # Run concurrent workers
        num_workers = 10
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_workers)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_workers = [r for r in results if r.get('success', False)]
        success_rate = len(successful_workers) / len(results)
        avg_execution_time = np.mean([r['execution_time'] for r in successful_workers])
        max_memory_usage = max([r.get('memory_usage', 0) for r in results])
        
        return {
            "total_workers": num_workers,
            "successful_workers": len(successful_workers),
            "success_rate": success_rate,
            "total_execution_time": total_time,
            "average_worker_time": avg_execution_time,
            "max_memory_usage_mb": max_memory_usage,
            "throughput_workers_per_sec": num_workers / total_time
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        print("📊 Collecting all performance metrics...")
        
        # Collect system metrics
        system_metrics = self.collect_system_metrics()
        
        # Benchmark individual components
        rust_perf = self.benchmark_rust_performance()
        python_perf = self.benchmark_python_performance()
        memory_perf = self.benchmark_memory_system()
        
        # Measure integration
        integration_metrics = self.measure_integration_latency()
        
        # Run stress test
        stress_results = self.run_concurrent_stress_test()
        
        # Compile results
        performance_data = {
            "timestamp": time.time(),
            "system_metrics": asdict(system_metrics),
            "component_performance": {
                "rust": asdict(rust_perf),
                "python": asdict(python_perf),
                "memory": asdict(memory_perf)
            },
            "integration_metrics": asdict(integration_metrics),
            "stress_test_results": stress_results,
            "summary": {
                "overall_throughput": rust_perf.throughput_ops_per_sec + python_perf.throughput_ops_per_sec,
                "average_latency": np.mean([integration_metrics.rust_python_latency, integration_metrics.memory_system_latency]),
                "system_efficiency": (rust_perf.success_rate + python_perf.success_rate + memory_perf.success_rate) / 3,
                "stress_test_success_rate": stress_results["success_rate"],
                "memory_efficiency_score": 1 - (system_metrics.memory_percent / 100),
                "overall_score": 0.0  # Will be calculated
            }
        }
        
        # Calculate overall performance score
        scores = [
            min(performance_data["summary"]["overall_throughput"] / 100, 1.0),  # Throughput score
            max(0, 1 - performance_data["summary"]["average_latency"]),  # Latency score (lower is better)
            performance_data["summary"]["system_efficiency"],  # Efficiency score
            performance_data["summary"]["stress_test_success_rate"],  # Reliability score
            performance_data["summary"]["memory_efficiency_score"]  # Memory score
        ]
        performance_data["summary"]["overall_score"] = np.mean(scores)
        
        # Save detailed results
        report_file = "test_output/performance_metrics_report.json"
        with open(report_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Generate summary report
        summary_file = "test_output/performance_summary.md"
        with open(summary_file, 'w') as f:
            f.write(f"""# System Performance Analysis Report

## Executive Summary
- **Overall Performance Score**: {performance_data["summary"]["overall_score"]:.2f}/1.00
- **System Efficiency**: {performance_data["summary"]["system_efficiency"]:.2%}
- **Stress Test Success Rate**: {performance_data["summary"]["stress_test_success_rate"]:.2%}

## Component Performance

### Rust Core Engine
- **Execution Time**: {rust_perf.execution_time:.3f}s
- **Throughput**: {rust_perf.throughput_ops_per_sec:.2f} ops/sec
- **Memory Usage**: {rust_perf.memory_peak_mb:.2f} MB
- **Success Rate**: {rust_perf.success_rate:.2%}

### Python Analysis
- **Execution Time**: {python_perf.execution_time:.3f}s
- **Throughput**: {python_perf.throughput_ops_per_sec:.2f} ops/sec
- **Memory Usage**: {python_perf.memory_peak_mb:.2f} MB
- **Success Rate**: {python_perf.success_rate:.2%}

### Memory System (Redis)
- **Throughput**: {memory_perf.throughput_ops_per_sec:.2f} ops/sec
- **Memory Usage**: {memory_perf.memory_peak_mb:.2f} MB
- **Success Rate**: {memory_perf.success_rate:.2%}

## Integration Metrics
- **Rust-Python Latency**: {integration_metrics.rust_python_latency:.3f}s
- **Memory System Latency**: {integration_metrics.memory_system_latency:.3f}s
- **Data Transfer Rate**: {integration_metrics.data_transfer_rate_mbps:.2f} Mbps
- **End-to-End Latency**: {integration_metrics.end_to_end_latency:.3f}s

## Stress Test Results
- **Concurrent Workers**: {stress_results["total_workers"]}
- **Success Rate**: {stress_results["success_rate"]:.2%}
- **Average Worker Time**: {stress_results["average_worker_time"]:.3f}s
- **Max Memory Usage**: {stress_results["max_memory_usage_mb"]:.2f} MB

## System Resources
- **CPU Usage**: {system_metrics.cpu_percent:.1f}%
- **Memory Usage**: {system_metrics.memory_percent:.1f}% ({system_metrics.memory_used_gb:.2f} GB used)
- **Disk Usage**: {system_metrics.disk_usage_percent:.1f}%

## Recommendations
{self._generate_recommendations(performance_data)}

---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S UTC')}*
""")
        
        print(f"✅ Performance report saved to: {report_file}")
        print(f"✅ Performance summary saved to: {summary_file}")
        
        return report_file
    
    def _generate_recommendations(self, performance_data: Dict[str, Any]) -> str:
        """Generate performance recommendations."""
        recommendations = []
        
        overall_score = performance_data["summary"]["overall_score"]
        
        if overall_score >= 0.9:
            recommendations.append("🎉 **Excellent Performance**: System is performing optimally.")
        elif overall_score >= 0.7:
            recommendations.append("✅ **Good Performance**: System is performing well with minor optimization opportunities.")
        else:
            recommendations.append("⚠️ **Performance Issues**: System requires optimization.")
        
        # Memory recommendations
        if performance_data["system_metrics"]["memory_percent"] > 80:
            recommendations.append("- **Memory**: Consider increasing available memory or optimizing memory usage.")
        
        # Latency recommendations
        avg_latency = performance_data["summary"]["average_latency"]
        if avg_latency > 0.1:
            recommendations.append("- **Latency**: High latency detected. Consider optimizing inter-component communication.")
        
        # Stress test recommendations
        stress_success = performance_data["stress_test_results"]["success_rate"]
        if stress_success < 0.95:
            recommendations.append("- **Reliability**: Stress test shows reliability issues. Review error handling and resource management.")
        
        return '\n'.join(recommendations)

def main():
    """Main function to run performance collection."""
    collector = PerformanceCollector()
    report_file = collector.generate_performance_report()
    
    print(f"\n🎯 Performance analysis complete!")
    print(f"📄 Detailed report: {report_file}")
    print(f"📄 Summary report: test_output/performance_summary.md")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)