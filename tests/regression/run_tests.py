#!/usr/bin/env python3
"""
API Regression Test Runner
Comprehensive test execution with reporting and memory namespace management

Memory Namespace: swarm-regression-1752301224
"""

import sys
import os
import argparse
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add API path for imports
api_path = Path(__file__).parent.parent.parent / "trial_database" / "api"
sys.path.append(str(api_path))


class RegressionTestRunner:
    """Manages execution of comprehensive API regression tests"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.memory_namespace = "swarm-regression-1752301224"
        self.start_time = datetime.now()
        
    def run_all_tests(self, verbose: bool = True, coverage: bool = True) -> Dict[str, Any]:
        """Run all regression tests and generate comprehensive report"""
        print("🚀 Starting Comprehensive API Regression Tests")
        print(f"📅 Start Time: {self.start_time.isoformat()}")
        print(f"💾 Memory Namespace: {self.memory_namespace}")
        print("=" * 60)
        
        results = {
            "start_time": self.start_time.isoformat(),
            "memory_namespace": self.memory_namespace,
            "test_results": {},
            "reports_generated": [],
            "summary": {}
        }
        
        # Run main API tests
        print("\n📊 Running Main API Tests...")
        api_results = self._run_pytest(
            "test_api.py",
            markers=None,
            verbose=verbose,
            coverage=coverage
        )
        results["test_results"]["api_tests"] = api_results
        
        # Run security tests
        print("\n🔒 Running Security Vulnerability Tests...")
        security_results = self._run_pytest(
            "test_security_vulnerabilities.py",
            markers="security",
            verbose=verbose
        )
        results["test_results"]["security_tests"] = security_results
        
        # Run performance tests
        print("\n⚡ Running Performance Benchmark Tests...")
        performance_results = self._run_pytest(
            "test_performance_benchmarks.py", 
            markers="performance",
            verbose=verbose
        )
        results["test_results"]["performance_tests"] = performance_results
        
        # Generate summary
        results["summary"] = self._generate_summary(results["test_results"])
        
        # Generate consolidated report
        report_file = self._generate_consolidated_report(results)
        results["reports_generated"].append(report_file)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def run_security_only(self, verbose: bool = True) -> Dict[str, Any]:
        """Run only security vulnerability tests"""
        print("🔒 Running Security Vulnerability Tests Only")
        
        results = self._run_pytest(
            "test_security_vulnerabilities.py",
            markers="security",
            verbose=verbose
        )
        
        print(f"\n🔒 Security Tests Complete: {results['summary']}")
        return results
    
    def run_performance_only(self, verbose: bool = True) -> Dict[str, Any]:
        """Run only performance benchmark tests"""
        print("⚡ Running Performance Benchmark Tests Only")
        
        results = self._run_pytest(
            "test_performance_benchmarks.py",
            markers="performance",
            verbose=verbose
        )
        
        print(f"\n⚡ Performance Tests Complete: {results['summary']}")
        return results
    
    def run_api_only(self, verbose: bool = True, coverage: bool = True) -> Dict[str, Any]:
        """Run only main API functionality tests"""
        print("📊 Running Main API Tests Only")
        
        results = self._run_pytest(
            "test_api.py",
            markers=None,
            verbose=verbose,
            coverage=coverage
        )
        
        print(f"\n📊 API Tests Complete: {results['summary']}")
        return results
    
    def _run_pytest(self, test_file: str, markers: str = None, 
                   verbose: bool = True, coverage: bool = False) -> Dict[str, Any]:
        """Run pytest with specified parameters"""
        
        cmd = ["python", "-m", "pytest"]
        
        # Add test file
        cmd.append(str(self.test_dir / test_file))
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        
        # Add markers
        if markers:
            cmd.extend(["-m", markers])
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=app",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        
        # Add other options
        cmd.extend([
            "--tb=short",
            "--asyncio-mode=auto",
            "--timeout=300",  # 5 minute timeout per test
            f"--junitxml={self.test_dir}/test-results-{test_file.replace('.py', '')}.xml"
        ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute total timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "summary": self._parse_pytest_output(result.stdout)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "execution_time": 1800,
                "stdout": "",
                "stderr": "Test execution timed out after 30 minutes",
                "command": " ".join(cmd),
                "summary": {"error": "timeout"}
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "execution_time": 0,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd),
                "summary": {"error": str(e)}
            }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results"""
        lines = output.split('\n')
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0
        }
        
        for line in lines:
            if "passed" in line and "failed" in line:
                # Look for summary line like "5 passed, 2 failed in 10.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        if i > 0 and parts[i-1].isdigit():
                            summary["passed"] = int(parts[i-1])
                    elif part == "failed":
                        if i > 0 and parts[i-1].isdigit():
                            summary["failed"] = int(parts[i-1])
                    elif part == "skipped":
                        if i > 0 and parts[i-1].isdigit():
                            summary["skipped"] = int(parts[i-1])
                    elif part == "error" or part == "errors":
                        if i > 0 and parts[i-1].isdigit():
                            summary["errors"] = int(parts[i-1])
            
            elif "warnings summary" in line.lower():
                summary["warnings"] = 1
        
        summary["total_tests"] = (
            summary["passed"] + summary["failed"] + 
            summary["skipped"] + summary["errors"]
        )
        
        return summary
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test execution summary"""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_time = 0
        
        for test_type, results in test_results.items():
            if "summary" in results and isinstance(results["summary"], dict):
                summary = results["summary"]
                total_tests += summary.get("total_tests", 0)
                total_passed += summary.get("passed", 0)
                total_failed += summary.get("failed", 0)
                total_errors += summary.get("errors", 0)
            
            if "execution_time" in results:
                total_time += results["execution_time"]
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "success_rate": success_rate,
            "total_execution_time": total_time,
            "overall_success": total_failed == 0 and total_errors == 0
        }
    
    def _generate_consolidated_report(self, results: Dict[str, Any]) -> str:
        """Generate consolidated test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_dir / f"consolidated_regression_report_{timestamp}.json"
        
        # Add additional metadata
        results["end_time"] = datetime.now().isoformat()
        results["execution_duration"] = (datetime.now() - self.start_time).total_seconds()
        results["test_environment"] = {
            "python_version": sys.version,
            "platform": sys.platform,
            "test_directory": str(self.test_dir),
            "working_directory": os.getcwd()
        }
        
        # Find and include generated reports
        report_files = list(self.test_dir.glob("*_report_*.json"))
        results["individual_reports"] = [str(f) for f in report_files]
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(report_file)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test execution summary"""
        summary = results["summary"]
        
        print("\n" + "=" * 60)
        print("🎯 API REGRESSION TEST SUMMARY")
        print("=" * 60)
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['total_passed']}")
        print(f"❌ Failed: {summary['total_failed']}")
        print(f"⚠️  Errors: {summary['total_errors']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total Time: {summary['total_execution_time']:.1f}s")
        
        if summary['overall_success']:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {summary['total_failed'] + summary['total_errors']} TEST FAILURES DETECTED")
        
        print(f"\n📋 Reports Generated:")
        for report in results.get("reports_generated", []):
            print(f"   📄 {report}")
        
        print(f"\n💾 Memory Namespace: {self.memory_namespace}")
        print("=" * 60)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="API Regression Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                 # Run all tests
  python run_tests.py --security           # Security tests only
  python run_tests.py --performance        # Performance tests only
  python run_tests.py --api                # API functionality tests only
  python run_tests.py --all --no-coverage  # All tests without coverage
        """
    )
    
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--all", action="store_true", help="Run all regression tests")
    test_group.add_argument("--security", action="store_true", help="Run security tests only")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests only")
    test_group.add_argument("--api", action="store_true", help="Run API functionality tests only")
    
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    
    args = parser.parse_args()
    
    runner = RegressionTestRunner()
    
    verbose = not args.no_verbose
    coverage = not args.no_coverage
    
    try:
        if args.all:
            results = runner.run_all_tests(verbose=verbose, coverage=coverage)
        elif args.security:
            results = runner.run_security_only(verbose=verbose)
        elif args.performance:
            results = runner.run_performance_only(verbose=verbose)
        elif args.api:
            results = runner.run_api_only(verbose=verbose, coverage=coverage)
        
        # Exit with appropriate code
        if results.get("summary", {}).get("overall_success", False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()