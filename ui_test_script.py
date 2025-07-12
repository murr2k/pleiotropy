#!/usr/bin/env python3
"""
UI Testing Script for Genomic Pleiotropy Dashboard
Tests the UI functionality with real data and generates reports
"""

import json
import time
import requests
from datetime import datetime
import subprocess
import sys

class UITester:
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.ui_base = "http://localhost:3000"
        self.test_results = []
        
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        
    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            success = response.status_code == 200 and response.json().get("status") == "healthy"
            self.log_test("API Health Check", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_test("API Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_ui_loading(self):
        """Test UI main page loading"""
        try:
            response = requests.get(f"{self.ui_base}", timeout=10)
            success = response.status_code == 200 and "Vite + React" in response.text
            self.log_test("UI Main Page Loading", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_test("UI Main Page Loading", False, f"Error: {str(e)}")
            return False
    
    def test_ui_health(self):
        """Test UI health endpoint"""
        try:
            response = requests.get(f"{self.ui_base}/health", timeout=5)
            success = response.status_code == 200 and "healthy" in response.text
            self.log_test("UI Health Check", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_test("UI Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_api_endpoints(self):
        """Test various API endpoints"""
        endpoints = [
            ("/", "API Root"),
            ("/docs", "API Documentation"),
            ("/api/v1/trials/", "Trials Endpoint"),
        ]
        
        all_success = True
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.api_base}{endpoint}", timeout=5)
                success = response.status_code == 200
                self.log_test(f"API {name}", success, f"Status: {response.status_code}")
                all_success = all_success and success
            except Exception as e:
                self.log_test(f"API {name}", False, f"Error: {str(e)}")
                all_success = False
        
        return all_success
    
    def test_api_proxy_through_ui(self):
        """Test API proxy through UI nginx"""
        try:
            response = requests.get(f"{self.ui_base}/api/", timeout=5)
            # Should return JSON from API
            success = response.status_code == 200
            self.log_test("API Proxy Through UI", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_test("API Proxy Through UI", False, f"Error: {str(e)}")
            return False
    
    def test_ui_static_assets(self):
        """Test UI static asset loading"""
        try:
            # Test CSS
            response = requests.get(f"{self.ui_base}/assets/index-DRzzxluL.css", timeout=5)
            css_success = response.status_code == 200
            
            # Test JS
            response = requests.get(f"{self.ui_base}/assets/index-kj_pXPFm.js", timeout=5)
            js_success = response.status_code == 200
            
            success = css_success and js_success
            self.log_test("UI Static Assets", success, f"CSS: {css_success}, JS: {js_success}")
            return success
        except Exception as e:
            self.log_test("UI Static Assets", False, f"Error: {str(e)}")
            return False
    
    def test_performance(self):
        """Test basic performance metrics"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.ui_base}", timeout=10)
            load_time = time.time() - start_time
            
            success = response.status_code == 200 and load_time < 5.0
            self.log_test("UI Performance", success, f"Load time: {load_time:.2f}s")
            return success
        except Exception as e:
            self.log_test("UI Performance", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all UI tests"""
        print("🚀 Starting UI Testing for Genomic Pleiotropy Dashboard")
        print("=" * 60)
        
        # Test services are running
        api_healthy = self.test_api_health()
        ui_healthy = self.test_ui_health()
        
        if not api_healthy or not ui_healthy:
            print("❌ Core services not healthy. Aborting tests.")
            return False
        
        # Run tests
        tests = [
            self.test_ui_loading,
            self.test_api_endpoints,
            self.test_api_proxy_through_ui,
            self.test_ui_static_assets,
            self.test_performance,
        ]
        
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append(False)
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {passed}/{total} tests passed")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Check logs above.")
        
        return passed == total
    
    def generate_report(self):
        """Generate detailed test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "ui_testing_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r["success"]),
                "failed": sum(1 for r in self.test_results if not r["success"]),
                "success_rate": sum(1 for r in self.test_results if r["success"]) / len(self.test_results) * 100 if self.test_results else 0
            }
        }
        
        report_file = f"ui_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📝 Test report saved to: {report_file}")
        return report_file

if __name__ == "__main__":
    tester = UITester()
    success = tester.run_all_tests()
    report_file = tester.generate_report()
    
    sys.exit(0 if success else 1)