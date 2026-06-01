"""
Test script for FastAPI ML Model Service.

This script tests all API endpoints to ensure proper functionality.
Run this after deploying the service to validate everything works.

Usage:
    python test_api.py
"""

import requests
import json
from typing import Dict, Any
from datetime import datetime


class APITester:
    """Test suite for ML Model API."""

    def __init__(self, base_url: str = "http://localhost:8800"):
        """
        Initialize API tester.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def _log_test(self, test_name: str, passed: bool, message: str = "") -> None:
        """
        Log test result.

        Args:
            test_name: Name of the test
            passed: Whether test passed
            message: Additional message
        """
        status = "✓ PASS" if passed else "✗ FAIL"
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status}: {test_name}")
        if message:
            print(f"  → {message}")

    def test_root_endpoint(self) -> bool:
        """Test root endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/")
            passed = response.status_code == 200
            message = f"Status: {response.status_code}"

            if passed:
                data = response.json()
                message += f" | Message: {data.get('message', 'N/A')}"

            self._log_test("Root Endpoint", passed, message)
            return passed

        except Exception as e:
            self._log_test("Root Endpoint", False, str(e))
            return False

    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            passed = response.status_code == 200

            if passed:
                data = response.json()
                model_loaded = data.get("model_loaded", False)
                status = data.get("status", "unknown")
                message = f"Status: {status} | Model Loaded: {model_loaded}"

                if not model_loaded:
                    passed = False
                    message += " | WARNING: Model not loaded!"

            else:
                message = f"Status: {response.status_code}"

            self._log_test("Health Check", passed, message)
            return passed

        except Exception as e:
            self._log_test("Health Check", False, str(e))
            return False

    def test_model_info(self) -> bool:
        """Test model info endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            passed = response.status_code == 200

            if passed:
                data = response.json()
                model_name = data.get("name", "N/A")
                model_version = data.get("version", "N/A")
                message = f"Model: {model_name} v{model_version}"
            else:
                message = f"Status: {response.status_code}"

            self._log_test("Model Info", passed, message)
            return passed

        except Exception as e:
            self._log_test("Model Info", False, str(e))
            return False

    def test_prediction(self) -> bool:
        """Test prediction endpoint."""
        try:
            # Example prediction request
            payload = {
                "data": [
                    {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0},
                    {"feature1": 5.0, "feature2": 6.0, "feature3": 7.0, "feature4": 8.0}
                ],
                "return_probabilities": True
            }

            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            passed = response.status_code == 200

            if passed:
                data = response.json()
                predictions = data.get("predictions", [])
                n_predictions = len(predictions)
                message = f"Received {n_predictions} predictions"

                if "probabilities" in data and data["probabilities"]:
                    message += " with probabilities"

            else:
                message = f"Status: {response.status_code}"
                if response.status_code == 422:
                    message += " | Invalid input format"
                elif response.status_code == 503:
                    message += " | Model not loaded"

            self._log_test("Prediction", passed, message)
            return passed

        except Exception as e:
            self._log_test("Prediction", False, str(e))
            return False

    def test_prediction_single_input(self) -> bool:
        """Test prediction with single input."""
        try:
            payload = {
                "data": {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0},
                "return_probabilities": False
            }

            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            passed = response.status_code == 200

            if passed:
                data = response.json()
                predictions = data.get("predictions", [])
                message = f"Single input prediction successful"
            else:
                message = f"Status: {response.status_code}"

            self._log_test("Prediction (Single Input)", passed, message)
            return passed

        except Exception as e:
            self._log_test("Prediction (Single Input)", False, str(e))
            return False

    def test_invalid_input(self) -> bool:
        """Test prediction with invalid input."""
        try:
            payload = {
                "data": [],  # Empty data
                "return_probabilities": False
            }

            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            # Should return 422 for invalid input
            passed = response.status_code == 422

            message = f"Correctly rejected invalid input (Status: {response.status_code})"

            self._log_test("Invalid Input Handling", passed, message)
            return passed

        except Exception as e:
            self._log_test("Invalid Input Handling", False, str(e))
            return False

    def test_docs_endpoint(self) -> bool:
        """Test documentation endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/docs")
            passed = response.status_code == 200
            message = f"Status: {response.status_code}"

            self._log_test("Documentation Endpoint", passed, message)
            return passed

        except Exception as e:
            self._log_test("Documentation Endpoint", False, str(e))
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests.

        Returns:
            dict: Test results summary
        """
        print("\n" + "="*60)
        print("🧪 Running API Tests")
        print("="*60 + "\n")

        print(f"Target: {self.base_url}\n")

        # Run tests
        tests = [
            self.test_root_endpoint,
            self.test_health_check,
            self.test_model_info,
            self.test_docs_endpoint,
            self.test_prediction,
            self.test_prediction_single_input,
            self.test_invalid_input,
        ]

        for test in tests:
            test()
            print()

        # Summary
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed

        print("="*60)
        print("📊 Test Summary")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print("="*60 + "\n")

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed/total)*100,
            "results": self.test_results
        }

        return summary


def main():
    """Main function."""
    import sys

    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8800"

    # Run tests
    tester = APITester(base_url)
    summary = tester.run_all_tests()

    # Exit with appropriate code
    exit_code = 0 if summary["failed"] == 0 else 1

    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Please check the logs.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
