#!/usr/bin/env python3
"""
Egyptian ID OCR - API Endpoint Verification Script
Tests all API endpoints and measures performance.

Usage:
    python verify-endpoints.py http://localhost:8000
    python verify-endpoints.py https://your-domain.com
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import requests
    from requests.exceptions import RequestException, Timeout, ConnectionError
except ImportError:
    print("Installing required package: requests")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def print_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


class APIVerifier:
    """Verifies Egyptian ID OCR API endpoints."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.results: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'base_url': base_url,
            'endpoints': {},
            'summary': {'passed': 0, 'failed': 0, 'total': 0}
        }

    def _make_request(self, method: str, endpoint: str, **kwargs) -> tuple:
        """Make HTTP request and return (status_code, response_json, duration_ms)."""
        url = f"{self.base_url}{endpoint}"
        start = time.time()
        
        try:
            response = self.session.request(
                method, url,
                timeout=self.timeout,
                **kwargs
            )
            duration_ms = int((time.time() - start) * 1000)
            
            try:
                data = response.json()
            except:
                data = response.text[:500]
            
            return response.status_code, data, duration_ms
            
        except Timeout:
            return 0, {'error': 'Request timeout'}, int((time.time() - start) * 1000)
        except ConnectionError as e:
            return 0, {'error': str(e)}, 0
        except RequestException as e:
            return 0, {'error': str(e)}, 0

    def test_root(self) -> Dict:
        """Test GET / endpoint."""
        print_info("Testing GET /...")
        status, data, duration = self._make_request('GET', '/')
        
        result = {
            'endpoint': 'GET /',
            'status_code': status,
            'duration_ms': duration,
            'passed': status == 200,
            'response': data
        }
        
        if status == 200:
            print_success(f"GET / - HTTP {status} ({duration}ms)")
        else:
            print_error(f"GET / - HTTP {status}")
        
        return result

    def test_health(self) -> Dict:
        """Test GET /api/v1/health endpoint."""
        print_info("Testing GET /api/v1/health...")
        status, data, duration = self._make_request('GET', '/api/v1/health')
        
        result = {
            'endpoint': 'GET /api/v1/health',
            'status_code': status,
            'duration_ms': duration,
            'passed': status == 200,
            'response': data
        }
        
        if status == 200:
            print_success(f"GET /api/v1/health - HTTP {status} ({duration}ms)")
            if isinstance(data, dict):
                models_loaded = data.get('models_loaded', 'unknown')
                print_info(f"Models loaded: {models_loaded}")
        else:
            print_error(f"GET /api/v1/health - HTTP {status}")
        
        return result

    def test_models(self) -> Dict:
        """Test GET /api/v1/models endpoint."""
        print_info("Testing GET /api/v1/models...")
        status, data, duration = self._make_request('GET', '/api/v1/models')
        
        result = {
            'endpoint': 'GET /api/v1/models',
            'status_code': status,
            'duration_ms': duration,
            'passed': status == 200,
            'response': data
        }
        
        if status == 200:
            print_success(f"GET /api/v1/models - HTTP {status} ({duration}ms)")
        else:
            print_error(f"GET /api/v1/models - HTTP {status}")
        
        return result

    def test_docs(self) -> Dict:
        """Test GET /docs endpoint (Swagger UI)."""
        print_info("Testing GET /docs...")
        status, data, duration = self._make_request('GET', '/docs')
        
        result = {
            'endpoint': 'GET /docs',
            'status_code': status,
            'duration_ms': duration,
            'passed': status == 200,
            'response': 'HTML content' if status == 200 else data
        }
        
        if status == 200:
            print_success(f"GET /docs - HTTP {status} ({duration}ms)")
        else:
            print_error(f"GET /docs - HTTP {status}")
        
        return result

    def test_redoc(self) -> Dict:
        """Test GET /redoc endpoint."""
        print_info("Testing GET /redoc...")
        status, data, duration = self._make_request('GET', '/redoc')
        
        result = {
            'endpoint': 'GET /redoc',
            'status_code': status,
            'duration_ms': duration,
            'passed': status == 200,
            'response': 'HTML content' if status == 200 else data
        }
        
        if status == 200:
            print_success(f"GET /redoc - HTTP {status} ({duration}ms)")
        else:
            print_error(f"GET /redoc - HTTP {status}")
        
        return result

    def test_extract(self, image_path: Optional[str] = None) -> Dict:
        """Test POST /api/v1/extract endpoint."""
        print_info("Testing POST /api/v1/extract...")
        
        # Find test image
        if image_path and Path(image_path).exists():
            test_image = Path(image_path)
        else:
            # Search for test images
            possible_paths = [
                'debug/test_id.jpg',
                'tests/test_id.jpg',
                'test_images/test_id.jpg',
                'samples/test_id.jpg',
            ]
            test_image = None
            for path in possible_paths:
                if Path(path).exists():
                    test_image = Path(path)
                    break
        
        if test_image is None:
            result = {
                'endpoint': 'POST /api/v1/extract',
                'status_code': 0,
                'duration_ms': 0,
                'passed': False,
                'skipped': True,
                'response': {'error': 'No test image found'}
            }
            print_warning("POST /api/v1/extract - SKIPPED (no test image)")
            return result
        
        print_info(f"Using test image: {test_image}")
        
        try:
            with open(test_image, 'rb') as f:
                files = {'file': f}
                status, data, duration = self._make_request(
                    'POST', '/api/v1/extract',
                    files=files
                )
        except Exception as e:
            status, data, duration = 0, {'error': str(e)}, 0
        
        result = {
            'endpoint': 'POST /api/v1/extract',
            'status_code': status,
            'duration_ms': duration,
            'passed': status == 200,
            'response': data
        }
        
        if status == 200:
            print_success(f"POST /api/v1/extract - HTTP {status} ({duration}ms)")
            
            # Extract performance info
            if isinstance(data, dict):
                processing_ms = data.get('processing_ms', 'N/A')
                confidence = data.get('confidence', {})
                overall_conf = confidence.get('overall', 'N/A') if isinstance(confidence, dict) else 'N/A'
                print_info(f"Processing time: {processing_ms}ms")
                print_info(f"Overall confidence: {overall_conf}")
                
                # Extracted fields
                extracted = data.get('extracted', {})
                if extracted:
                    print_info(f"Extracted fields: {list(extracted.keys())}")
        else:
            print_error(f"POST /api/v1/extract - HTTP {status}")
        
        return result

    def test_extract_performance(self, image_path: Optional[str] = None, iterations: int = 3) -> Dict:
        """Run performance benchmark for extract endpoint."""
        print_info(f"Running performance benchmark ({iterations} iterations)...")
        
        # Find test image
        if image_path and Path(image_path).exists():
            test_image = Path(image_path)
        else:
            possible_paths = ['debug/test_id.jpg', 'tests/test_id.jpg']
            test_image = None
            for path in possible_paths:
                if Path(path).exists():
                    test_image = Path(path)
                    break
        
        if test_image is None:
            return {'skipped': True, 'reason': 'No test image'}
        
        durations = []
        success_count = 0
        
        for i in range(iterations):
            print_info(f"  Iteration {i+1}/{iterations}...")
            
            try:
                with open(test_image, 'rb') as f:
                    start = time.time()
                    response = self.session.post(
                        f"{self.base_url}/api/v1/extract",
                        files={'file': f},
                        timeout=self.timeout
                    )
                    duration_ms = int((time.time() - start) * 1000)
                    
                    if response.status_code == 200:
                        success_count += 1
                        durations.append(duration_ms)
                        print_info(f"    Success: {duration_ms}ms")
                    else:
                        print_error(f"    Failed: HTTP {response.status_code}")
                        
            except Exception as e:
                print_error(f"    Error: {e}")
        
        if not durations:
            return {
                'skipped': True,
                'reason': 'No successful requests'
            }
        
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        result = {
            'iterations': iterations,
            'successful': success_count,
            'avg_duration_ms': round(avg_duration, 2),
            'min_duration_ms': min_duration,
            'max_duration_ms': max_duration,
            'success_rate': round(success_count / iterations * 100, 2)
        }
        
        print_success(f"Benchmark complete:")
        print_info(f"  Success rate: {result['success_rate']}%")
        print_info(f"  Avg duration: {result['avg_duration_ms']}ms")
        print_info(f"  Min duration: {result['min_duration_ms']}ms")
        print_info(f"  Max duration: {result['max_duration_ms']}ms")
        
        return result

    def run_all_tests(self, include_benchmark: bool = False, image_path: Optional[str] = None) -> Dict:
        """Run all verification tests."""
        print_info("=" * 50)
        print_info("Egyptian ID OCR API Verification")
        print_info(f"Base URL: {self.base_url}")
        print_info("=" * 50)
        print()
        
        # Run basic endpoint tests
        endpoints = [
            ('root', self.test_root),
            ('health', self.test_health),
            ('models', self.test_models),
            ('docs', self.test_docs),
            ('redoc', self.test_redoc),
            ('extract', lambda: self.test_extract(image_path)),
        ]
        
        for name, test_func in endpoints:
            self.results['endpoints'][name] = test_func()
            self.results['summary']['total'] += 1
            if self.results['endpoints'][name].get('passed', False):
                self.results['summary']['passed'] += 1
            elif not self.results['endpoints'][name].get('skipped', False):
                self.results['summary']['failed'] += 1
            print()
        
        # Run performance benchmark
        if include_benchmark:
            self.results['benchmark'] = self.test_extract_performance(image_path)
        
        # Print summary
        self._print_summary()
        
        return self.results

    def _print_summary(self):
        """Print test summary."""
        print()
        print(Colors.GREEN + "=" * 50 + Colors.NC)
        print(Colors.GREEN + "Test Summary" + Colors.NC)
        print(Colors.GREEN + "=" * 50 + Colors.NC)
        print()
        
        summary = self.results['summary']
        total = summary['total']
        passed = summary['passed']
        failed = summary['failed']
        
        print(f"Total endpoints: {total}")
        print(f"Passed: {Colors.GREEN}{passed}{Colors.NC}")
        print(f"Failed: {Colors.RED}{failed}{Colors.NC}")
        print()
        
        if failed == 0:
            print_success("All tests passed!")
        else:
            print_error(f"{failed} test(s) failed")
        
        # Performance summary
        if 'benchmark' in self.results and not self.results['benchmark'].get('skipped'):
            print()
            print_info("Performance Summary:")
            bench = self.results['benchmark']
            print(f"  Average response time: {bench['avg_duration_ms']}ms")
            print(f"  Success rate: {bench['success_rate']}%")
            
            # Check against targets
            if bench['avg_duration_ms'] < 15000:
                print_success("  ✓ Processing time target met (<15s)")
            else:
                print_warning("  ✗ Processing time exceeds target (>15s)")


def main():
    parser = argparse.ArgumentParser(
        description='Egyptian ID OCR API Endpoint Verification'
    )
    parser.add_argument(
        'base_url',
        nargs='?',
        default='http://localhost:8000',
        help='Base URL of the API (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=60,
        help='Request timeout in seconds (default: 60)'
    )
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run performance benchmark'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to test image for extract endpoint'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output results to JSON file'
    )
    
    args = parser.parse_args()
    
    verifier = APIVerifier(args.base_url, timeout=args.timeout)
    results = verifier.run_all_tests(
        include_benchmark=args.benchmark,
        image_path=args.image
    )
    
    # Save results to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print_info(f"Results saved to: {args.output}")
    
    # Exit with error code if tests failed
    sys.exit(0 if results['summary']['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
