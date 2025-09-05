#!/usr/bin/env python3
"""
Backend API Testing for Customer Segmentation & Targeting Application
Tests all backend endpoints according to test_result.md requirements
"""

import requests
import json
import time
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')
BASE_URL = os.getenv('REACT_APP_BACKEND_URL', 'https://customer-segment-pro.preview.emergentagent.com')
API_BASE = f"{BASE_URL}/api"

class CustomerSegmentationTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.test_results = {
            'data_management': {'passed': 0, 'failed': 0, 'errors': []},
            'clustering_engine': {'passed': 0, 'failed': 0, 'errors': []},
            'ai_recommendations': {'passed': 0, 'failed': 0, 'errors': []},
            'segmentation_analysis': {'passed': 0, 'failed': 0, 'errors': []}
        }
        
    def log_result(self, category: str, test_name: str, success: bool, error_msg: str = None):
        """Log test results"""
        if success:
            self.test_results[category]['passed'] += 1
            print(f"✅ {test_name}")
        else:
            self.test_results[category]['failed'] += 1
            self.test_results[category]['errors'].append(f"{test_name}: {error_msg}")
            print(f"❌ {test_name}: {error_msg}")
    
    def test_api_health(self):
        """Test basic API connectivity"""
        print(f"\n🔍 Testing API Health at {API_BASE}")
        try:
            response = self.session.get(f"{API_BASE}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Health Check: {data.get('message', 'OK')}")
                return True
            else:
                print(f"❌ API Health Check Failed: Status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Health Check Failed: {str(e)}")
            return False
    
    def test_data_management_apis(self):
        """Test Customer Data Management API endpoints"""
        print(f"\n🧪 Testing Customer Data Management APIs")
        
        # Test 1: Generate Sample Data
        try:
            print("Testing POST /api/data/sample...")
            response = self.session.post(f"{API_BASE}/data/sample")
            
            if response.status_code == 200:
                data = response.json()
                if 'count' in data and data['count'] > 0:
                    self.log_result('data_management', 'Sample Data Generation', True)
                    print(f"  Generated {data['count']} sample customers")
                else:
                    self.log_result('data_management', 'Sample Data Generation', False, "No count in response")
            else:
                self.log_result('data_management', 'Sample Data Generation', False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result('data_management', 'Sample Data Generation', False, str(e))
        
        # Test 2: Retrieve Customers
        try:
            print("Testing GET /api/data/customers...")
            response = self.session.get(f"{API_BASE}/data/customers")
            
            if response.status_code == 200:
                customers = response.json()
                if isinstance(customers, list) and len(customers) > 0:
                    # Validate customer structure
                    sample_customer = customers[0]
                    required_fields = ['customer_id', 'age', 'annual_income', 'spending_score']
                    
                    if all(field in sample_customer for field in required_fields):
                        self.log_result('data_management', 'Customer Retrieval', True)
                        print(f"  Retrieved {len(customers)} customers with valid structure")
                        
                        # Validate data types and ranges
                        valid_data = True
                        for customer in customers[:5]:  # Check first 5
                            if not (18 <= customer['age'] <= 70):
                                valid_data = False
                                break
                            if not (10000 <= customer['annual_income'] <= 150000):
                                valid_data = False
                                break
                            if not (1 <= customer['spending_score'] <= 100):
                                valid_data = False
                                break
                        
                        if valid_data:
                            print("  ✅ Customer data ranges are realistic")
                        else:
                            print("  ⚠️ Some customer data outside expected ranges")
                    else:
                        self.log_result('data_management', 'Customer Retrieval', False, "Missing required fields")
                else:
                    self.log_result('data_management', 'Customer Retrieval', False, "No customers returned")
            else:
                self.log_result('data_management', 'Customer Retrieval', False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result('data_management', 'Customer Retrieval', False, str(e))
        
        # Test 3: Get Statistics
        try:
            print("Testing GET /api/data/stats...")
            response = self.session.get(f"{API_BASE}/data/stats")
            
            if response.status_code == 200:
                stats = response.json()
                required_stats = ['total_customers', 'age_stats', 'income_stats', 'spending_stats']
                
                if all(stat in stats for stat in required_stats):
                    # Validate statistics structure
                    age_stats = stats['age_stats']
                    if all(key in age_stats for key in ['mean', 'std', 'min', 'max']):
                        self.log_result('data_management', 'Statistics Calculation', True)
                        print(f"  Total customers: {stats['total_customers']}")
                        print(f"  Age range: {age_stats['min']:.1f} - {age_stats['max']:.1f} (avg: {age_stats['mean']:.1f})")
                        print(f"  Income range: ${stats['income_stats']['min']:,.0f} - ${stats['income_stats']['max']:,.0f}")
                    else:
                        self.log_result('data_management', 'Statistics Calculation', False, "Invalid age_stats structure")
                else:
                    self.log_result('data_management', 'Statistics Calculation', False, "Missing required statistics")
            else:
                self.log_result('data_management', 'Statistics Calculation', False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result('data_management', 'Statistics Calculation', False, str(e))
    
    def test_clustering_engine(self):
        """Test Clustering Analysis Engine"""
        print(f"\n🧪 Testing Clustering Analysis Engine")
        
        # Test 1: Elbow Analysis
        try:
            print("Testing POST /api/analysis/elbow...")
            payload = ["age", "annual_income", "spending_score"]
            response = self.session.post(f"{API_BASE}/analysis/elbow", json=payload)
            
            if response.status_code == 200:
                elbow_data = response.json()
                required_fields = ['k_values', 'inertias', 'silhouette_scores']
                
                if all(field in elbow_data for field in required_fields):
                    k_values = elbow_data['k_values']
                    inertias = elbow_data['inertias']
                    silhouette_scores = elbow_data['silhouette_scores']
                    
                    if len(k_values) == len(inertias) == len(silhouette_scores):
                        # Validate that inertias decrease (generally expected)
                        inertia_decreasing = all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1))
                        
                        self.log_result('clustering_engine', 'Elbow Analysis', True)
                        print(f"  K-values tested: {k_values}")
                        print(f"  Inertia trend: {'Decreasing' if inertia_decreasing else 'Variable'}")
                        print(f"  Best silhouette score: {max(silhouette_scores):.3f}")
                    else:
                        self.log_result('clustering_engine', 'Elbow Analysis', False, "Mismatched array lengths")
                else:
                    self.log_result('clustering_engine', 'Elbow Analysis', False, "Missing required fields")
            else:
                self.log_result('clustering_engine', 'Elbow Analysis', False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result('clustering_engine', 'Elbow Analysis', False, str(e))
        
        # Test 2: K-Means Clustering
        try:
            print("Testing POST /api/analysis/cluster (K-Means)...")
            payload = {
                "algorithm": "kmeans",
                "n_clusters": 4,
                "features": ["age", "annual_income", "spending_score"]
            }
            response = self.session.post(f"{API_BASE}/analysis/cluster", json=payload)
            
            if response.status_code == 200:
                cluster_result = response.json()
                required_fields = ['algorithm', 'n_clusters', 'silhouette_score', 'cluster_labels']
                
                if all(field in cluster_result for field in required_fields):
                    silhouette_score = cluster_result['silhouette_score']
                    cluster_labels = cluster_result['cluster_labels']
                    
                    # Validate silhouette score range (-1 to 1)
                    if -1 <= silhouette_score <= 1:
                        # Check if we have the right number of clusters
                        unique_clusters = len(set(cluster_labels))
                        if unique_clusters == payload['n_clusters']:
                            self.log_result('clustering_engine', 'K-Means Clustering', True)
                            print(f"  Silhouette Score: {silhouette_score:.3f}")
                            print(f"  Clusters formed: {unique_clusters}")
                            if 'inertia' in cluster_result:
                                print(f"  Inertia: {cluster_result['inertia']:.2f}")
                        else:
                            self.log_result('clustering_engine', 'K-Means Clustering', False, f"Expected {payload['n_clusters']} clusters, got {unique_clusters}")
                    else:
                        self.log_result('clustering_engine', 'K-Means Clustering', False, f"Invalid silhouette score: {silhouette_score}")
                else:
                    self.log_result('clustering_engine', 'K-Means Clustering', False, "Missing required fields")
            else:
                self.log_result('clustering_engine', 'K-Means Clustering', False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result('clustering_engine', 'K-Means Clustering', False, str(e))
        
        # Test 3: Hierarchical Clustering
        try:
            print("Testing POST /api/analysis/cluster (Hierarchical)...")
            payload = {
                "algorithm": "hierarchical",
                "n_clusters": 3,
                "features": ["age", "annual_income", "spending_score"]
            }
            response = self.session.post(f"{API_BASE}/analysis/cluster", json=payload)
            
            if response.status_code == 200:
                cluster_result = response.json()
                
                if cluster_result['algorithm'] == 'hierarchical':
                    silhouette_score = cluster_result['silhouette_score']
                    cluster_labels = cluster_result['cluster_labels']
                    
                    if -1 <= silhouette_score <= 1:
                        unique_clusters = len(set(cluster_labels))
                        if unique_clusters == payload['n_clusters']:
                            self.log_result('clustering_engine', 'Hierarchical Clustering', True)
                            print(f"  Silhouette Score: {silhouette_score:.3f}")
                            print(f"  Clusters formed: {unique_clusters}")
                        else:
                            self.log_result('clustering_engine', 'Hierarchical Clustering', False, f"Expected {payload['n_clusters']} clusters, got {unique_clusters}")
                    else:
                        self.log_result('clustering_engine', 'Hierarchical Clustering', False, f"Invalid silhouette score: {silhouette_score}")
                else:
                    self.log_result('clustering_engine', 'Hierarchical Clustering', False, "Algorithm mismatch")
            else:
                self.log_result('clustering_engine', 'Hierarchical Clustering', False, f"Status {response.status_code}")
        except Exception as e:
            self.log_result('clustering_engine', 'Hierarchical Clustering', False, str(e))
    
    def test_ai_recommendations_and_segmentation(self):
        """Test AI-Powered Business Recommendations and Customer Segmentation Analysis"""
        print(f"\n🧪 Testing AI-Powered Segmentation Analysis")
        
        # Test: Get Customer Segments with AI Recommendations
        try:
            print("Testing GET /api/analysis/segments...")
            response = self.session.get(f"{API_BASE}/analysis/segments")
            
            if response.status_code == 200:
                segments = response.json()
                
                if isinstance(segments, list) and len(segments) > 0:
                    # Validate segment structure
                    sample_segment = segments[0]
                    required_fields = ['cluster_id', 'label', 'description', 'customer_count', 
                                     'avg_age', 'avg_income', 'avg_spending', 'business_recommendations']
                    
                    if all(field in sample_segment for field in required_fields):
                        # Validate AI recommendations
                        recommendations = sample_segment['business_recommendations']
                        
                        if isinstance(recommendations, list) and len(recommendations) > 0:
                            # Check if recommendations are meaningful (not just default)
                            meaningful_recommendations = any(
                                len(rec) > 50 and not rec.startswith("Develop targeted") 
                                for rec in recommendations
                            )
                            
                            self.log_result('ai_recommendations', 'AI Business Recommendations', True)
                            self.log_result('segmentation_analysis', 'Customer Segmentation Analysis', True)
                            
                            print(f"  Generated {len(segments)} customer segments")
                            for i, segment in enumerate(segments):
                                print(f"  Segment {i+1}: {segment['label']} ({segment['customer_count']} customers)")
                                print(f"    Avg Age: {segment['avg_age']:.1f}, Income: ${segment['avg_income']:,.0f}, Spending: {segment['avg_spending']:.1f}")
                                print(f"    Recommendations: {len(segment['business_recommendations'])} items")
                            
                            if meaningful_recommendations:
                                print("  ✅ AI recommendations appear to be customized")
                            else:
                                print("  ⚠️ AI recommendations may be using fallback defaults")
                        else:
                            self.log_result('ai_recommendations', 'AI Business Recommendations', False, "No recommendations generated")
                            self.log_result('segmentation_analysis', 'Customer Segmentation Analysis', False, "Missing recommendations")
                    else:
                        missing_fields = [f for f in required_fields if f not in sample_segment]
                        self.log_result('ai_recommendations', 'AI Business Recommendations', False, f"Missing fields: {missing_fields}")
                        self.log_result('segmentation_analysis', 'Customer Segmentation Analysis', False, f"Missing fields: {missing_fields}")
                else:
                    self.log_result('ai_recommendations', 'AI Business Recommendations', False, "No segments returned")
                    self.log_result('segmentation_analysis', 'Customer Segmentation Analysis', False, "No segments returned")
            else:
                error_msg = f"Status {response.status_code}"
                if response.status_code == 404:
                    error_msg += " - No clustered data found (run clustering first)"
                
                self.log_result('ai_recommendations', 'AI Business Recommendations', False, error_msg)
                self.log_result('segmentation_analysis', 'Customer Segmentation Analysis', False, error_msg)
        except Exception as e:
            self.log_result('ai_recommendations', 'AI Business Recommendations', False, str(e))
            self.log_result('segmentation_analysis', 'Customer Segmentation Analysis', False, str(e))
    
    def test_error_handling(self):
        """Test API error handling"""
        print(f"\n🧪 Testing Error Handling")
        
        # Test invalid clustering algorithm
        try:
            payload = {
                "algorithm": "invalid_algorithm",
                "n_clusters": 3,
                "features": ["age", "annual_income", "spending_score"]
            }
            response = self.session.post(f"{API_BASE}/analysis/cluster", json=payload)
            
            if response.status_code == 400:
                print("✅ Invalid algorithm properly rejected")
            else:
                print(f"⚠️ Expected 400 for invalid algorithm, got {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error testing invalid algorithm: {str(e)}")
        
        # Test invalid cluster count
        try:
            payload = {
                "algorithm": "kmeans",
                "n_clusters": 0,
                "features": ["age", "annual_income", "spending_score"]
            }
            response = self.session.post(f"{API_BASE}/analysis/cluster", json=payload)
            
            if response.status_code in [400, 422]:
                print("✅ Invalid cluster count properly rejected")
            else:
                print(f"⚠️ Expected 400/422 for invalid cluster count, got {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error testing invalid cluster count: {str(e)}")
    
    def run_all_tests(self):
        """Run comprehensive backend API tests"""
        print("=" * 80)
        print("🚀 CUSTOMER SEGMENTATION & TARGETING - BACKEND API TESTING")
        print("=" * 80)
        
        # Check API health first
        if not self.test_api_health():
            print("\n❌ API is not accessible. Stopping tests.")
            return False
        
        # Run all test suites
        self.test_data_management_apis()
        self.test_clustering_engine()
        self.test_ai_recommendations_and_segmentation()
        self.test_error_handling()
        
        # Print summary
        self.print_test_summary()
        
        return self.is_overall_success()
    
    def print_test_summary(self):
        """Print comprehensive test results summary"""
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed
            
            status = "✅ PASS" if failed == 0 else "❌ FAIL"
            category_name = category.replace('_', ' ').title()
            
            print(f"{category_name:30} | {status} | {passed} passed, {failed} failed")
            
            if results['errors']:
                for error in results['errors']:
                    print(f"  ❌ {error}")
        
        print("-" * 80)
        overall_status = "✅ ALL TESTS PASSED" if total_failed == 0 else f"❌ {total_failed} TESTS FAILED"
        print(f"{'OVERALL RESULT':30} | {overall_status} | {total_passed} passed, {total_failed} failed")
        print("=" * 80)
    
    def is_overall_success(self):
        """Check if all critical tests passed"""
        total_failed = sum(results['failed'] for results in self.test_results.values())
        return total_failed == 0

def main():
    """Main test execution"""
    tester = CustomerSegmentationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All backend API tests completed successfully!")
        exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the summary above for details.")
        exit(1)

if __name__ == "__main__":
    main()