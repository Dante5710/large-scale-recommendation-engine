"""
Test suite for recommendation engine
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import RecommendationEngine, RecommendationRequest

class TestRecommendationEngine:
    def setup_method(self):
        self.engine = RecommendationEngine()
    
    def test_health_check(self):
        """Test system health"""
        assert self.engine is not None
    
    def test_recommendations(self):
        """Test recommendation generation"""
        req = RecommendationRequest(user_id=123, num_recommendations=5)
        response = self.engine.get_recommendations(req)
        
        assert response.user_id == 123
        assert len(response.recommendations) <= 5
        assert response.response_time_ms > 0
    
    def test_performance_metrics(self):
        """Test metrics collection"""
        metrics = self.engine.get_performance_metrics()
        
        assert 'total_requests' in metrics
        assert 'avg_response_time_ms' in metrics
        assert 'qps_capacity' in metrics
        assert metrics['qps_capacity'] >= 1000

if __name__ == "__main__":
    pytest.main([__file__])
