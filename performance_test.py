from locust import HttpUser, task, between
import random
import json

class RecommendationUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task(3)
    def get_recommendations(self):
        user_id = random.randint(1, 1000)
        with self.client.get(f"/recommendations/{user_id}?count=10", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def log_interaction(self):
        user_id = random.randint(1, 1000)
        payload = {
            "item_id": random.randint(1, 1000),
            "rating": round(random.uniform(1, 5), 1),
            "context": {"device": random.choice(["mobile", "desktop"])}
        }
        self.client.post(f"/user/{user_id}/interaction", json=payload)
    
    @task(1)
    def check_health(self):
        self.client.get("/health")

if __name__ == "__main__":
    import os
    os.system("locust -f performance_test.py --host=http://localhost:5000")
