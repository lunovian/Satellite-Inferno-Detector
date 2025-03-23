from typing import Dict, Optional
import json
import redis
from pathlib import Path

# Setup Redis for task status tracking
redis_client = redis.Redis(host="localhost", port=6379, db=0)


class TaskManager:
    """Manage background detection tasks"""

    @staticmethod
    def update_status(task_id: str, status: str, result: Optional[Dict] = None):
        """Update task status in Redis"""
        data = {"status": status, "result": result}
        redis_client.set(f"task:{task_id}", json.dumps(data), ex=3600)

    @staticmethod
    def get_status(task_id: str) -> Optional[Dict]:
        """Get task status from Redis"""
        data = redis_client.get(f"task:{task_id}")
        return json.loads(data) if data else None


# Add more utility functions as needed...
