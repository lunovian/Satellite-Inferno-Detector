from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    max_requests_per_minute: int = 60
    cache_ttl: int = 3600
    results_dir: str = "results"

    class Config:
        env_file = ".env"


settings = Settings()
