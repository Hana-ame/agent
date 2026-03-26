import os
from dotenv import load_dotenv
from pathlib import Path

# 尝试加载 .env 文件（如果存在）
env_path = Path(".env")
if env_path.exists():
    load_dotenv()
else:
    print("No .env file found, using defaults")

class Config:
    APP_NAME = os.getenv("APP_NAME", "MyApp")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")

if __name__ == "__main__":
    print(f"APP_NAME: {Config.APP_NAME}")
    print(f"DEBUG: {Config.DEBUG}")
    print(f"DATABASE_URL: {Config.DATABASE_URL}")
    print(f"SECRET_KEY: {Config.SECRET_KEY[:10]}...")
