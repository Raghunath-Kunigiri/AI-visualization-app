#!/usr/bin/env python3
"""
Smart Data App Deployment Script
Run this to prepare your app for cloud deployment
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        missing_packages = []
        for req in requirements:
            package = req.split('>=')[0].split('==')[0]
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(req)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed!")
        else:
            print("✅ All requirements satisfied!")
        
        return True
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def create_dockerfile():
    """Create a Dockerfile for containerization"""
    dockerfile_content = """# Smart Data App Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created!")

def create_docker_compose():
    """Create a docker-compose.yml for easy deployment"""
    compose_content = """version: '3.8'

services:
  smart-data-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data  # Optional: for persistent data storage
    restart: unless-stopped

networks:
  default:
    name: smart-data-network
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("✅ docker-compose.yml created!")

def create_streamlit_config():
    """Create Streamlit configuration"""
    os.makedirs('.streamlit', exist_ok=True)
    
    config_content = """[general]
email = "your-email@example.com"

[server]
headless = true
enableCORS = false
port = 8501

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config_content)
    
    print("✅ Streamlit config created!")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Data files (optional)
*.csv
*.xlsx
*.json
data/

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created!")

def print_deployment_instructions():
    """Print deployment instructions"""
    print("""
🚀 DEPLOYMENT READY! 

📋 Your deployment options:

1️⃣ STREAMLIT CLOUD (Easiest):
   • Push to GitHub
   • Go to share.streamlit.io
   • Connect your repo
   • Deploy!

2️⃣ DOCKER (Local/Cloud):
   docker build -t smart-data-app .
   docker run -p 8501:8501 smart-data-app

3️⃣ DOCKER COMPOSE:
   docker-compose up -d

4️⃣ HEROKU:
   • Install Heroku CLI
   • heroku create your-app-name
   • git push heroku main

5️⃣ AWS/AZURE/GCP:
   • Use the Dockerfile
   • Deploy to container services

🌐 Your app will be available at: http://localhost:8501
""")

def main():
    """Main deployment preparation function"""
    print("🚀 Smart Data App Deployment Preparation\n")
    
    if not check_requirements():
        print("❌ Failed to install requirements. Please check manually.")
        return
    
    create_dockerfile()
    create_docker_compose()
    create_streamlit_config()
    create_gitignore()
    
    print_deployment_instructions()

if __name__ == "__main__":
    main() 