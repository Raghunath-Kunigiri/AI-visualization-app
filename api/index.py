from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess
import threading
import time
import os

def run_streamlit():
    """Run Streamlit in background"""
    subprocess.run([
        "streamlit", "run", "../app.py", 
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])

def handler(environ, start_response):
    """WSGI handler for Vercel"""
    # Start Streamlit in background thread
    thread = threading.Thread(target=run_streamlit, daemon=True)
    thread.start()
    
    # Wait a moment for Streamlit to start
    time.sleep(2)
    
    # Redirect to Streamlit port
    status = '302 Found'
    headers = [('Location', 'http://localhost:8501')]
    start_response(status, headers)
    return [b'Redirecting to Streamlit app...']

# Vercel entry point
app = handler 