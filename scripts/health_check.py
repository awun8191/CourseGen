#!/usr/bin/env python3
"""
Health check script for CourseGen Docker container.
Verifies that all essential components are working properly.
"""

import sys
import os
import json
from pathlib import Path

def check_python_imports():
    """Check if all required Python packages can be imported."""
    required_packages = [
        'chromadb',
        'google.genai',
        'requests',
        'numpy',
        'pandas',
        'pydantic',
        'PIL',
        'cv2'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError as e:
            failed_imports.append(f"{package}: {e}")
    
    if failed_imports:
        print("❌ Failed to import required packages:")
        for failure in failed_imports:
            print(f"  - {failure}")
        return False
    
    print("✅ All required packages imported successfully")
    return True

def check_data_directories():
    """Check if essential data directories exist and are accessible."""
    required_dirs = [
        '/app/data/textbooks',
        '/app/OUTPUT_DATA2/emdeddings',
        '/app/OUTPUT_DATA2/cache'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    print("✅ All required directories exist")
    return True

def check_chromadb_connection():
    """Check if ChromaDB can be accessed."""
    try:
        import chromadb
        from chromadb import PersistentClient
        
        chroma_path = os.environ.get('CHROMA_PERSIST_DIR', '/app/OUTPUT_DATA2/emdeddings')
        client = PersistentClient(path=chroma_path)
        
        # Try to list collections
        collections = client.list_collections()
        print(f"✅ ChromaDB connection successful ({len(collections)} collections found)")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False

def check_courses_json():
    """Check if courses.json exists and is valid."""
    try:
        courses_path = os.environ.get('COURSEGEN_COURSES_JSON', '/app/data/textbooks/courses.json')
        
        if not Path(courses_path).exists():
            print(f"❌ courses.json not found at {courses_path}")
            return False
        
        with open(courses_path, 'r') as f:
            courses_data = json.load(f)
        
        if not isinstance(courses_data, list) or len(courses_data) == 0:
            print("❌ courses.json is empty or invalid format")
            return False
        
        print(f"✅ courses.json is valid ({len(courses_data)} courses found)")
        return True
        
    except Exception as e:
        print(f"❌ courses.json check failed: {e}")
        return False

def check_environment_variables():
    """Check if required environment variables are set."""
    required_env_vars = [
        'PYTHONPATH',
        'COURSEGEN_COURSES_JSON',
        'CHROMA_PERSIST_DIR'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_api_keys():
    """Check if API keys are configured (optional for basic functionality)."""
    api_keys = {
        'GOOGLE_API_KEY': 'Google Gemini API',
        'CLOUDFLARE_ACCOUNT_ID': 'Cloudflare Account ID',
        'CLOUDFLARE_API_TOKEN': 'Cloudflare API Token'
    }
    
    missing_keys = []
    for key, description in api_keys.items():
        if not os.environ.get(key):
            missing_keys.append(f"{key} ({description})")
    
    if missing_keys:
        print("⚠️  Missing API keys (required for question generation):")
        for key in missing_keys:
            print(f"  - {key}")
        return False
    
    print("✅ All API keys are configured")
    return True

def main():
    """Run all health checks."""
    print("🏥 CourseGen Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Imports", check_python_imports),
        ("Data Directories", check_data_directories),
        ("Environment Variables", check_environment_variables),
        ("courses.json", check_courses_json),
        ("ChromaDB Connection", check_chromadb_connection),
        ("API Keys", check_api_keys)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All health checks passed! CourseGen is ready to use.")
        sys.exit(0)
    elif passed >= total - 1:  # Allow API keys to be missing
        print("⚠️  Most checks passed. Missing API keys will prevent question generation.")
        sys.exit(0)
    else:
        print("💥 Critical health checks failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
