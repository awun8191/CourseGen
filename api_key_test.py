#!/usr/bin/env python3
"""
Script to test if a Gemini API key works.
Usage:
    python3 api_key_test.py --key YOUR_API_KEY
"""

import argparse
import google.generativeai as genai

def test_gemini_api_key(api_key: str):
    try:
        genai.configure(api_key=api_key)
        # Use a supported model name
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content("Hi")
        print("✅ API Key is working!")
        print("Response:", response.text.strip())
    except Exception as e:
        print("❌ API call failed:", e)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test a Gemini API key.")
    # parser.add_argument("--key", required=True, help="Gemini API key to test")
    # args = parser.parse_args()
    #
    # test_gemini_api_key(args.key)

    from services.Gemini.gemini_api_keys import GeminiApiKeys

    key = GeminiApiKeys().api_keys

    for i in key:
        print("*" * 60)
        print(f"Key: {i}")
        test_gemini_api_key(i)
        print("i")


