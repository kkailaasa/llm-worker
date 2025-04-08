#!/usr/bin/env python3
import os
from huggingface_hub import HfApi

def validate_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: No Hugging Face token found in environment variables")
        return False

    print(f"Token found (first 5 chars): {token[:5]}...")

    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"Token is valid for user: {user_info['name']}")

        # Try checking access to the specific model
        try:
            model_info = api.model_info("google/gemma-3-27b-it")
            print(f"Access to Gemma 3 confirmed. Model: {model_info.modelId}")
        except Exception as e:
            print(f"Could not access Gemma 3 model: {e}")
            return False

        return True
    except Exception as e:
        print(f"Token validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_token()