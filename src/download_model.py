import os
import json
import logging
import glob
import sys
from shutil import rmtree
from huggingface_hub import snapshot_download, HfApi
from utils import timer_decorator

BASE_DIR = "/"
TOKENIZER_PATTERNS = [["*.json", "tokenizer*"]]
MODEL_PATTERNS = [["*.safetensors"], ["*.bin"], ["*.pt"]]

def validate_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: No Hugging Face token found in environment variables")
        return False

    print(f"HF Token present (first 5 chars): {token[:5]}...")

    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"Token is valid for user: {user_info['name']}")
        return True
    except Exception as e:
        print(f"Token validation failed: {e}")
        return False

def setup_env():
    if os.getenv("TESTING_DOWNLOAD") == "1":
        BASE_DIR = "tmp"
        os.makedirs(BASE_DIR, exist_ok=True)
        os.environ.update({
            "HF_HOME": f"{BASE_DIR}/hf_cache",
            "MODEL_NAME": "openchat/openchat-3.5-0106",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TENSORIZE": "1",
            "TENSORIZER_NUM_GPUS": "1",
            "DTYPE": "auto"
        })

@timer_decorator
def download(name, revision, type, cache_dir):
    # Get token for authentication
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("No Hugging Face token found in environment variables")

    print(f"Downloading {type} from {name} with revision {revision}")

    if type == "model":
        pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
    elif type == "tokenizer":
        pattern_sets = TOKENIZER_PATTERNS
    else:
        raise ValueError(f"Invalid type: {type}")
    try:
        for pattern_set in pattern_sets:
            path = snapshot_download(
                name,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=pattern_set,
                token=token  # Explicitly pass the token
            )
            for pattern in pattern_set:
                if glob.glob(os.path.join(path, pattern)):
                    logging.info(f"Successfully downloaded {pattern} model files.")
                    return path
    except ValueError:
        raise ValueError(f"No patterns matching {pattern_sets} found for download.")
    except Exception as e:
        raise Exception(f"Error downloading {type} from {name}: {e}")

if __name__ == "__main__":
    # Validate token before proceeding
    if not validate_token():
        print("ERROR: Failed to validate Hugging Face token. Please check your token and model access permissions.")
        sys.exit(1)

    setup_env()
    cache_dir = os.getenv("HF_HOME")
    model_name, model_revision = os.getenv("MODEL_NAME"), os.getenv("MODEL_REVISION") or None
    tokenizer_name, tokenizer_revision = os.getenv("TOKENIZER_NAME") or model_name, os.getenv("TOKENIZER_REVISION") or model_revision

    try:
        model_path = download(model_name, model_revision, "model", cache_dir)

        metadata = {
            "MODEL_NAME": model_path,
            "MODEL_REVISION": os.getenv("MODEL_REVISION"),
            "QUANTIZATION": os.getenv("QUANTIZATION"),
        }

        # if os.getenv("TENSORIZE") == "1": TODO: Add back once tensorizer is ready
        #     serialized_uri, tensorizer_num_gpus, dtype = tensorize_model(model_path)
        #     metadata.update({
        #         "MODEL_NAME": serialized_uri,
        #         "TENSORIZER_URI": serialized_uri,
        #         "TENSOR_PARALLEL_SIZE": tensorizer_num_gpus,
        #         "DTYPE": dtype
        #     })

        tokenizer_path = download(tokenizer_name, tokenizer_revision, "tokenizer", cache_dir)
        metadata.update({
            "TOKENIZER_NAME": tokenizer_path,
            "TOKENIZER_REVISION": tokenizer_revision
        })

        with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
            json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)

    except Exception as e:
        print(f"ERROR during model download/setup: {e}")
        sys.exit(1)