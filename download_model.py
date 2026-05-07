import os
from google.cloud import storage

def main():
    bucket_name = os.environ["MODEL_BUCKET"]
    object_name = os.environ["MODEL_OBJECT"]   # e.g. models/student-merged-q5_k_m.gguf
    dst_path    = os.environ.get("MODEL_PATH", "/mnt/models/model.gguf")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        print(f"[model] exists: {dst_path} ({os.path.getsize(dst_path)} bytes)")
        return

    print(f"[model] downloading gs://{bucket_name}/{object_name} -> {dst_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.download_to_filename(dst_path)
    print("[model] download done")

if __name__ == "__main__":
    main()