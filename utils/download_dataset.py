import os
import subprocess

def download_with_wget():
    os.makedirs('dataset_parts', exist_ok=True)
    base_url = "https://huggingface.co/datasets/akameswa/trash/resolve/main/zip/dataset.zip."
    
    for i in range(1, 25):
        part = str(i).zfill(3)
        url = f"{base_url}{part}"
        output_file = f"dataset_parts/dataset.zip.{part}"
        
        print(f"Downloading part {part}...")
        cmd = [
            "wget",
            "--header", "Authorization: Bearer " + os.getenv('HF_TOKEN', ''),
            "-O", output_file,
            url
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    download_with_wget()