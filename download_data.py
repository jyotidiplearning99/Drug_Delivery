"""Download LAI dataset from Zenodo"""
import requests
import zipfile
from pathlib import Path
import os

def download_zenodo_dataset(record_id="7309141", output_dir="data"):
    print(f"Downloading from Zenodo record: {record_id}")
    
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        record = response.json()
        
        print(f"\nDataset: {record['metadata']['title']}")
        print(f"DOI: {record['doi']}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        files = record['files']
        print(f"\nFound {len(files)} file(s)")
        
        for file_info in files:
            filename = file_info['key']
            download_url = file_info['links']['self']
            
            # Extract just the filename without directory structure
            clean_filename = os.path.basename(filename)
            destination = Path(output_dir) / clean_filename
            
            print(f"\nDownloading {filename} as {clean_filename}...")
            
            file_response = requests.get(download_url, stream=True, timeout=300)
            file_response.raise_for_status()
            
            total_size = int(file_response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='')
            
            print(f"\n✓ Downloaded {clean_filename}")
            
            if clean_filename.endswith('.zip'):
                print(f"Extracting {clean_filename}...")
                with zipfile.ZipFile(destination, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"✓ Extracted")
        
        print("\n✓ Download complete!")
        
        # List contents of data directory
        print("\nContents of data directory:")
        for item in Path(output_dir).iterdir():
            print(f"  - {item.name}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    download_zenodo_dataset()
