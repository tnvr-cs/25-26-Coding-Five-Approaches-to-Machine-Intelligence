import os
import glob

# The folder where your images are stored
DATA_DIR = "grocery_data"

print(f"🧹 Scanning '{DATA_DIR}' for corrupt or empty files...")

deleted_count = 0

# Walk through all folders
for root, dirs, files in os.walk(DATA_DIR):
    for filename in files:
        file_path = os.path.join(root, filename)
        
        try:
            # CHECK 1: Is the file empty? (0 bytes)
            if os.path.getsize(file_path) == 0:
                print(f"❌ Deleting empty file: {filename}")
                os.remove(file_path)
                deleted_count += 1
                continue
            
            # CHECK 2: Is it actually an image?
            # We check the first few bytes to see if it's a valid image format
            with open(file_path, "rb") as f:
                header = f.read(10)
                # Valid headers for JPG, PNG, BMP, GIF
                if not (header.startswith(b'\xff\xd8') or 
                        header.startswith(b'\x89PNG') or 
                        header.startswith(b'BM') or 
                        header.startswith(b'GIF')):
                    print(f"⚠️ Deleting invalid format: {filename}")
                    os.remove(file_path)
                    deleted_count += 1
                    
        except Exception as e:
            print(f"Error checking {filename}: {e}")

print("-" * 30)
print(f"✅ Cleanup Complete. Deleted {deleted_count} bad files.")
