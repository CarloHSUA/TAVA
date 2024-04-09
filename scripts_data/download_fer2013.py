import gdown
import zipfile
import os

# URL of the zip file on Google Drive
url = "https://drive.google.com/uc?id=1xliH99EwUzxyNn2ha6nbeuVv0brhDsax"

# Path to save the downloaded zip file
zip_file_path = "../datasets/fer_2013.zip"

# Path to extract the contents of the zip file
extract_folder_path = "../datasets"

# Download the zip file
gdown.download(url, zip_file_path, quiet=False)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

# Remove the downloaded zip file
os.remove(zip_file_path)