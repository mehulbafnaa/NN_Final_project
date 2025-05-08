#!/bin/bash

# Script to download and extract League of Legends dataset
# Usage: ./download_lol_dataset.sh

echo "Starting download of League of Legends dataset..."

# Create download directory if it doesn't exist
mkdir -p ~/Downloads

# Download the dataset
curl -L -o ~/Downloads/lol-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/soumikrakshit/lol-dataset

# Check if download was successful
if [ $? -eq 0 ]; then
  echo "Download completed successfully!"
  
  # Create extraction directory
  mkdir -p ~/Downloads/lol-dataset
  
  # Extract the ZIP file
  unzip ~/Downloads/lol-dataset.zip -d ~/Downloads/lol-dataset
  
  # Check if extraction was successful
  if [ $? -eq 0 ]; then
    echo "Extraction completed successfully!"
    echo "Files extracted to: ~/Downloads/lol-dataset"
    echo "Contents of the extracted directory:"
    ls -la ~/Downloads/lol-dataset
  else
    echo "Error: Failed to extract the ZIP file."
    exit 1
  fi
else
  echo "Error: Failed to download the dataset."
  exit 1
fi

echo "Process completed!"
