sudo apt update
sudo apt install blobfuse2 -y
mkdir -p ~/.blobfuse
nano ~/.blobfuse/config.yaml
sudo mkdir -p /mnt/blob
sudo chown azureuser:azureuser /mnt/blob

# Create mount point
sudo mkdir -p /mnt/blob
sudo chown azureuser:azureuser /mnt/blob


# mounting Blob Storage
blobfuse2 mount /mnt/blob --config-file ~/.blobfuse/config.yaml
echo "Blob storage mounted at /mnt/blob"


