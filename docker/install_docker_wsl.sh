#!/usr/bin/env bash
# Install Docker Engine + the Compose plugin in WSL Ubuntu 22.04 (asks for your sudo password). Run once, in an
# Ubuntu terminal (not from PowerShell):
#   bash /mnt/c/Users/Bieber/Desktop/robot_simulation/docker/install_docker_wsl.sh
# Then in PowerShell: wsl --terminate Ubuntu-22.04 (the docker group applies from the next WSL session).
# In WSL the GPU reaches containers through /dev/dxg and /usr/lib/wsl (docker/compose.wsl.yaml), so the NVIDIA
# Container Toolkit is not needed here; a Linux server needs it instead (docker/README.md).
set -euo pipefail
# The ROS 1 repository has no jammy release, so its "does not have a Release file" error stops apt-get update
# (and this script). Nothing could ever be installed from it: it is switched off by renaming, reversible.
ros=/etc/apt/sources.list.d/ros-latest.list
if [ -f "$ros" ] && grep -q "packages.ros.org/ros/ubuntu jammy" "$ros"; then
    sudo mv "$ros" "$ros.disabled"
    echo "switched off the broken ROS 1 repository: $ros -> $ros.disabled"
fi
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker
sudo docker run --rm hello-world > /dev/null && echo "Docker works."
echo "Now in PowerShell: wsl --terminate Ubuntu-22.04 (then docker runs without sudo)."
