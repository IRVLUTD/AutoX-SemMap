# AutoX-SemMap Docker Instructions

Prior to following these instructions, make sure you have pulled this repo.

## Install Docker

### 1. For computers without a Nvidia GPU

Install Docker and grant user permission:
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install additional packages:
```
sudo apt update && sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

### 2. For computers with Nvidia GPUs

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install Nvidia Container Toolkit (Nvidia GPU Driver
should be installed already).

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```
sudo apt update && sudo apt install nvidia-container-toolkit
```
Configure Docker runtime and restart Docker daemon.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Build and Run Docker Image

Inside the AutoX-SemMap folder, allow remote X connection:
```
xhost +
```

Navigate to the Docker folder:
```
cd AutoX-SemMap/docker/
```

Build the Docker Image from the Dockerfile:
```
docker build -f Dockerfile -t semmap:latest .
```

Run the Docker Image:
```
docker run -it --rm \
    --gpus all \
    --env="ROS_MASTER_URI=http://{ROS_MASTER_URI}:11311" \
    --env="ROS_IP={CONTAINER_ROS_IP}" \
    --env="DISPLAY=$DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    semmap:latest
```

## Testing AutoX-SemMap

Once inside the Docker container, follow the instructions in the main section of the repository to test **AutoX-SemMap**.