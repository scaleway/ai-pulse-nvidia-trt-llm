# Setup
## Prerequisites
- Scaleway Account with the following
  - Quotas (cp_servers_type_H100_2_80G) enabled for H100-PCIE 
  - IAM Token with at least permissions below :
    - InstancesFullAccess
  - SSH Keys defined at project Level. See [here](https://www.scaleway.com/en/docs/console/project/how-to/create-ssh-key/)
- Terraform >= 1.3.X See [here](https://www.terraform.io/downloads.html).
# Infrastructure Layer
## Deployment
1. Configure your environment variables, so that Scaleway terraform providers can interact with Scaleway backbone :
   - SCW_ACCESS_KEY
   - SCW_SECRET_KEY
   - SCW_DEFAULT_PROJECT_ID
   - SCW_DEFAULT_ORGANIZATION_ID

![Astuce icon](./images/common/astuce_icon.png)  See [here](https://registry.terraform.io/providers/scaleway/scaleway/latest/docs#environment-variables) for more details.

2. Configure your terraform variables by renaming the infrastructure/terraform.tfvars.template -> infrastructure/terraform.tfvars
- Update the users_ips_lists that is used to restrict  access to your instance to a list of IP.

![Astuce icon](./images/common/astuce_icon.png) You can set this value at 0.0.0.0 to grant access whatever the IP.

3. Deploy the infrastructure using the command below
```
terraform -chdir=sources/infrastructure init &&  terraform -chdir=sources/infrastructure apply
```
- This script will rely on terraform to deploy an **H100-2-80G** that will be reached using SSH connection.
- A root volume of 500 GB will be deployed and a [scratch volume](https://www.scaleway.com/en/docs/compute/gpu/how-to/use-scratch-storage-h100-instances/)(i.e. :  NVMe disks which are fine-tuned for high-speed data access) of 3.9 Tb will be added to this instance.
- The Scratch volume will be automatically mount on **/scratch-volume** mountpoint that will be used to store datasets further.

![Infrastructure Setup](images/setup/infra_setup.png)
## Validation
Here we will show how you can connect to the Instance and validate that the 
1. Connect to your Scaleway Console 
2. Retrieve your instance public ip from the console
![Public IP retrieving](images/setup/public_ip_ssh.png)
3. Connect to your instance using ssh client
```
ssh ubuntu@$PUBLIC_IP
```
4. Validate that you have the right configuration (2*H100-PCIE) and right drivers using the nvidia-smi command
```
nvidia-smi
```
![nvidia smi capabilities](../docs/images/setup/nvidia-smi-capabilities.png)

# Software Layer
## Description
As explained before, TensorRT-LLM will be used within Triton Inference Server to deploy the engines it has generated.
In this chapter, we will explain how to deploy TensorRT-LLM backend within the Triton Inference Server and also how to get the TensorTRT Toolbox.

## Triton Inference Server
We will use here the [official docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) provided by Nvidia ?


## TensorRT-LLM
### Usage overview
![tensor rrt usage](images/setup/tensor_rrt_llm_usage.png)

### Setup
We could have downloaded directly TRT-LLM from the [official github repository](https://github.com/NVIDIA/TensorRT-LLM) however because we will rely on Triton , we will directly download it from [TensorRT-LLM backend repository](https://github.com/triton-inference-server/tensorrtllm_backend.git) on which it is added as submodules.

1. Clone the repository on the scratch volume
```
git -C /scratch clone https://github.com/triton-inference-server/tensorrtllm_backend.git  
```
2. Download Tensor RT LLM through the Git modules
```
cat <<'EOF'> /scratch/tensorrtllm_backend/.gitmodules 
[submodule "tensorrt_llm"]
        path = tensorrt_llm
        url = https://github.com/NVIDIA/TensorRT-LLM.git
EOF
```
```
git  -C /scratch/tensorrtllm_backend submodule update --init --recursive 
git  -C /scratch/tensorrtllm_backend lfs install
git  -C /scratch/tensorrtllm_backend lfs pull  
```


### TensorRT-LLM Backend integration in Triton
#### Server
At the moment of writing this document, triton docker image does not yet contains all the resources required to  launch TRT-LLM builded model.
We need to add it within the official triton image using the process below : 

1. Create a folder to save docker files
```
mkdir /scratch/docker
```
2. Create the Dockerfile that will be used to deploy TRT-LLM based on the [Triton official docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
```
cat<<'EOF'>/scratch/docker/Dockerfile.server
ARG TAG=23.10-trtllm-python-py3
FROM nvcr.io/nvidia/tritonserver:${TAG}

ENV DEBIAN_FRONTEND noninteractive

# Install TRT LLM
RUN pip install --no-cache-dir \
        git+https://github.com/NVIDIA/TensorRT-LLM.git

# Dependency copy
RUN mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/ && \
    cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/

RUN pip install  --no-cache-dir \
    datasets==2.14.5 \
    rouge_score~=0.1.2 \
    sentencepiece~=0.1.99

RUN mkdir -p /workspace

WORKDIR /workspace
EOF
```
![Astuce icon](./images/common/astuce_icon.png) Scaleway H100 instances are provided with docker pre-installed .

3. Build the docker image
```
sudo docker build -t tritonserver-aipulse:23.10 -f /scratch/docker/Dockerfile.server .
```
4. Then we can start a container based on the image and bind the workspace volume to our scratch volume location
```
#!/bin/bash
IMAGE_NAME=tritonserver-aipulse:23.10
WORK_DIR=/scratch
sudo docker run -d                                      \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --name triton_server                            \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v $WORK_DIR:/workspace                         \
        $IMAGE_NAME bash 
```
5. You can then ensure than your Triton server is running
```
sudo docker ps
```
![Container List Setup](images/setup/container_list.png)

## Next Step
[Model's weights download and preparation](02-model_preparation.md) 

# Resources
- [TRT LLM guides](https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md)
- [Tensor RRT LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)
- [Triton Inference Server NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
- [Triton Inference Server github](https://github.com/triton-inference-server/server)
