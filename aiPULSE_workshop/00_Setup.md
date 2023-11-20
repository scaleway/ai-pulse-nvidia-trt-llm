# Setup the environment

Pull [Triton Inference Server container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) from NGC. You need the support with TensorRT-LLM backend.

```bash
docker pull nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

We will use the public tutorial available on the Triton Inference Server [github](https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md).

```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
```


Make sure to modify the `.gitmodules` file to the following:

```
[submodule "tensorrt_llm"]
        path = tensorrt_llm
        url = https://github.com/NVIDIA/TensorRT-LLM.git
```
and run the following:

```bash
git submodule update --init --recursive
git lfs install
git lfs pull
```

Use this script to run the Triton Inference server container:

```bash
#!/bin/bash
CONTAINER_NAME=nvcr.io/nvidia/tritonserver
TAG=23.10-trtllm-python-py3
WORK_DIR=/data
docker run                                              \
        --runtime=nvidia                                \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v $WORK_DIR:/workspace                         \
        $CONTAINER_NAME:$TAG bash
```

Install TensorRT-LLM python package inside the tritonserver container. We will use it to generate engines

```bash
pip install git+https://github.com/NVIDIA/TensorRT-LLM.git
mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/
cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/
```

To avoid the installation of the required packages during this demo, we provide dockerfiles to build customized containers: one for the [server](./docker/Dockerfile.server) and one for the [client](./docker/Dockerfile.client). To build these containers, use the following commands:

```bash
docker build -t tritonserver-aipulse:23.10 -f docker/Dockerfile.server .
```
```bash
docker build -t tritonserver-sdk-aipulse:23.10 -f docker/Dockerfile.client .
```


## Next Step
[Model's weights download and preparation](01_Models.md) 