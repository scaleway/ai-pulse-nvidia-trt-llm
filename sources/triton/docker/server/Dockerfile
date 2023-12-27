ARG TAG=23.10-trtllm-python-py3
FROM nvcr.io/nvidia/tritonserver:${TAG}

ENV DEBIAN_FRONTEND noninteractive

# Install TRT LLM
RUN pip install --no-cache-dir \
        git+https://github.com/NVIDIA/TensorRT-LLM.git@release/0.5.0

# Dependency copy
RUN mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/ && \
    cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/

RUN pip install  --no-cache-dir \
    datasets==2.14.5 \
    rouge_score~=0.1.2 \
    sentencepiece~=0.1.99

RUN mkdir -p /workspace

WORKDIR /workspace