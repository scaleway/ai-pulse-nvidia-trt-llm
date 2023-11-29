ARG TAG=23.10-trtllm-python-py3
FROM nvcr.io/nvidia/tritonserver:${TAG}

ENV DEBIAN_FRONTEND noninteractive

RUN pip install  --no-cache-dir \
    regex \
    fire \
    tritonclient[all] \
    transformers==4.31.0 \
    pandas \
    tabulate \
    datasets==2.14.5 \
    rouge_score~=0.1.2 \
    sentencepiece~=0.1.99

RUN mkdir -p /workspace

WORKDIR /workspace