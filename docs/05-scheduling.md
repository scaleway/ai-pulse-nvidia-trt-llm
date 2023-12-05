# Batching Scheduling
For each of the steps here, we will need to create the Triton model repository as explained [here](03-Triton.md#models-repository). As a reminder, each of these runs needs to have the following folder structure:
```
├── ensemble
│   ├── 1
│   └── config.pbtxt
├── postprocessing
│   ├── 1
│   └── config.pbtxt
├── preprocessing
│   ├── 1
│   └── config.pbtxt
└── tensorrt_llm
    ├── 1
    └── config.pbtxt
```
Depending on the batching policy (static or in-flight batching), the file `tensorrt_llm/config.pbtxt` needs to be adapted accordingly. The parameter `gpt_model_type` should be set to:
- `"V1"` for *static* scheduling
- `"inflight_fused_batching"` for *in-flight batching*

We can also change the value of the parameter `batch_scheduler_policy` to define how requests are scheduled for execution. The possible values are:
- `"max_utilization"` to maximize the utilization of the GPUs by aggressively scheduling the requests.
- `"guaranteed_completion"` to schedule requests only when the memory allocated is sufficient to process all active requests consumption.

Refer to TensorRT-LLM [documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/docs/source/batch_manager.md#gptmanager-design) for further details.

### Static Batching
#### Model Repository
1. Create the triton model repository that will holds all the triton model 
```
mkdir -p /scratch/triton_model_repo/llama_7b/fp16/no-inflight
```
2. Initiates the  folder with template files from tensorrtllm_backend
```
cp -R /scratch/tensorrtllm_backend/all_models/inflight_batcher_llm/* /scratch/triton_model_repo/llama_7b/fp16/no-inflight/.
```
3. Update the preprocessing template values
```
sed -i 's#${tokenizer_dir}#/workspace/meta/llama_models#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/preprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#llama#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/preprocessing/config.pbtxt
```
4. Update the postprocessing template values
```
sed -i 's#${tokenizer_dir}#/workspace/meta/llama_models#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/postprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#llama#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/postprocessing/config.pbtxt
```
5. Update the tensorrt_llm template values
```
sed -i 's#${decoupled_mode}#False#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/tensorrt_llm/config.pbtxt
sed -i 's#inflight_fused_batching#V1#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/tensorrt_llm/config.pbtxt
sed -i 's#${engine_dir}#/workspace/trt-engines/llama_7b/fp16/1-gpu#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/tensorrt_llm/config.pbtxt
sed -i 's#${max_tokens_in_paged_kv_cache}##' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/tensorrt_llm/config.pbtxt
sed -i 's#${batch_scheduler_policy}#guaranteed_completion#' /scratch/triton_model_repo/llama_7b/fp16/no-inflight/tensorrt_llm/config.pbtxt
```

### Run the Triton inference server
1. Run the Server using the following command
   
```
docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        -d                                              \
        --name triton_server_scheduling                \
        tritonserver-aipulse:23.10 tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp16/no-inflight/
```
2. Use The [identity_test_python_vs_trtllm.py](./scripts/identity_test_python_vs_trtllm.py) script to call the server and measure the latency

```
sudo docker run                                        \
        --runtime=nvidia                                \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        --name triton_client                            \
        -v /scratch:/workspace                          \
        tritonclient-aipulse:23.10 python /usr/local/src/benchmark/scripts/identity_test.py -u localhost:8001  -i grpc \
    -u localhost:8001 \
    --max_input_len 2048 \
    --op_stats_csv h100_llama-7b-fp16_V1.csv \
    --dataset tools/dataset/mini_cnn_eval.json  \
    --tokenizer_dir /workspace/meta/llama_models/ \
    --tokenizer_type llama
```