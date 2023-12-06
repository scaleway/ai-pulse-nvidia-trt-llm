![ai pulse banner](./images/common/ai-pulse-banner.jpeg)

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
2. We run the test using the [identity_test.py script provided by TensorRT](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/tools/inflight_batcher_llm/identity_test.py).

```
 docker run                                        \
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
    --dataset /workspace/datasets/mini_cnn_eval.json  \
    --tokenizer_dir /workspace/meta/llama_models/ \
    --tokenizer_type llama
```

```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 12154.388 ms
[INFO] Total request latencies: 615980.253 ms
979
+----------------------------+----------+
|            Stat            |  Value   |
+----------------------------+----------+
|        Requests/Sec        |   8.06   |
|       OP tokens/sec        |  334.54  |
|     Avg. latency (ms)      | 6285.51  |
|      P99 latency (ms)      | 12114.01 |
|      P90 latency (ms)      | 10760.40 |
| Avg. IP tokens per request |  799.66  |
| Avg. OP tokens per request |  41.49   |
|   Avg. InFlight requests   |  50.74   |
|     Total latency (ms)     | 12153.90 |
|       Total requests       |  98.00   |
+----------------------------+----------+
Expected op tokens 41.49

```

### Inflight Batching
#### Models Repository
1. We copy the model from the static batching
```
mkdir /scratch/triton_model_repo/llama_7b/fp16/inflight
cp -R /scratch/triton_model_repo/llama_7b/fp16/no-inflight/* /scratch/triton_model_repo/llama_7b/fp16/inflight/.
```
2. We update the value of the gpt_model_type -> inflight_fused_batching
```
sed -i 's#V1#inflight_fused_batching#' /scratch/triton_model_repo/llama_7b/fp16/inflight/tensorrt_llm/config.pbtxt
```
### Run the Triton inference server
1. Stop the previous container
```
docker container stop triton_server_scheduling
```
2. Run the Server using the following command
```
docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        -d                                              \
        --name triton_server_scheduling_inflight                \
        tritonserver-aipulse:23.10 tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp16/inflight/
```
3. We run the test using the [identity_test.py script provided by TensorRT](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/tools/inflight_batcher_llm/identity_test.py).

```
 docker run                                        \
        --runtime=nvidia                                \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        --name triton_client                            \
        -v /scratch:/workspace                          \
        tritonclient-aipulse:23.10 python /usr/local/src/benchmark/scripts/identity_test.py -u localhost:8001  -i grpc \
    -u localhost:8001 \
    --max_input_len 2048 \
    --op_stats_csv h100_llama-7b-fp16_IFB.csv \
    --dataset /workspace/datasets/mini_cnn_eval.json  \
    --tokenizer_dir /workspace/meta/llama_models/ \
    --tokenizer_type llama
```
4. The result should look like the following
```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 9315.312 ms
[INFO] Total request latencies: 501834.1599999999 ms
979
+----------------------------+---------+
|            Stat            |  Value  |
+----------------------------+---------+
|        Requests/Sec        |  10.52  |
|       OP tokens/sec        | 436.51  |
|     Avg. latency (ms)      | 5120.76 |
|      P99 latency (ms)      | 9277.44 |
|      P90 latency (ms)      | 8689.87 |
| Avg. IP tokens per request | 799.66  |
| Avg. OP tokens per request |  41.49  |
|   Avg. InFlight requests   |  53.92  |
|     Total latency (ms)     | 9314.86 |
|       Total requests       |  98.00  |
+----------------------------+---------+
Expected op tokens 41.49
```
## Next Steps
### Clean up
```
docker container stop triton_server_scheduling 
docker container stop triton_server_scheduling_inflight
```
[Quantization](06-quantization.md)