![ai pulse banner](./images/common/ai-pulse-banner.jpeg)

# FP 8 Quantization
Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).
- https://huggingface.co/docs/optimum/concept_guides/quantization
- https://en.wikipedia.org/wiki/Quantization
## Model Build
For the FP8 quantization, we build the engine using the command below. In our case, we apply FP8 quantization for the KV cache for further optimizations.
1. Run the `build.py` script to compile the TRT-LLM engines.
```
 docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        tritonserver-aipulse:23.10   python  /workspace/tensorrtllm_backend/tensorrt_llm/examples/llama/build.py --model_dir /workspace/meta/hf-weights/7B-chat  \
    --dtype float16 \
    --use_gpt_attention_plugin float16  \
    --paged_kv_cache \
    --remove_input_padding \
    --use_gemm_plugin float16  \
    --output_dir "/workspace/trt-engines/llama_7b/fp8/1-gpu"  \
    --max_input_len 2048 --max_output_len 512 \
    --use_rmsnorm_plugin float16  \
    --enable_context_fmha \
    --use_inflight_batching \
    --enable_fp8 \
    --fp8_kv_cache
```
2. The following outputs show the engines sizes of Llama2 7B before and after FP8 quantization. We divide their sizes by 2 goining from 13GB  for the fp16 to 6.6GB for the fp8 one.

**FP16**
```
ls -lsh /scratch/trt-engines/llama_7b/fp16/1-gpu/
```
```
total 13G
4,0K -rw-r--r-- 1 root root 1,3K déc.   5 08:27 config.json
 13G -rw-r--r-- 1 root root  13G déc.   5 08:27 llama_float16_tp1_rank0.engine
 52K -rw-r--r-- 1 root root  49K déc.   5 08:27 model.cache
```
**FP8**
```
ls -lsh /scratch/trt-engines/llama_7b/fp8/1-gpu/
```
```
total 6,6G
4,0K -rw-r--r-- 1 root root 1,3K déc.   5 17:11 config.json
6,6G -rw-r--r-- 1 root root 6,6G déc.   5 17:11 llama_float16_tp1_rank0.engine
236K -rw-r--r-- 1 root root 236K déc.   5 17:11 model.cache

```

## Evaluation
### FP16 Model
```
 docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        tritonserver-aipulse:23.10 python /workspace/tensorrtllm_backend/tensorrt_llm/examples/llama/run.py --engine_dir=/workspace/trt-engines/llama_7b/fp16/1-gpu \
        --max_output_len 100 --tokenizer_dir /workspace/meta/llama_models \
        --input_text "How do I count in French ? 1 un"
```
**The following result is provided**
```
Running the float16 engine ...
Input: "How do I count in French ? 1 un"
Output: ", 2 deux, 3 trois, 4 quatre, 5 cinq, 6 six, 7 sept, 8 huit, 9 neuf, 10 dix.
```
### FP8 Model
```
 docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        tritonserver-aipulse:23.10 python /workspace/tensorrtllm_backend/tensorrt_llm/examples/llama/run.py  --engine_dir=/workspace/trt-engines/llama_7b/fp8/1-gpu \
        --max_output_len 100 \
        --tokenizer_dir /workspace/meta/llama_models \
        --input_text "How do I count in French ? 1 un"
```
**The following result is provided**
```
Running the float16 engine ...
Input: "How do I count in French ? 1 un"
Output: ", 2 deux, 3 trois, 4 quatre, 5 cinq, 6 six, 7 sept, 8 huit, 9 neuf, 10 dix.
```

## Performance 
### FP16 Model
1. Run the triton inference server
```
docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        -d                                              \
        --name triton_server_quantize_inflight                \
        tritonserver-aipulse:23.10 tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp16/inflight/
```
2. The following command can be run to ensure that the Triton server is up 

```
docker logs -f triton_server_quantize_inflight
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
        --max_input_len 2048 \
        --op_stats_csv h100_llama-7b-fp16_IFB.csv \
        --dataset /workspace/datasets/mini_cnn_eval.json  \
        --tokenizer_dir /workspace/meta/llama_models \
       --tokenizer_type llama
```
3. The following result is output
```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 9345.807 ms
[INFO] Total request latencies: 502845.79800000007 ms
980
+----------------------------+---------+
|            Stat            |  Value  |
+----------------------------+---------+
|        Requests/Sec        |  10.49  |
|       OP tokens/sec        | 435.08  |
|     Avg. latency (ms)      | 5131.08 |
|      P99 latency (ms)      | 9305.05 |
|      P90 latency (ms)      | 8714.54 |
| Avg. IP tokens per request | 799.66  |
| Avg. OP tokens per request |  41.49  |
|   Avg. InFlight requests   |  53.86  |
|     Total latency (ms)     | 9345.36 |
|       Total requests       |  98.00  |
+----------------------------+---------+
Expected op tokens 41.49
```

### FP8 Model
#### Model Repository
We need to first create the model repository that will be used by Triton to locate the model to launch. We will use the fp16 (1-gpu) as basis here.
1. Create the fp8 model repository
```
mkdir -p /scratch/triton_model_repo/llama_7b/fp8
```
2. Copy the content of the fp16 inflight model into the fp8 one
```
cp -R /scratch/triton_model_repo/llama_7b/fp16/inflight/* /scratch/triton_model_repo/llama_7b/fp8/.
```
3. Update the value of the gpt_model_path
```
sed -i 's#/workspace/trt-engines/llama_7b/fp16/1-gpu#/workspace/trt-engines/llama_7b/fp8/1-gpu#' /scratch/triton_model_repo/llama_7b/fp8/tensorrt_llm/config.pbtxt
``` 
#### Run The Inference Server
1. Stop the previous inference server
```
docker container stop triton_server_quantize_inflight
```

2. Run the triton inference server on the **FP8 model**
```
docker run                                       \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        -v /scratch:/workspace                          \
        -d                                              \
        --name triton_server_quantize_fp8                \
        tritonserver-aipulse:23.10 tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp8
```

3. The following command can be run to ensure that the Triton server is up 

```
docker logs -f triton_server_quantize_fp8
```

4. We run the test using the [identity_test.py script provided by TensorRT](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/tools/inflight_batcher_llm/identity_test.py).

```
docker run                                        \
        --runtime=nvidia                                \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        --name triton_client                            \
        -v /scratch:/workspace                          \
        tritonclient-aipulse:23.10 python /usr/local/src/benchmark/scripts/identity_test.py -u localhost:8001  -i grpc \
        --max_input_len 2048 \
        --op_stats_csv h100_llama-7b-fp8_IFB.csv \
        --dataset /workspace/datasets/mini_cnn_eval.json  \
        --tokenizer_dir /workspace/meta/llama_models \
       --tokenizer_type llama
```

3. The following result is output

```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 6854.241 ms
[INFO] Total request latencies: 372122.97499999986 ms
979
+----------------------------+---------+
|            Stat            |  Value  |
+----------------------------+---------+
|        Requests/Sec        |  14.30  |
|       OP tokens/sec        | 593.25  |
|     Avg. latency (ms)      | 3797.17 |
|      P99 latency (ms)      | 6815.05 |
|      P90 latency (ms)      | 6436.62 |
| Avg. IP tokens per request | 799.66  |
| Avg. OP tokens per request |  41.49  |
|   Avg. InFlight requests   |  54.35  |
|     Total latency (ms)     | 6853.77 |
|       Total requests       |  98.00  |
+----------------------------+---------+
Expected op tokens 41.49
```
**The performance is better with the quantized model than the fp16 one.**

## Next Steps
### Clean up
From the local computer , clean up the infrastructure and components
```
terraform -chdir=sources/infrastructure destroy
```
### Nvidia TRT Presentation
[TensorRT-LLM-AI-Pulse-Nov-2023](./nvidia/TensorRT-LLM-AI-Pulse-Nov-2023.pdf)
