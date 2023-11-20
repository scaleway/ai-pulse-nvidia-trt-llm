# FP 8 Quantization

## Model Build

For the FP8 quantization, we build the engine using the command below. In our case, we apply FP8 quantization for the KV cache for further optimizations.

```bash
cd /workspace/tensorrtllm_backend/tensorrt_llm/examples/llama

python3 build.py --model_dir /workspace/meta/hf-weights/7B-chat  \
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

The following outputs show the engines sizes of Llama2 7B before and after FP8 quantization. We divide their sizes by 2 goining from 13GB to 6.6GB.
```
root@nvidia-workshop-gpu-instance:/workspace# ls -lrtsh /workspace/trt-engines/llama_7b/fp16/1-gpu
total 13G
4.0K -rw-r--r-- 1 root root 1.3K Nov 15 18:57 config.json
 13G -rw-r--r-- 1 root root  13G Nov 15 18:57 llama_float16_tp1_rank0.engine
 52K -rw-r--r-- 1 root root  49K Nov 15 18:57 model.cache
```

```
root@nvidia-workshop-gpu-instance:/workspace# ls -lrtsh /workspace/trt-engines/llama_7b/fp8/1-gpu
total 6.6G
4.0K -rw-r--r-- 1 root root 1.3K Nov 15 19:10 config.json
6.6G -rw-r--r-- 1 root root 6.6G Nov 15 19:10 llama_float16_tp1_rank0.engine
236K -rw-r--r-- 1 root root 236K Nov 15 19:10 model.caches
```

## Evaluation

```bash
python run.py   --engine_dir=/workspace/trt-engines/llama_7b/fp16/1-gpu \
                --max_output_len 100 \
                --tokenizer_dir /workspace/meta/llama_models \
                --input_text "How do I count in French ? 1 un"
```

```
Running the float16 engine ...
Input: "How do I count in French ? 1 un"
Output: ", 2 deux, 3 trois, 4 quatre, 5 cinq, 6 six, 7 sept, 8 huit, 9 neuf, 10 dix, 11 onze, 12 douze, 13 treize, 14 quatorze, 15 quinze, 16 seize, 17 dix-sept, 18 dix-huit, 19 dix-neuf"
```

```bash
python run.py   --engine_dir=/workspace/trt-engines/llama_7b/fp8/1-gpu \
                --max_output_len 100 \
                --tokenizer_dir /workspace/meta/llama_models \
                --input_text "How do I count in French ? 1 un"
```
```
Running the float16 engine ...
Input: "How do I count in French ? 1 un"
Output: ", 2 deux, 3 trois, 4 quatre, 5 cinq, 6 six, 7 sept, 8 huit, 9 neuf, 10 dix, 11 onze, 12 douze, 13 treize, 14 quatorze, 15 quinze, 16 seize, 17 dix-sept, 18 dix-huit, 19 dix-neuf"
```

## Performance 

```bash
tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp16/inflight/
```

```bash
python3 tools/inflight_batcher_llm/identity_test.py \
    -i grpc \
    --max_input_len 2048 \
    --op_stats_csv h100_llama-7b-fp16_IFB.csv \
    --dataset tools/dataset/mini_cnn_eval.json  \
    --tokenizer_dir /workspace/meta/llama_models \
    --tokenizer_type llama
```

```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 9316.631 ms
[INFO] Total request latencies: 501628.17299999995 ms
980
+----------------------------+---------+
|            Stat            |  Value  |
+----------------------------+---------+
|        Requests/Sec        |  10.52  |
|       OP tokens/sec        | 436.44  |
|     Avg. latency (ms)      | 5118.65 |
|      P99 latency (ms)      | 9296.89 |
|      P90 latency (ms)      | 8705.40 |
| Avg. IP tokens per request | 799.66  |
| Avg. OP tokens per request |  41.49  |
|   Avg. InFlight requests   |  53.89  |
|     Total latency (ms)     | 9316.21 |
|       Total requests       |  98.00  |
+----------------------------+---------+
Expected op tokens 41.49
```

```bash
tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp8
```

```bash
python3 tools/inflight_batcher_llm/identity_test.py \
    -i grpc \
    --max_input_len 2048 \
    --op_stats_csv h100_llama-7b-fp8_IFB.csv \
    --dataset tools/dataset/mini_cnn_eval.json  \
    --tokenizer_dir /workspace/meta/llama_models \
    --tokenizer_type llama
```

```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 7001.744 ms
[INFO] Total request latencies: 380716.78800000006 ms
980
+----------------------------+---------+
|            Stat            |  Value  |
+----------------------------+---------+
|        Requests/Sec        |  14.00  |
|       OP tokens/sec        | 580.75  |
|     Avg. latency (ms)      | 3884.87 |
|      P99 latency (ms)      | 6980.63 |
|      P90 latency (ms)      | 6595.25 |
| Avg. IP tokens per request | 799.66  |
| Avg. OP tokens per request |  41.49  |
|   Avg. InFlight requests   |  54.42  |
|     Total latency (ms)     | 7001.33 |
|       Total requests       |  98.00  |
+----------------------------+---------+
Expected op tokens 41.49
```
