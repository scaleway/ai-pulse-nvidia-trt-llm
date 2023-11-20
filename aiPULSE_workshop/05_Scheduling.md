# Batching Scheduling 

For each of these runs, we create the model repository as explained in the [Triton section](./03_Triton.md). As a reminder, each of these runs needs to have the following folder structure:
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

The file `tensorrt_llm/config.pbtxt` should be modified as follow:
```
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "V1"
  }
}
...
parameters: {
  key: "batch_scheduler_policy"
  value: {
    string_value: "guaranteed_completion"
  }
}
```


```bash
tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp16/no-inflight/
```

```bash
python3 tools/inflight_batcher_llm/identity_test.py \
    -i grpc \
    -u 192.168.1.3:8001 \
    --max_input_len 2048 \
    --op_stats_csv h100_llama-7b-fp16_V1.csv \
    --dataset tools/dataset/mini_cnn_eval.json  \
    --tokenizer_dir /workspace/meta/llama_models/ \
    --tokenizer_type llama
```

```
Tokens per word:  1.471
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 98 prompts.
[INFO] Total Latency: 12173.135 ms
[INFO] Total request latencies: 617150.9320000001 ms
980
+----------------------------+----------+
|            Stat            |  Value   |
+----------------------------+----------+
|        Requests/Sec        |   8.05   |
|       OP tokens/sec        |  334.03  |
|     Avg. latency (ms)      | 6297.46  |
|      P99 latency (ms)      | 12150.95 |
|      P90 latency (ms)      | 10775.46 |
| Avg. IP tokens per request |  799.66  |
| Avg. OP tokens per request |  41.49   |
|   Avg. InFlight requests   |  50.75   |
|     Total latency (ms)     | 12172.66 |
|       Total requests       |  98.00   |
+----------------------------+----------+
Expected op tokens 41.49
```

### Inflight batching
As in the static batching case, we start by adapting the batching policy in the `tensorrt_llm/config.pbtxt` file.
```
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "inflight_fused_batching"
  }
}
...
parameters: {
  key: "batch_scheduler_policy"
  value: {
    string_value: "guaranteed_completion"
  }
}
```
```bash
tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/fp16/inflight/
```

```bash
python3 tools/inflight_batcher_llm/identity_test.py \
    -i grpc \
    -u 192.168.1.3:8001 \
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


<!-- **TODO** test with CNN Daily mail 

**TODO** WIP/ renommer les configs pour pouvoir lancer les 2 modeles en meme temps  -->

## Next Step
[Quantization](06_Quantization.md)
