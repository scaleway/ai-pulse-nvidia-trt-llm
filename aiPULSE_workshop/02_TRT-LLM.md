# TensorRT-LLM to Compile and Run Models's Engines

## Llama 2 Example
In the Triton Inference Server container, make sure to navigate to the example folder where you have all the models supported by TensorRT-LLM. For each of these models, you need to install the dependencies to run the scripts.

```bash
cd /workspace/tensorrtllm_backend/tensorrt_llm/examples/llama
pip install -r requirements.txt
```

## Compile Engines for Llama 2 models
We use the same variables as in the previous step. Make sure to fill them with your values.

Run the `build.py` script to compile the TRT-LLM engines.

```bash
cd /workspace/tensorrtllm_backend/tensorrt_llm/examples/llama

python build.py --model_dir /workspace/meta/hf-weights/7B-chat  \
                --dtype float16 \
                --use_gpt_attention_plugin float16  \
                --paged_kv_cache \
                --remove_input_padding \
                --use_gemm_plugin float16  \
                --output_dir "/workspace/trt-engines/llama_7b/fp16/1-gpu"  \
                --max_input_len 2048 --max_output_len 512 \
                --use_rmsnorm_plugin float16  \
                --enable_context_fmha \
                --use_inflight_batching
```

You can test the output of the model with `run.py` located in the same llama examples folder.

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

## Next Step
[Serving with Triton Inference Server](03_Triton.md)