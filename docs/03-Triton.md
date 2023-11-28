Tout se fait dans le container server
Copy the model rtiton server
rm -rf /opt/tritonserver/inflight_batcher_llm
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.


# preprocessing
sed -i 's#${tokenizer_dir}#/workspace/meta/hf-weights/7B-chat/#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#auto#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
sed -i 's#${tokenizer_dir}#/workspace/meta/hf-weights/7B-chat//#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#auto#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt
sed -i 's#${decoupled_mode}#false#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
sed -i 's#${engine_dir}#/engines/1-gpu/#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt


tritonserver --model-repository=/opt/tritonserver/inflight_batcher_llm/

Documentation
https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#create-the-model-repository
https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#create-the-model-repository



tokenizer_dir=/workspace/trt-engines/llama_7b/fp16/1-gpu


mkdir triton_model_repo
Batch generation
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.

cp -R  /opt/tritonserver/postprocessing  /opt/tritonserver/llama-python
sed -i 's#${tokenizer_dir}#/workspace/meta/llama_models#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#llama#' /opt/tritonserver/inflight_batcher_llm/preprocessing/config.pbtxt
sed -i 's#${tokenizer_dir}#/workspace/meta/llama_models#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#llama#' /opt/tritonserver/inflight_batcher_llm/postprocessing/config.pbtxt
sed -i 's#${decoupled_mode}#False#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt

#Why infloight fuse -> v1
sed -i 's#inflight_fused_batching#V1#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
sed -i 's#${engine_dir}#/workspace/trt-engines/llama_7b/fp16/1-gpu#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
sed -i 's#${max_tokens_in_paged_kv_cache}##' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt
sed -i 's#${batch_scheduler_policy}#guaranteed_completion#' /opt/tritonserver/inflight_batcher_llm/tensorrt_llm/config.pbtxt


Model.py généré manuellement peut etre demandé à Dora et Asama le pourquoi et de la doc

# TensorRT-LLM to Compile and Run Models's Engines
## Compile Engines for Llama 2 models
For the compile phase , we will use the 