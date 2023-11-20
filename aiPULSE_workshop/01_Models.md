# Model's Weights Download and Preparation

## Llama 2 Models

### Download and convert the model's weights
Make sure to download the model weights from [Meta](https://ai.meta.com/llama/).

```bash
cd /workspace; mkdir meta
git clone https://github.com/facebookresearch/llama.git

mkdir llama_models; cd llama_models
../llama/download.sh
```

Your folder's content should look similar to the following:
```
-rw-rw-r-- 1 ubuntu ubuntu   7020 Jul 15 00:06 LICENSE
-rw-rw-r-- 1 ubuntu ubuntu   4766 Jul 15 00:06 USE_POLICY.md
drwxrwxr-x 2 ubuntu ubuntu   4096 Nov  7 08:43 llama-2-13b
drwxrwxr-x 2 ubuntu ubuntu   4096 Nov  7 09:38 llama-2-13b-chat
drwxrwxr-x 2 ubuntu ubuntu   4096 Nov  7 09:14 llama-2-70b
drwxrwxr-x 2 ubuntu ubuntu   4096 Nov  7 10:08 llama-2-70b-chat
drwxrwxr-x 2 ubuntu ubuntu   4096 Nov  7 08:38 llama-2-7b
-rw-rw-r-- 1 ubuntu ubuntu 499723 Jul 13 22:27 tokenizer.model
-rw-rw-r-- 1 ubuntu ubuntu     50 Jul 13 22:27 tokenizer_checklist.chk
```

### Convert the model's weights to HF format
For some of our commands, we will need to convert the weights from Meta Checkpoint into Hugging Face Transformers format as explained [here](https://huggingface.co/docs/transformers/main/en/model_doc/llama). Inside the tritonserver container execute the following commands:

```bash
git clone https://github.com/huggingface/transformers.git

cd transformers
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /workspace/meta/llama_models/llama-2-7b-chat \
        --model_size 7B \
        --output_dir /workspace/meta/hf-weights/7B-chat
```

## Next step
[TensorRT-LLM engines for a single GPU run](02_TRT-LLM.md) 
