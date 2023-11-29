# Serving with Triton Inference Server
## Introduction
The purpose here is to compare performances of the llama2-7b model serve by Triton when it has been optimized using TensorRT-LLM Vs huggingface python one. 
## Models Repository 
Before launching Triton Inference Server, we need to prepare the models repository beforehand and it should respect the structure below. For further details on the model repository in Triton Inference server please refer to this [documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).
Model repository are used by the Triton Server to detect model locations.
```
<model-repository-path>/
  <model-name>/
    [config.pbtxt]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
```
## TensorRT-LLM Model - Triton Ensemble
### Overview
An ensemble model represents a pipeline of one or more models and the connection of input and output tensors between those models. Read more about [ensembles](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models).

Our pipeline for TensorRT-LLM model serving in Triton comprises 4 components :

- `preprocessing`: tokenize the input text (Python Backend)

- `tensorrtllm`: infer the TRTLLM engine (TensorRT- LLM backend)

- `postprocessing`: decode the text output (Python Backend) 

- `ensemble` folder describing the Inputs/Outputs and the sequence of models to call during inference. 

When querying the TensorRT-LLM model, we will query only the "ensemble" which is responsible for all the pipeline. The folder's structure should be similar to the following :

```
triton_model_repo/llama_7b/fp16/no-inflight/
├── ensemble
│   ├── 1
│   └── config.pbtxt
├── postprocessing
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
├── preprocessing
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── tensorrt_llm
    ├── 1
    └── config.pbtxt
```
### Steps
1. Create the triton model repository that will holds all the triton model 
```
mkdir -p /scratch/triton_model_repo/llama_7b/python
```
2. Initiates the python folder with template files from tensorrtllm_backend
```
cp -R /scratch/tensorrtllm_backend/all_models/inflight_batcher_llm/* /scratch/triton_model_repo/llama_7b/python/.
```
3. Update the preprocessing template values
```
sed -i 's#${tokenizer_dir}#/workspace/meta/llama_models#' /scratch/triton_model_repo/llama_7b/python/preprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#llama#' /scratch/triton_model_repo/llama_7b/python/preprocessing/config.pbtxt
```
4. Update the postprocessing template values
```
sed -i 's#${tokenizer_dir}#/workspace/meta/llama_models#' /scratch/triton_model_repo/llama_7b/python/postprocessing/config.pbtxt
sed -i 's#${tokenizer_type}#llama#' /scratch/triton_model_repo/llama_7b/python/postprocessing/config.pbtxt
```
5. Update the tensorrt_llm template values
```
sed -i 's#${decoupled_mode}#False#' /scratch/triton_model_repo/llama_7b/python/tensorrt_llm/config.pbtxt
sed -i 's#inflight_fused_batching#V1#' /scratch/triton_model_repo/llama_7b/python/tensorrt_llm/config.pbtxt
sed -i 's#${engine_dir}#/workspace/trt-engines/llama_7b/fp16/1-gpu#' /scratch/triton_model_repo/llama_7b/python/tensorrt_llm/config.pbtxt
sed -i 's#${max_tokens_in_paged_kv_cache}##' /scratch/triton_model_repo/llama_7b/python/tensorrt_llm/config.pbtxt
sed -i 's#${batch_scheduler_policy}#guaranteed_completion#' /scratch/triton_model_repo/llama_7b/python/tensorrt_llm/config.pbtxt
```
## Python Model - HF Llama model
### Overview 
We will serve one reference model called *llama_python*, using Hugging Face Text Generation Pipeline and Triton Python Backend.
The Text Generation Pipeline includes the tokenization, the inference on the model and the text decoding process. 

![Astuce Icon](images/common/astuce_icon.png)For simplicity in this particular experiment, the `llama-python` folder resides at the same level as the TensorRT-LLM components described above. Thus, the model repository should be similar to the following snippet. But, this is **not** recommended for production runs. 
```
triton_model_repo/llama_7b/python
├── ensemble
├── llama-python
├── postprocessing
├── preprocessing
└── tensorrt_llm
```

1. Create the **llama_python** hugging face template
   - **Folder creation**
```
mkdir -p /scratch/triton_model_repo/llama_7b/python/llama-python/1
```
   - **Add the python's triton model**
```
cat<<'EOF'>/scratch/triton_model_repo/llama_7b/python/llama-python/1/model.py
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os

os.environ[
    "TRANSFORMERS_CACHE"
] = "/opt/tritonserver/model_repository/llama-2-7b-chat-hf/hf_cache"
import json

import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        default_hf_model = "meta-llama/Llama-2-7b-chat-hf"
        #default_max_gen_length = "15"
        # Check for user-specified model name in model config parameters
        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )
        # NOTE: MOVED AS A REQUIRED INPUT  
        #Check for user-specified max length in model config parameters
        #self.max_output_length = int(
        #    self.model_params.get("max_output_length", {}).get(
        #        "string_value", default_max_gen_length
        #    )
        #)
        #self.logger.log_info(f"Max sequence length: {self.max_output_length}")

        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")
        # Assume tokenizer available for same model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=hf_model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def execute(self, requests):
        prompts = []
        out_lengths = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            max_out_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")      
            prompt = input_tensor.as_numpy()[0].decode()
            out_length = max_out_tensor.as_numpy()[0]
            #self.logger.log_info(f"Generating sequences of max output size: {out_length} for text_input: {prompt} ")
            prompts.append(prompt)
            out_lengths.append(out_length)

        batch_size = len(prompts)
        return self.generate(prompts, out_lengths, batch_size)

    def generate(self, prompts, out_lengths, batch_size):    
        sequences = self.pipeline(
            prompts,
            #max_length= out_lengths[0], #self.max_output_length,
#The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set.
#max_new_tokens (int, optional) — The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            max_new_tokens = out_lengths[0],
            pad_token_id=self.tokenizer.eos_token_id,
            batch_size=batch_size,
        )
        responses = []
        texts = []
        for i, seq in enumerate(sequences):
            output_tensors = []
            text = seq[0]["generated_text"]
            texts.append(text)
            #self.logger.log_info(f"Output result : {text}")
            tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
            output_tensors.append(tensor)
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def finalize(self):
        print("Cleaning up...")
EOF
```
   - **Add the Model configuration**

```
cat<<'EOF'>/scratch/triton_model_repo/llama_7b/python/llama-python/config.pbtxt
# Triton backend to use
backend: "python"

# Hugging face model path. Parameters must follow this
# key/value structure
parameters: {
  key: "huggingface_model",
  value: {string_value: "/workspace/meta/hf-weights/7B-chat"}
}

# Triton should expect as input a single string of set
# length named 'text_input' and a single INT value of set length named "max_output_length"
input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  {
    name: "max_tokens"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  }
]

# Triton should expect to respond with a single string
# output of variable length named 'text_output'
output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    }
  ]

EOF
```

## Inferencing using Triton Inference Server
### Launch the Server
1. We use the docker command below to run the container based on the model repository created above
```
sudo docker run   -d                                    \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        --name triton_server_huggingface                \
        -v /scratch:/workspace                          \
        tritonserver-aipulse:23.10 tritonserver --model-repository=/workspace/triton_model_repo/llama_7b/python
```
![Astuce icon](./images/common/astuce_icon.png)**NB**:
If you built the engine with `--world_size X` where `X` is greater than 1, you will need to use the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) script.

2. You can follow your container log using the command below ,  The server is ready when all the models' status are `READY`. The output should be similiar to this screenshot below : 
```
sudo docker logs triton_server_huggingface -f
```
![triton server ready](./images/triton/tritonserver-ready.PNG)

### Send Requests from a Client
```
sudo docker run   -d                                    \
        --runtime=nvidia                                \
        --gpus all                                      \
        -it --rm                                        \
        --net host --shm-size=2g                        \
        --ulimit memlock=-1 --ulimit stack=67108864     \
        --name triton_client                            \
        -v /scratch:/workspace                          \
        tritonclient-aipulse:23.10 bash
```
```
#!/usr/bin/python

import os
import sys
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import queue
import sys
from datetime import datetime

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import np_to_triton_dtype


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        output = result.as_numpy('text_output')
        print(output[0], flush=True)

def prepare_tensor(name, input, protocol):
        client_util = grpcclient
        t = client_util.InferInput(name, input.shape,
                                   np_to_triton_dtype(input.dtype))
        t.set_data_from_numpy(input)
        return t


def test(triton_client, prompt, max_out, triton_model_name):
    model_name = triton_model_name
    
    if model_name == "ensemble":
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * max_out

        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)
        inputs = [
            prepare_tensor("text_input", input0_data, "grpc"),
            prepare_tensor("max_tokens", output0_len, "grpc"),
            prepare_tensor("bad_words", bad_words_list, "grpc"),
            prepare_tensor("stop_words", stop_words_list,
                                 "grpc")
        ]

    else:
        input0 = [prompt]
        input0_data = np.array(input0).astype(object)
        input1 = [max_out]
        input1_data = np.array(input1).astype(np.uint32)
    
        streaming = [[FLAGS.streaming]]
        streaming_data = np.array(streaming, dtype=bool)
    
        inputs= [prepare_tensor("text_input", input0_data, "grpc"),
                prepare_tensor("max_tokens", input1_data, "grpc"),
                ]
    
    user_data = UserData()

    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    # Send request
    triton_client.async_stream_infer(model_name, inputs)

    #Wait for server to close the stream
    triton_client.stop_stream()

    # Parse the responses
    while True:
        try:
            result = user_data._completed_requests.get(block=False)
        except Exception:
            break

        if type(result) == InferenceServerException:
            print("Received an error from server:")
            print(result)
        else:
            result.as_numpy('text_output')            
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')

    parser.add_argument('-p',
                        '--prompt',
                        type=str,
                        required=True,
                        help='Input prompt.')
    parser.add_argument('-o',
                        '--max_tokens',
                        type=int,
                        required=True,
                        help='Max num token output')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=True,
                        help='Triton Model name.')
    
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
        )

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "192.168.1.3:8001"

    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    test(client, FLAGS.prompt, FLAGS.max_tokens, FLAGS.model_name)

```



# Add cleanup for container server