#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import json
import sys
from datetime import datetime
from functools import partial

import numpy as np
from utils import utils


def callback(user_data, start_time, result, error):
    user_data._completed_requests.put((result, error))
    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    user_data._latencies.append(latency)
 
    #output = result.as_numpy('text_output')
    #print(output[0], flush=True)

#TODO: eventually check output similarity 
def test_functionality(client, prompts, output_lens):
    print(f"[INFO] Start testing on {len(prompts)} prompts.")
    for i, prompt in enumerate(prompts):

        # 1. Using Python backend 

        # 2. Use TRTLLM ensemble model
        model_name = "ensemble"
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)        

        inputs = [
            utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
            utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
            utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("stop_words", stop_words_list,
                                 FLAGS.protocol),
        ]

        result = client.infer(model_name, inputs, request_id=str(i))

        # 3. Check the results between python models and the ensemble trt llm model
        ensemble_output = result.as_numpy('text_output')
        assert output0 == ensemble_output
        if FLAGS.verbose:
            print('Response: {}'.format(result.get_response()))
            print('Output: {}'.format(ensemble_output))
    print(f"[INFO] Functionality test succeed.")


def test_performance(client, prompts, output_lens, model_name):
    input_model_name = model_name

    print(f"[INFO] Warm up for benchmarking.")
    for i in range(10):
        if input_model_name == "ensemble":
            input0 = [[prompts[0]]]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.uint32) * output_lens[0]

            bad_words_list = np.array([[""]], dtype=object)
            stop_words_list = np.array([[""]], dtype=object)
            inputs = [
                utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
                utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
                utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
                utils.prepare_tensor("stop_words", stop_words_list,
                                     FLAGS.protocol)
            ]
        else: 
            input0 = [prompts[0]]
            input0_data = np.array(input0).astype(object)
            output0_len = [output_lens[0]]
            output0_len = np.array(output0_len).astype(np.uint32)
            inputs = [
                utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
                utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol)
            ]
        client.infer(model_name, inputs, request_id=str(i))

    print(f"[INFO] Start benchmarking on {len(prompts)} prompts.")
    latency = 0
    async_requests = []
    start_time = datetime.now()
    user_data = utils.UserData()
    for i, prompt in enumerate(prompts):
        if input_model_name == "ensemble":
            input0 = [[prompt]]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.uint32) * output_lens[i]
        
            bad_words_list = np.array([[""]], dtype=object)
            stop_words_list = np.array([[""]], dtype=object)

            inputs = [
                utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
                utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
                utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
                utils.prepare_tensor("stop_words", stop_words_list,
                                     FLAGS.protocol)
            ]
        else:
          input0 = [prompt]
          input0_data = np.array(input0).astype(object)
          output0_len = [output_lens[i]]
          output0_len = np.array(output0_len).astype(np.uint32)
          inputs = [
                        utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
                        utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
                    ]

        if FLAGS.protocol == "http":
            async_requests.append(
                client.async_infer(model_name, inputs, request_id=str(i)))
        elif FLAGS.protocol == "grpc":
            async_requests.append(
                client.async_infer(model_name,
                                   inputs,
                                   callback=partial(callback, user_data,
                                                    datetime.now()),
                                   request_id=str(i)))

    if FLAGS.protocol == "http":
        utils.get_http_results(async_requests)
    elif FLAGS.protocol == "grpc":
        utils.get_grpc_results(user_data, len(prompts))
    else:
        raise RuntimeError("Invalid protocol")

    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    print(f"[INFO] Total Latency: {latency} ms")


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
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        choices=['http', 'grpc'],
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=True,
                        help='Triton Model name.')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=128,
                        required=False,
                        help='Specify concurrency')
    parser.add_argument('--max_input_len',
                        type=int,
                        required=True,
                        help='Specify max input length')

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    try:
        client = utils.create_inference_server_client(
            FLAGS.protocol,
            FLAGS.url,
            concurrency=FLAGS.concurrency,
            verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    prompts = []
    output_lens = []
    with open(FLAGS.dataset, 'r') as f:
        data_dict = json.load(f)
        for req in data_dict:
            prompt = req['input'] + ' ' + req['instruction']
            output = req['output']
            # 1.3 is a magic number that converts number of words to number of tokens
            if int(len(prompt.split(' ')) / 1.3) > FLAGS.max_input_len:
                continue
            prompts.append(prompt)
            # 1.3 is a magic number that converts number of words to number of tokens
            output_lens.append(int(len(output.split(' ')) * 1.3))
    test_performance(client, prompts, output_lens, FLAGS.model_name)
    
# OPTIONNAL: 
#  test_functionality(client, prompts, output_lens)
  