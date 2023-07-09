---
pipeline_tag: text-generation
inference: true
widget:
- text: 'def print_hello_world():'
  example_title: Hello world
  group: Python
license: bigcode-openrail-m
datasets:
- bigcode/the-stack-dedup
metrics:
- code_eval
library_name: transformers
tags:
- code
model-index:
- name: Tiny-StarCoder-Py
  results:
  - task:
      type: text-generation
    dataset:
      type: openai_humaneval
      name: HumanEval
    metrics:
    - name: pass@1
      type: pass@1
      value: 7.84%
      verified: false
---

# TinyStarCoderPy

This is a 164M parameters model with the same architecture as [StarCoder](https://huggingface.co/bigcode/starcoder) (8k context length, MQA & FIM). It was trained on the Python data from [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata)
for ~6 epochs which amounts to 100B tokens.


## Use

### Intended use

The model was trained on GitHub code, to assist with some tasks like [Assisted Generation](https://huggingface.co/blog/assisted-generation). For pure code completion, we advise using our 15B models [StarCoder]() or [StarCoderBase]().


### Generation
```python
# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/tiny_starcoder_py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### Fill-in-the-middle
Fill-in-the-middle uses special tokens to identify the prefix/middle/suffix part of the input and output:

```python
input_text = "<fim_prefix>def print_one_two_three():\n    print('one')\n    <fim_suffix>\n    print('three')<fim_middle>"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

# Training

## Model

- **Architecture:** GPT-2 model with multi-query attention and Fill-in-the-Middle objective
- **Pretraining steps:** 50k
- **Pretraining tokens:** 100 billion
- **Precision:** bfloat16

## Hardware

- **GPUs:** 32 Tesla A100
- **Training time:** 18 hours

## Software

- **Orchestration:** [Megatron-LM](https://github.com/bigcode-project/Megatron-LM)
- **Neural networks:** [PyTorch](https://github.com/pytorch/pytorch)
- **BP16 if applicable:** [apex](https://github.com/NVIDIA/apex)

# License
The model is licensed under the BigCode OpenRAIL-M v1 license agreement. You can find the full agreement [here](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).
