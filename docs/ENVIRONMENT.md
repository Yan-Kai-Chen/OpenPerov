# Runtime environments

The default package, public score reproduction, HTTP client and offline tests use the Python standard library. They do not load model weights or require GPU packages.

```sh
python -m pip install --no-deps .
python -m unittest discover -s tests -v
python scripts/evaluate.py reproduce --root .
```

Local model inference requires PyTorch, Transformers, Accelerate and safetensors. Training, merging and learned ranking additionally require PEFT. Install the appropriate CUDA-enabled PyTorch build for the target machine before installing the optional package extras:

```sh
python -m pip install '.[inference]'
python -m pip install '.[training,ranking]'
```

The model implementation must expose `Qwen3_5ForConditionalGeneration`. The original project-server model configuration records Transformers 5.9.0; the package declares the corresponding major-version range. This is a dependency specification, not a claim that every version combination has been GPU-tested. Pin the complete environment after validating it on the deployment hardware. The official [Qwen3.6 model documentation](https://huggingface.co/Qwen/Qwen3.6-27B) describes current serving support.

The paper's inference profile is distinct from upstream default sampling: use the supplied OpenPerov configuration and preserve the non-thinking prompt behavior, generation limits and repetition penalty. Hardware-dependent fast kernels and training throughput have not been tested by the local source-only preparation pass. An HTTP endpoint can instead be supplied explicitly to the inference client; no endpoint is configured by default.
