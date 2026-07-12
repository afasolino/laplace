# NPU status

`outputs/system_probe.json` records the exact AMD/NPU and ONNX Runtime availability. The device query returned a WMI `0x80041003` permission error and ONNX Runtime remains absent, so this host is recorded as `NPU_OPTIONAL_NOT_BENEFICIAL`. The measured embedding baseline uses Ollama on the NVIDIA/CPU runtime; no claim is made that the NPU accelerates the NVIDIA-hosted LLM.
