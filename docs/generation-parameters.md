# Generation parameters

All of the scripts that involve LLM data generation accept a common set of parameters.

- **--model**: Either path to the model file or an API model name
- **--server_type**: `nemo`, `trtllm`, `vllm` or `openai`. This is used on the client side
  to correctly format a request to a particular server. This needs to match model
  checkpoint format if self-hosting the model or has to be `openai` for both Nvidia NIM API
  as well as the OpenAI API.
- **--server_address**:
- **--server_gpus**:
- **--server_nodes**:
- **--server_args**:
-