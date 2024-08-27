# LLMServiceFactory README

The `LLMServiceFactory` module manages instances of LLM (Large Language Model) services, allowing support for multiple LLM providers and reuse of instances. This module currently supports Azure OpenAI and Hugging Face.

## Usage

Ensure API keys are specified in the `.env` file. See the `example.env` file for an example.

Here is an example of how to use the `LLMServiceFactory` module:

```python
from llm_service.llm_factory import LLMServiceFactory

# Initialize the service
gpt_35_service = LLMServiceFactory.get_service(provider="AzureOpenAI", model_name="gpt-35-turbo-16k", temperature=0.5, max_tokens=100)
hf_service = LLMServiceFactory.get_service(provider="HuggingFace", model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")

# Make a request to the service
res = gpt_35_service.make_request([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])

# Print the response
print(res)
```

The Hugging Face serverless API is free to use. Models that perform well are:

- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `codellama/CodeLlama-34b-Instruct-hf`
- `HuggingFaceH4/zephyr-7b-beta`

## Adding New Providers

- Implement the `LLMService` interface.
- Example for a hypothetical `GoogleGeminiService`:

```python
class GoogleGeminiService(LLMService):
    def initialize_client(self):
        # Initialize the Google Gemini client
        pass

    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        # Implement the request handling
        pass
```

- Add the new provider to the `LLMServiceFactory`:

```python
elif provider == "GoogleGemini":
     cls._services[key] = GoogleGeminiService(model_name, temperature, max_tokens)
```

- Add the model to the `SUPPORTED_MODELS` list in the `llm_factory.py` file.
- Update `.env`: Add necessary API tokens.

## Token Usage Tracking

The `tokens_usage.json` file tracks the number of tokens consumed for monitoring purposes. Ensure this file is included in `.gitignore` to avoid version control issues.
