from typing import List
from openai.types.chat import ChatCompletionMessageParam

import os
import json
from time import sleep
from abc import ABC, abstractmethod
from openai import AzureOpenAI, OpenAI
from threading import Lock
from dotenv import load_dotenv

load_dotenv(".env", override=True)
# Used for Hugging Face API
from huggingface_hub import InferenceClient

""" 
This module contains the factory class `LLMServiceFactory` which manages instances of LLM services. This allows
to add support for multiple LLM providers in the future (if needed) and allows to reuse instances of LLM services
and makes adding new LLM services straightforward. Each LLM service is represented by a class that inherits from
the abstract base class `LLMService`. Currently, only Azure OpenAI and Hugging Face are supported.
NOTE: Only instantiate LLM services using the factory to ensure singleton behavior! Never instantiate them directly.

To use the factory, call the `get_service` method with the model name, provider, temperature, and max_tokens as arguments.
Note that one has to specify the necessary API keys and other configurations in the `.env` file. Furthermore, one has to 
specify the default model one wants to use in the `.env` file.

The `tokens_usage.json` file is used to track the number of tokens consumed by each model for monitoring purposes. This file
stores the prompt tokens and completion tokens separately for each model, and it calculates the overall cost based on predefined 
cost rates. Note that this file is included in `.gitignore` to ensure it is not staged or committed to version control. This means 
the token usage data persists across different versions of the codebase, even when newly pulled, allowing the cost tracking to 
remain accurate and up-to-date.

To add a new provider (e.g., Google Gemini):
1. Create a class for the service which implements the LLMService interface.
2. Implement the `initialize_client` and `make_request` methods in this new class.
3. Register the new service in the LLMServiceFactory class.
4. Insert the necessary API tokens and other configurations in the `.env` file.
"""


class LLMService(ABC):
    """Abstract base class to define the interface for LLM services."""

    def __init__(self, model_name, temperature, max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self.initialize_client()

    @abstractmethod
    def initialize_client(self):
        """Initializes the client specific to the LLM provider."""
        raise NotImplementedError(
            "Method `initialize_client` must be implemented in the derived class."
        )

    @abstractmethod
    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Makes a request to the LLM service and returns the response."""
        raise NotImplementedError(
            "Method `make_request` must be implemented in the derived class."
        )


class OpenAIService(LLMService):
    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)

    def initialize_client(self):
        print(os.getenv("OPENAI_API_KEY"))
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # TODO: Add token usage tracking
        return response.choices[0].message.content


class AzureOpenAIService(LLMService):
    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)

    def initialize_client(self) -> AzureOpenAI:
        """Initialize the Azure OpenAI client with API credentials."""
        if self.model_name == "gpt-4o":
            api_key = os.getenv("AZURE_OPENAI_KEY_GPT_4o")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT_4o")
        else:
            api_key = os.getenv("AZURE_OPENAI_KEY_GPT_35")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT_35")
        if not api_key or not endpoint:
            raise EnvironmentError(
                "Azure OpenAI API key or endpoint not found in environment variables."
            )

        # Use api version `2023-05-15` even for `gpt-4o` (else it wont work), but i don't know why
        api_version = "2023-05-15"

        return AzureOpenAI(
            api_version=api_version, api_key=api_key, azure_endpoint=endpoint
        )

    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Make a request to the Azure OpenAI API and handle the response."""
        num_attempts = 1  # Increase this number if you want to retry failed requests
        for i in range(num_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    user="backend",
                )
                response_message = response.choices[0].message.content

                # Track the number of tokens used for each request, for monitoring purposes
                LLMServiceFactory.update_tokens_usage(
                    self.model_name,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

                return response_message
            except Exception as e:
                if i == num_attempts - 1:
                    # If the last attempt failed, raise an error
                    raise RuntimeError(f"API request failed: {e}")
                sleep(0.5)  # Wait for 0.5 seconds before retrying
        raise RuntimeError(f"API request failed. Tried {num_attempts} times.")


class HuggingFaceService(LLMService):
    """
    Inference models from the free Hugging Face serverless API. Doc: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client

    The models `mistralai/Mistral-7B-Instruct-v0.2`, `HuggingFaceH4/zephyr-7b-beta`, `meta-llama/Meta-Llama-3-8B-Instruct`,
    `codellama/CodeLlama-34b-Instruct-hf`, `microsoft/Phi-3-mini-4k-instruct` and `mistralai/Mixtral-8x7B-Instruct-v0.1` perform
    well for instructive text generation.
    """

    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)

    def initialize_client(self):
        return InferenceClient(
            self.model_name, token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

    def make_request(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Make a request to the Hugging Face API and handle the response."""
        self.client: InferenceClient
        num_attempts = 1  # Increase this number if you want to retry failed requests
        for i in range(num_attempts):
            try:
                response = self.client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                LLMServiceFactory.update_tokens_usage(
                    self.model_name,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if i == num_attempts - 1:
                    # If the last attempt failed, raise an error
                    raise RuntimeError(f"API request failed: {e}")
                sleep(0.2)  # Wait for 0.2 seconds before retrying
        raise RuntimeError(f"API request failed. Tried {num_attempts} times.")

    def _get_available_models(self):
        """Get a list of all available models from the Hugging Face API.
        See here: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.list_deployed_models
        """
        client = InferenceClient(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        models = client.list_deployed_models()
        return models[
            "text-generation"
        ]  # We are only interested in text generation models


class LLMServiceFactory:
    """
    Factory class to manage LLM instances.
    This factory ensures that a single instance of an LLM service is created per unique set of parameters (model, provider, temperature, max_tokens).
    """

    _services = {}
    _tokens_usage_file = "llm_service/tokens_usage.json"  # File to persistently store the accumulated tokens usage
    _lock = Lock()  # ensure thread safety

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-35-turbo",
        "gpt-35-turbo-16k",
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "codellama/CodeLlama-34b-Instruct-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]

    @classmethod
    def get_service(
        cls,
        model_name: str,
        provider: str,
        temperature: float = 0,
        max_tokens: int = 512,
    ) -> LLMService:
        key = (model_name, provider, temperature, max_tokens)
        if key not in cls._services:
            # Check if the model is supported
            if model_name not in cls.SUPPORTED_MODELS:
                raise ValueError(
                    f"Model {model_name} not found in the list of supported models: {cls.SUPPORTED_MODELS}."
                )
            if provider == "AzureOpenAI":
                cls._services[key] = AzureOpenAIService(
                    model_name, temperature, max_tokens
                )
            elif provider == "HuggingFace":
                cls._services[key] = HuggingFaceService(
                    model_name, temperature, max_tokens
                )
            elif provider == "OpenAI":
                cls._services[key] = OpenAIService(model_name, temperature, max_tokens)
            else:
                # Add support for other providers here, if needed
                raise ValueError(
                    f"Unsupported provider: {provider}. Supported providers: AzureOpenAI, HuggingFace"
                )
        return cls._services[key]

    @classmethod
    def update_tokens_usage(cls, model_name, prompt_tokens, completion_tokens):
        """Update tokens usage data and save to the JSON file."""
        # Define cost rates, from here: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
        COST_RATES = {
            "gpt-4o": {
                "prompt_token_cost": 0.0045 / 1000,
                "completion_token_cost": 0.0135 / 1000,
            },
            "gpt-35-turbo": {
                "prompt_token_cost": 0.0018 / 1000,
                "completion_token_cost": 0.0018 / 1000,
            },
            "gpt-35-turbo-16k": {
                "prompt_token_cost": 0.0027 / 1000,
                "completion_token_cost": 0.0036 / 1000,
            },
        }

        cls._lock.acquire()
        try:
            # Load existing data
            if os.path.exists(cls._tokens_usage_file):
                with open(cls._tokens_usage_file, "r") as file:
                    tokens_usage = json.load(file)
            else:
                tokens_usage = {}

            # Update the data
            tokens_usage.setdefault(
                model_name, {"prompt_tokens": 0, "completion_tokens": 0}
            )
            tokens_usage[model_name]["prompt_tokens"] += prompt_tokens
            tokens_usage[model_name]["completion_tokens"] += completion_tokens

            # Update overall cost
            if not COST_RATES.get(model_name):
                # We assume no cost for the model if the cost rates are not defined
                COST_RATES[model_name] = {
                    "prompt_token_cost": 0,
                    "completion_token_cost": 0,
                }
            prompt_cost = prompt_tokens * COST_RATES[model_name]["prompt_token_cost"]
            completion_cost = (
                completion_tokens * COST_RATES[model_name]["completion_token_cost"]
            )
            tokens_usage["overall_cost"] = (
                tokens_usage.get("overall_cost", 0) + prompt_cost + completion_cost
            )

            # Save the updated data
            with open(cls._tokens_usage_file, "w") as file:
                json.dump(tokens_usage, file, indent=4)
        finally:
            cls._lock.release()
