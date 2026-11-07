"""
MaintiDoc - Ollama LLM Client
Provides connectivity check, model verification, automated pull, and lightweight inference.
"""

from typing import List, Dict, Any, Optional
import requests
import json
import time

from config import settings, setup_logger

logger = setup_logger("llm.ollama_client")


class OllamaClient:
    """Interface to communicate with local Ollama service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def is_alive(self) -> bool:
        """Checks if the Ollama server is running and responding."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """Returns the list of locally available model names."""
        if not self.is_alive():
            logger.warning("Ollama server is not accessible.")
            return []

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", []) if "name" in m]
                return models
            return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def is_model_present(self, target_model: Optional[str] = None) -> bool:
        """Verifies if a specific model (or default) is installed locally."""
        model = target_model or self.model_name
        available = self.list_models()
        # Handle exact match or tag match (e.g. qwen2.5:1.5b vs qwen2.5:1.5b-instruct)
        return any(model in m or m.startswith(model.split(":")[0]) for m in available)

    def pull_model(self, target_model: Optional[str] = None) -> bool:
        """Pulls a model from Ollama library with streaming progress logs."""
        model = target_model or self.model_name
        logger.info(f"Initiating Ollama pull for model: '{model}' (this may take a few minutes)...")
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": model, "stream": False}
            response = requests.post(url, json=payload, timeout=self.timeout * 5)
            if response.status_code == 200:
                logger.info(f"Successfully pulled model '{model}'.")
                return True
            else:
                logger.error(f"Failed to pull model '{model}': HTTP {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Exception during Ollama model pull: {e}")
            return False

    def ensure_model(self) -> bool:
        """Verifies model existence and pulls it if missing."""
        if not self.is_alive():
            logger.error(f"Cannot ensure model: Ollama server is unreachable at {self.base_url}.")
            return False

        if self.is_model_present():
            logger.info(f"Model '{self.model_name}' is available locally.")
            return True

        logger.warning(f"Model '{self.model_name}' not found locally. Attempting automatic download...")
        return self.pull_model()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generates a non-streaming completion from the Ollama model.
        """
        temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                err_msg = f"Ollama generation failed (HTTP {response.status_code}): {response.text}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)
        except Exception as e:
            logger.error(f"Error calling Ollama generate endpoint: {e}")
            raise


# Default singleton instance
ollama_client = OllamaClient()
