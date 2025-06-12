"""
LLM Manager for handling language model interactions.
Supports multiple providers including OpenAI, Anthropic, and local models.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str  # "openai", "anthropic", "local"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.3


class LLMManager:
    """
    Manages interactions with various Language Model providers.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM Manager.

        Args:
            config: LLM configuration. If None, will try to auto-detect from environment.
        """
        self.config = config or self._auto_detect_config()
        self.client = None
        self._initialize_client()

    def _auto_detect_config(self) -> LLMConfig:
        """Auto-detect LLM configuration from environment variables."""

        # Check for OpenAI
        if os.getenv("OPENAI_API_KEY"):
            return LLMConfig(
                provider="openai",
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

        # Check for Anthropic
        elif os.getenv("ANTHROPIC_API_KEY"):
            return LLMConfig(
                provider="anthropic",
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

        # Default to OpenAI (will require API key to be set)
        else:
            logger.warning("No API keys found in environment. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key=None)

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        try:
            if self.config.provider == "openai":
                self.client = openai.OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
            elif self.config.provider == "anthropic":
                self.client = Anthropic(api_key=self.config.api_key)
            else:
                logger.error(f"Unsupported provider: {self.config.provider}")

        except Exception as e:
            logger.error(f"Error initializing {self.config.provider} client: {e}")
            self.client = None

    def generate_content(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate content using the configured LLM.

        Args:
            prompt: The main prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt (for models that support it)

        Returns:
            Generated content
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Please check your configuration.")

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        try:
            if self.config.provider == "openai":
                return self._generate_openai(prompt, max_tokens, temperature, system_prompt)
            elif self.config.provider == "anthropic":
                return self._generate_anthropic(prompt, max_tokens, temperature, system_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float, system_prompt: Optional[str]) -> str:
        """Generate content using OpenAI API."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False
        )

        return response.choices[0].message.content

    def _generate_anthropic(
        self, prompt: str, max_tokens: int, temperature: float, system_prompt: Optional[str]
    ) -> str:
        """Generate content using Anthropic API."""

        # Anthropic handles system prompts differently
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\n\nAssistant:"

        response = self.client.completions.create(
            model=self.config.model, prompt=full_prompt, max_tokens_to_sample=max_tokens, temperature=temperature
        )

        return response.completion

    def generate_structured_content(
        self, prompt: str, schema: Dict[str, Any], max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate structured content (JSON) using the LLM.

        Args:
            prompt: The main prompt
            schema: JSON schema describing the expected output structure
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Parsed JSON response
        """
        structured_prompt = f"""
{prompt}

Please respond with valid JSON that follows this schema:
{schema}

Ensure your response is valid JSON with no additional text or explanations.
"""

        response = self.generate_content(structured_prompt, max_tokens=max_tokens, temperature=temperature)

        try:
            import json

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise

    def generate_multiple_variants(
        self, prompt: str, num_variants: int = 3, max_tokens: Optional[int] = None, temperature: float = 0.7
    ) -> List[str]:
        """
        Generate multiple variants of content for the same prompt.

        Args:
            prompt: The main prompt
            num_variants: Number of variants to generate
            max_tokens: Maximum tokens per variant
            temperature: Sampling temperature (higher for more variation)

        Returns:
            List of generated variants
        """
        variants = []

        for i in range(num_variants):
            try:
                variant = self.generate_content(prompt, max_tokens=max_tokens, temperature=temperature)
                variants.append(variant)
            except Exception as e:
                logger.error(f"Error generating variant {i+1}: {e}")
                continue

        return variants

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4

    def check_token_limit(self, prompt: str, max_tokens: int) -> bool:
        """
        Check if prompt + max_tokens would exceed model limits.

        Args:
            prompt: The prompt text
            max_tokens: Requested max tokens for generation

        Returns:
            True if within limits, False otherwise
        """
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
        }

        model_limit = model_limits.get(self.config.model, 4096)
        prompt_tokens = self.estimate_tokens(prompt)

        return (prompt_tokens + max_tokens) <= model_limit

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "client_initialized": self.client is not None,
        }
