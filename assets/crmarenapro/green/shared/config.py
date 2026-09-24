"""
Centralized Configuration Management

Enterprise-ready configuration using Pydantic Settings with environment variable support.
All configuration is loaded from environment variables or .env file.

Usage:
    from shared.config import settings
    
    # Access LLM settings
    model = settings.llm.model
    api_key = settings.llm.api_key
    
    # Access assessment defaults
    timeout = settings.assessment.timeout
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Determine project root for .env file loading
PROJECT_ROOT = Path(__file__).parent.parent


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    NEBIUS = "nebius"
    CUSTOM = "custom"


class LLMConfig(BaseSettings):
    """
    LLM Configuration.
    
    Supports multiple providers with automatic API key detection.
    Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY > NEBIUS_API_KEY
    
    API keys are loaded from .env file (secrets).
    Other settings are hardcoded defaults (non-secrets).
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys from .env (secrets only)
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key"
    )
    nebius_api_key: Optional[str] = Field(
        default=None, 
        validation_alias="NEBIUS_API_KEY",
        description="Nebius AI Studio API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
        description="Anthropic API key"
    )
    
    # =========================================================================
    # Non-secret defaults (hardcoded, not from env)
    # =========================================================================
    
    # Model configuration
    model: str = "claude-3-5-sonnet-latest"
    base_url: Optional[str] = None  # Auto-detected based on provider
    
    # Provider selection (auto-detected based on API key)
    provider: LLMProvider = LLMProvider.ANTHROPIC
    
    # Request settings
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 60  # seconds
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the active API key based on priority."""
        return self.anthropic_api_key or self.openai_api_key or self.nebius_api_key
    
    @property
    def effective_base_url(self) -> str:
        """Get the appropriate base URL for the provider."""
        if self.base_url:
            return self.base_url
        
        # Auto-detect based on provider/model
        if self.anthropic_api_key:
            return "https://api.anthropic.com/v1/"
        elif self.openai_api_key:
            return "https://api.openai.com/v1/"
        elif self.nebius_api_key:
            return "https://api.studio.nebius.ai/v1/"
        else:
            return "https://api.anthropic.com/v1/"
    
    @field_validator("model", mode="before")
    @classmethod
    def set_model_from_env(cls, v):
        """Allow model override from LLM_MODEL env var (for advanced users)."""
        return os.getenv("LLM_MODEL") or v
    
    def model_post_init(self, __context):
        """Auto-detect provider based on API key and model."""
        # Check env vars directly as fallback
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.nebius_api_key:
            self.nebius_api_key = os.getenv("NEBIUS_API_KEY")
        
        # Auto-detect provider
        if self.anthropic_api_key and (
            self.model.startswith("claude") or self.model.startswith("anthropic/")
        ):
            self.provider = LLMProvider.ANTHROPIC
        elif self.openai_api_key and self.model.startswith("gpt"):
            self.provider = LLMProvider.OPENAI
        elif self.anthropic_api_key:
            self.provider = LLMProvider.ANTHROPIC
        elif self.openai_api_key:
            self.provider = LLMProvider.OPENAI
        elif self.nebius_api_key:
            self.provider = LLMProvider.NEBIUS
        elif self.base_url:
            self.provider = LLMProvider.CUSTOM


class AssessmentDefaults(BaseSettings):
    """
    Default values for assessment configuration.
    
    All values are hardcoded defaults (non-secrets).
    Can be overridden via request config at runtime.
    """
    model_config = SettingsConfigDict(extra="ignore")
    
    # =========================================================================
    # Non-secret defaults (hardcoded)
    # =========================================================================
    
    # Task selection defaults (configurable from leaderboard)
    task_percentage: float = 5.0
    task_limit: Optional[int] = None
    
    # =========================================================================
    # HARDCODED ADVERSARIAL PARAMETERS
    # These are now HARDCODED in agent.py and cannot be overridden from config.
    # Listed here for documentation purposes only.
    # =========================================================================
    # drift_level: "medium"  - Schema Drift (tests adaptation to renamed columns)
    # rot_level: "medium"    - Context Rot (tests filtering of distractor records)
    # max_steps: 10          - Maximum agent turns per task
    # timeout: 300           - Timeout per task in seconds
    # org_type: "b2b"        - Business-to-Business scenarios (CRMArenaPro B2B split)
    
    # Legacy fields (kept for backward compatibility, but ignored by agent)
    drift_level: str = "medium"
    rot_level: str = "medium"
    max_steps: int = 10
    timeout: int = 300  # seconds
    org_type: str = "b2b"
    
    # Mode defaults
    skip_original: bool = False  # Run both original and entropic by default


class ServerConfig(BaseSettings):
    """
    Server configuration.
    
    All values are hardcoded defaults (non-secrets).
    Can be overridden via CLI arguments.
    """
    model_config = SettingsConfigDict(extra="ignore")
    
    # =========================================================================
    # Non-secret defaults (hardcoded)
    # =========================================================================
    host: str = "0.0.0.0"
    port: int = 9009
    log_level: str = "INFO"


class Settings(BaseSettings):
    """
    Root settings class combining all configuration sections.
    
    Usage:
        from shared.config import settings
        
        print(settings.llm.model)
        print(settings.llm.api_key)
        print(settings.assessment.timeout)
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    assessment: AssessmentDefaults = Field(default_factory=AssessmentDefaults)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # Application metadata
    app_name: str = Field(default="Entropic CRMArena")
    version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)
    
    def validate_llm_config(self) -> bool:
        """Validate that LLM configuration is usable."""
        if not self.llm.api_key:
            logger.warning(
                "No LLM API key configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or NEBIUS_API_KEY. "
                "LLM-based answer extraction will be disabled."
            )
            return False
        return True
    
    def log_config_summary(self):
        """Log configuration summary (safe - no secrets)."""
        api_key_status = "configured" if self.llm.api_key else "NOT SET"
        logger.info(f"Configuration loaded:")
        logger.info(f"  LLM Provider: {self.llm.provider.value}")
        logger.info(f"  LLM Model: {self.llm.model}")
        logger.info(f"  LLM API Key: {api_key_status}")
        logger.info(f"  LLM Base URL: {self.llm.effective_base_url}")
        logger.info(f"  Assessment Timeout: {self.assessment.timeout}s")
        logger.info(f"  Skip Original Mode: {self.assessment.skip_original}")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    # Load .env file if it exists
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded .env from {env_path}")
    
    return Settings()


# Global settings instance
settings = get_settings()
