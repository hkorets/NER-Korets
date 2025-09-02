from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class ConfigBase(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

class OpenAIConfig(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    ENDPOINT: str = "https://api.openai.com"
    API_KEY: SecretStr
    REGION: str = "us-east-1"
    API_VERSION: str = "2024-04-01-preview"

    FIXER: str = "o1"
    BINARY_CLASSIFIER: str = "gpt-4o-mini"
    MULTI_CLASSIFIER: str = "gpt-4o-mini"
    EXTRACTOR: str = "o1"

    TEMPERATURE: float = 0.2
    
class Config(ConfigBase):
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)

    @classmethod
    def load_config(cls) -> "Config":
        return cls()