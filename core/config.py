import os
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings, loaded from environment variables and config file"""
    
    # App Settings
    APP_NAME: str = "Redshift Query AI Agent"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database Settings
    REDSHIFT_DBNAME: str = Field(default="", env="REDSHIFT_DBNAME")
    REDSHIFT_USER: str = Field(default="", env="REDSHIFT_USER")
    REDSHIFT_PASSWORD: str = Field(default="", env="REDSHIFT_PASSWORD")
    REDSHIFT_HOST: str = Field(default="", env="REDSHIFT_HOST")
    REDSHIFT_PORT: str = Field(default="5439", env="REDSHIFT_PORT")
    REDSHIFT_CONNECT_TIMEOUT: int = Field(default=30, env="REDSHIFT_CONNECT_TIMEOUT")
    REDSHIFT_QUERY_TIMEOUT: int = Field(default=300, env="REDSHIFT_QUERY_TIMEOUT")
    REDSHIFT_MAX_ROWS: int = Field(default=100000, env="REDSHIFT_MAX_ROWS")
    
    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: str = Field(default="", env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = Field(default="", env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = Field(default="2023-05-15", env="AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default="", env="AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # LLM Settings
    LLM_TEMPERATURE: float = Field(default=0.0, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=1024, env="LLM_MAX_TOKENS")
    
    # Agent Settings
    AGENT_MEMORY_LIMIT: int = Field(default=10, env="AGENT_MEMORY_LIMIT")
    AGENT_ENFORCE_LIMITS: bool = Field(default=True, env="AGENT_ENFORCE_LIMITS")
    AGENT_LOG_QUERIES: bool = Field(default=True, env="AGENT_LOG_QUERIES")
    
    # UI Settings
    UI_THEME: str = Field(default="light", env="UI_THEME")
    UI_SHOW_SQL: bool = Field(default=False, env="UI_SHOW_SQL")
    UI_SHOW_TIMING: bool = Field(default=False, env="UI_SHOW_TIMING")
    UI_PAGE_TITLE: str = Field(default="Financial Data Query Assistant", env="UI_PAGE_TITLE")
    UI_PAGE_ICON: str = Field(default="💰", env="UI_PAGE_ICON")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def update_from_yaml(self, config_path: Optional[str] = None):
        """Update settings from a YAML configuration file"""
        if not config_path:
            config_path = os.environ.get("CONFIG_FILE", "config.yaml")
        
        path = Path(config_path)
        if path.exists() and path.is_file():
            try:
                with open(path, 'r') as file:
                    yaml_config = yaml.safe_load(file)
                    self._update_from_dict(yaml_config)
            except Exception as e:
                print(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update settings from a dictionary"""
        if not config_dict:
            return
            
        # Flatten nested dictionary
        flat_dict = {}
        
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}".upper()
                if isinstance(v, dict):
                    flatten(v, f"{key}_")
                else:
                    flat_dict[key] = v
        
        flatten(config_dict)
        
        # Update settings
        for key, value in flat_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Create a global settings object
settings = Settings()

# Load from YAML if present
settings.update_from_yaml()