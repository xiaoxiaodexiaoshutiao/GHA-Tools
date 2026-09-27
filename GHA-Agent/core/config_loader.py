import os
import yaml
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path


class ConfigLoader:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_env()
            self._model_config = None
            self._prompts_config = None
            self._base_dir = Path(__file__).parent.parent
            ConfigLoader._initialized = True

    def _load_env(self):
        env_paths = [
            Path(__file__).parent.parent / '.env',
            Path.cwd() / '.env',
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment variables: {env_path}")
                return

        print("Warning: .env file not found, will use system environment variables")

    def _replace_env_vars(self, config: Any) -> Any:
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            if config.startswith('${') and config.endswith('}'):
                var_name = config[2:-1]
                return os.getenv(var_name, config)
            return config
        else:
            return config

    def load_model_config(self) -> Dict[str, Any]:
        if self._model_config is None:
            config_path = self._base_dir / 'config' / 'model_config.yaml'

            if not config_path.exists():
                raise FileNotFoundError(f"Model config file not found: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                self._model_config = yaml.safe_load(f)

            self._model_config = self._replace_env_vars(self._model_config)

            print(f"Loaded model config: {config_path}")

        return self._model_config

    def load_prompts_config(self) -> Dict[str, Any]:
        if self._prompts_config is None:
            config_path = self._base_dir / 'config' / 'prompts.yaml'

            if not config_path.exists():
                raise FileNotFoundError(f"Prompts config file not found: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                self._prompts_config = yaml.safe_load(f)

            print(f"Loaded prompts config: {config_path}")

        return self._prompts_config

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        model_config = self.load_model_config()

        if agent_name not in model_config:
            raise KeyError(f"Agent config not found: {agent_name}")

        return model_config[agent_name]

    def get_agent_prompt(self, agent_name: str) -> Dict[str, str]:
        prompts_config = self.load_prompts_config()

        if agent_name not in prompts_config:
            raise KeyError(f"Agent prompts not found: {agent_name}")

        return prompts_config[agent_name]

    def get_api_config(self) -> Dict[str, str]:
        return self.get_platform_config('default')

    def get_platform_config(self, platform_name: str) -> Dict[str, str]:
        model_config = self.load_model_config()
        platforms = model_config.get('platforms', {})

        if platform_name not in platforms:
            raise KeyError(f"Platform config not found: {platform_name}. Available platforms: {list(platforms.keys())}")

        return platforms[platform_name]

    def get_available_platforms(self) -> List[str]:
        model_config = self.load_model_config()
        platforms = model_config.get('platforms', {})
        return list(platforms.keys())

    def get_server_config(self) -> Dict[str, Any]:
        model_config = self.load_model_config()
        return model_config.get('server', {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False
        })

    def reload(self):
        self._model_config = None
        self._prompts_config = None
        print("Configs reloaded")


config_loader = ConfigLoader()
