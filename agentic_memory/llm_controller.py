from typing import Dict, Optional, Literal, Any
import os
import json
from abc import ABC, abstractmethod
from litellm import completion

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                api_key = "dummy-key"  # Use dummy key for local proxies
            
            # Get base URL from environment if set (for local proxies)
            base_url = os.getenv('OPENAI_API_BASE')
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        print(f"[LLM DEBUG START] Model: {self.model}, calling API...")
        
        # Build request parameters
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 4000  # Increased from 1000 to prevent truncation
        }
        
        # gpt-5 models don't support response_format properly - returns empty string
        # Only add response_format for non-gpt-5 models
        if not self.model.startswith("gpt-5"):
            params["response_format"] = response_format
        
        # Only add temperature if it's not default (some models like gpt-5 only support default)
        # Note: gpt-5 only supports temperature=1 (default), so we omit it
        
        try:
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
        except Exception as e:
            print(f"[LLM DEBUG ERROR] Exception during API call: {e}")
            print(f"[LLM DEBUG ERROR] Exception type: {type(e)}")
            raise
        
        # DEBUG: Log what we're returning
        print(f"[LLM DEBUG] Model: {self.model}")
        print(f"[LLM DEBUG] Response content: {repr(content)}")
        print(f"[LLM DEBUG] Response length: {len(content) if content else 0}")
        print(f"[LLM DEBUG] Response type: {type(content)}")
        
        return content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
