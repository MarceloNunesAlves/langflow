from datetime import datetime
import requests
from typing import List
from langchain_ibm import WatsonxLLM
from pydantic.v1 import SecretStr

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.io import DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput


class WatsonxModel(LCModelComponent):
    display_name: str = "Watsonx"
    description: str = "Generate text using Watsonx."
    icon = "Watsonx"
    name = "WatsonxModel"

    inputs = LCModelComponent._base_inputs + [
        SecretStrInput(
            name="watsonx_api_key",
            display_name="Watsonx API Key",
            info="API key for the Watsonx API.",
        ),
        MessageTextInput(
            name="url",
            display_name="Url to Watsonx or instance",
            info="Url to Watson Machine Learning or CPD instance.",
        ),
        MessageTextInput(
            name="project_id",
            display_name="ID of the project.",
            info="ID of the Watson Studio project.",
        ),
        MessageTextInput(
            name="decoding_method",
            display_name="Decoding methods",
            info="Supported decoding methods for text generation.",
            advanced=True,
            value='sample',
        ),
        IntInput(
            name="max_new_tokens",
            display_name="Max Output Tokens",
            info="The maximum number of tokens to generate.",
            advanced=True,
            value=100,
        ),
        IntInput(
            name="min_new_tokens",
            display_name="Min Output Tokens",
            info="The minimum number of tokens to generate.",
            advanced=True,
            value=1,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Run inference with this temperature. Must by in the closed interval [0.0, 1.0].",
            value=0.1,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            advanced=True,
            value=50,
        ),
        IntInput(
            name="top_p",
            display_name="Top P",
            advanced=True,
            value=1,
        ),
        DropdownInput(
            name="model_id",
            display_name="Model",
            info="The name of the model to use.",
            options=[],
            refresh_button=True,
        ),
    ]

    def get_models(self) -> List[str]:
        date_now_str = datetime.now().strftime('%Y-%m-%d')
        url = f"{self.url}/ml/v1/foundation_model_specs?version={date_now_str}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            model_list = response.json()
            return [model["model_id"] for model in model_list.get("resources", [])]
        except requests.RequestException as e:
            self.status = f"Error fetching models: {str(e)}"
            return []

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        if field_name == "url" or field_name == "model_id":
            models = self.get_models()
            build_config["model_id"]["options"] = models
        return build_config

    def build_model(self) -> LanguageModel:
        watsonx_api_key = self.watsonx_api_key
        url = self.url
        project_id = self.project_id
        model_id = self.model_id
        stream = self.stream
        decoding_method = self.decoding_method
        max_new_tokens = self.max_new_tokens
        min_new_tokens = self.min_new_tokens
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p

        parameters = {
            "decoding_method": decoding_method,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        output = WatsonxLLM(
            model_id=model_id,
            url=url,
            project_id=project_id,
            apikey=SecretStr(watsonx_api_key),
            streaming=stream,
            params=parameters,
        )

        return output  # type: ignore
