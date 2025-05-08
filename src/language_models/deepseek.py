import json
import os
import time
import openai
from openai import OpenAI
from typing import Optional
from openai._types import NotGiven

from src.language_models.base import LanguageModel
from src.log import logger


SLEEP_SEC = 3


class DeepSeek(LanguageModel):
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url or "https://api.deepseek.com"
        )
        logger.info(
            f"Initialized DeepSeek client with model: {model}, api_key: {api_key}, base_url: {base_url}")

    def chat(self, messages: str, system_msg: Optional[str] = None, **kwargs):
        try:
            return self._chat(messages, system_msg or "You are a helpful assistant.", **kwargs)
        except openai.BadRequestError as e:
            logger.error(f"Invalid request: {e}, input: {messages}")
            raise
        except openai.AuthenticationError as e:
            logger.error(
                f"Invalid API token: {e}, api_key: {self.client.api_key}, base_url: {self.client.base_url}")
            raise
        except openai.APIConnectionError as e:
            logger.error(
                f"The API connection failed: {e}, Retrying after {SLEEP_SEC} second...")
            time.sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)
        except openai.RateLimitError as e:
            logger.error(
                f"Token rate limit exceeded: {e}, Retrying after {SLEEP_SEC} second...")
            time.sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)
        except openai.APIError as e:
            logger.error(
                f"The API returned an error: {e}, Retrying after {SLEEP_SEC} second...")
            time.sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def _chat(
        self,
        messages: str,
        system_msg="",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,  # 不支持 https://bailian.console.aliyun.com/?spm=5176.29619931.0.0.74cd521cnbyZzt&tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2868565.html&renderType=iframe
        presence_penalty: float = 0.0,  # 默认0.95
        json_mode: bool = False,
        *args, **kwargs
    ):

        msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": messages},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={
                "type": "json_object"} if json_mode else NotGiven(),
            messages=msg,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    import os
    from openai import OpenAI
    client = OpenAI(
        # api_key="sk-768a72eaa5e14187b8edaa67023f27d3",
        # base_url="https://api.deepseek.com",

        api_key="sk-8aef464eb019408f94dc3cc5ef63d46e",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

    )
    completion = client.chat.completions.create(
        model="deepseek-v3",  # "deepseek-chat"
        messages=[
            {'role': 'user', 'content': '9.9和9.11谁大'}
        ]
    )
    print(completion.choices[0].message.content)
