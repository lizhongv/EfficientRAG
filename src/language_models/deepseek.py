import json
import os
from time import sleep
from dotenv import load_dotenv, find_dotenv

import openai
from openai import OpenAI
from openai._types import NotGiven

from .base import LanguageModel

# set api & url
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = "sk-768a72eaa5e14187b8edaa67023f27d3"
DASHSCOPE_API_KEY = "sk-8aef464eb019408f94dc3cc5ef63d46e"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SLEEP_SEC = 3


class DeepSeek(LanguageModel):
    def __init__(self, model: str = "deepseek-v3", api_key: str = None, base_url: str = None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        if api_key is None:
            api_key = DASHSCOPE_API_KEY
        if base_url is None:
            base_url = DASHSCOPE_BASE_URL

        print(f"Initiate deepseek client.")
        self.client = OpenAI(api_key=api_key, base_url=base_url,)

    def chat(self, messages: str, system_msg: str = None, **kwargs):
        try:
            response = self._chat(messages, system_msg, **kwargs)
            return response

        except openai.BadRequestError as e:
            err = json.loads(e.response.text)
            if err["error"]["code"] == "content_filter":
                print("Content filter triggered!")
                return None
            exit(f"The OpenAI API request was invalid: {e}")
            return None

        except openai.APIConnectionError as e:
            print(f"The OpenAI API connection failed: {e}")
            sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)

        except openai.RateLimitError as e:
            print(f"Token rate limit exceeded. Retrying after {SLEEP_SEC} second...")
            sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)

        except openai.AuthenticationError as e:
            print(f"Invalid API token: {e}")
            self.update_api_key()
            sleep(SLEEP_SEC)
            return self.chat(messages, system_msg, **kwargs)

        except openai.APIError as e:
            if "The operation was timeout" in str(e):
                # Handle the timeout error here
                print("The OpenAI API request timed out. Please try again later.")
                sleep(SLEEP_SEC)
                return self.chat(messages, system_msg, **kwargs)
            elif "DeploymentNotFound" in str(e):
                print("The API deployment for this resource does not exist")
                return None
            else:
                # Handle other API errors here
                print(f"The OpenAI API returned an error: {e}")
                sleep(SLEEP_SEC)
                return self.chat(messages, system_msg, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")

    def _chat(
        self,
        messages: str,
        system_msg="",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        json_mode: bool = False,
    ):
        if system_msg is None or system_msg == "":
            system_msg = "You are a helpful assistant."
        msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": messages},
        ]
        response = self.client.chat.completions.create(
            # model=self.model,
            model="deepseek-v3",
            response_format={"type": "json_object"} if json_mode else NotGiven(),
            messages=msg,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # aoai = DeepSeek(model="deepseek-chat")
    # print(aoai.chat("Hello, who are you?", json_mode=True))

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
