import json
import random
import re
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal, Optional, TypeVar
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from src.language_models import LanguageModel
from src.log import logger, LYELLOW, RESET


# @retry(stop=stop_after_attempt(3), reraise=False, retry_error_callback=lambda x: None)
@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(
        (openai.APIConnectionError, openai.RateLimitError, openai.APIError)),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying ({retry_state.attempt_number}/3) due to: {retry_state.outcome.exception()}"
    )
)
def ask_model(
    model: LanguageModel,
    prompt: str,
    system_msg: str = None,
    response_type: Literal["json", "text"] = "json",
    validator: Optional[Callable] = None,
    sleep: bool = True,
    mode: Literal["chat", "completion"] = "chat",
) -> dict:
    # 1. 随机延迟防止速率限制
    if sleep:
        time.sleep(random.uniform(1.0, 3.0))

     # 2. 调用模型
    try:
        if mode == "chat":
            response = model.chat(prompt, system_msg,
                                  json_mode=(response_type == "json"))
        elif mode == "completion":
            response = model.complete(prompt)
        else:
            logger.error(f"Invalid mode: {mode}")
            raise
    except Exception as e:
        logger.error(f"Model call failed: {str(e)}")
        raise

    # 3. 检查空响应
    if not response or not response.strip():
        logger.error("Received empty response")
        raise ValueError("Empty response from model")

    # 4. 解析响应
    try:
        parsed = get_type_parser(response_type)(response)
    except Exception as e:
        logger.error(f"Failed to parse response: {e}\nResponse: {response}")
        raise ValueError(f"Response parsing failed")

    # 5. 验证响应
    if validator and not validator(parsed):
        logger.error(f"Validation failed for response: {parsed}")
        raise ValueError("Response validation failed")

    return parsed


def ask_model_in_parallel(
    model: LanguageModel,
    prompts: list[str],
    system_msg: str = None,
    type: Literal["json", "text"] = "json",
    check_if_valid_list: list[Callable] = None,
    max_workers: int = 4,
    desc: str = "Processing...",
    verbose=True,
    mode: Literal["chat", "completion"] = "chat",
):
    if max_workers == -1:
        max_workers = len(prompts)
    assert max_workers >= 1, "max_workers should be greater than or equal to 1"
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if check_if_valid_list is None:
            check_if_valid_list = [None] * len(prompts)
        assert len(prompts) == len(
            check_if_valid_list
        ), "Length of prompts and check_if_valid_list should be the same"
        tasks = {
            executor.submit(
                ask_model, model, prompt, system_msg, type, check_if_valid, mode
            ): idx
            for idx, (prompt, check_if_valid) in enumerate(
                zip(prompts, check_if_valid_list)
            )
        }
        results = []
        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc=desc, disable=not verbose
        ):
            task_id = tasks[future]
            try:
                result = future.result()
                results.append((task_id, result))
            finally:
                ...
        results = [result[1] for result in sorted(results, key=lambda r: r[0])]
        return results


def get_type_parser(type: str) -> Callable:
    def json_parser(result: str):
        # 情况1：字符串是纯JSON（直接以 { 开头）
        if result.strip().startswith('{'):
            return json.loads(result)

        # 情况2：字符串包含Markdown代码块（```json ... ```）
        pattern = r'```json\n(.*?)\n```'  # 非贪婪匹配
        match = re.search(pattern, result, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass  # 如果代码块内不是合法JSON，继续尝试其他情况

        # 情况3：尝试匹配任意位置的 {...}（兜底方案）
        match = re.search(r'\{.*?\}', result, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 所有情况均失败
        raise ValueError("无法从字符串中解析出合法JSON")

    def text_parser(result: str):
        return result

    if type == "json":
        return json_parser
    elif type == "text":
        return text_parser
    else:
        raise ValueError(f"Unsupported type: {type}")
