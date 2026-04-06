import os
from typing import Any, Dict, List, AsyncGenerator, Optional, Tuple
from openai import AsyncOpenAI
from abc import abstractmethod
import asyncio
from langfuse.decorators import langfuse_context, observe
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from httpx import Limits, AsyncClient
import numpy as np
import aiohttp
import time
from langfuse.client import ChatPromptClient
from typing import TypedDict


class LlmCallDiagnosticInfo(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_token: int
    TTFT: float


class AbstractLlmClient:
    async def invoke(self, message: List[Any], prompt: ChatPromptClient | None = None):
        a = self.invoke_stream(message, prompt)
        content = ""
        async for delta_content in a:
            if delta_content:
                content += delta_content
        return content
    
    async def invoke_reasoning(self, message: List[Any], prompt: ChatPromptClient | None = None):
        a = self.invoke_stream_reasoning(message, prompt)
        reasoning_result = ""
        result = ""
        async for reasoning_content, content in a:
            if reasoning_content:
                reasoning_content += reasoning_content
            elif content:
                result += content
        return reasoning_result,result

    @abstractmethod
    async def invoke_stream(
        self, message: List[Any], prompt: ChatPromptClient | None = None
    ) -> AsyncGenerator[str, None]:
        yield ""
    
    @abstractmethod
    async def invoke_stream_reasoning(
        self, message: List[Any], prompt: ChatPromptClient | None = None
    ) -> AsyncGenerator[Tuple[str,str], None]:
        yield "",""


client = AsyncOpenAI(
    api_key="sk-00000000000000000000",  # 目前VLLM没有设定KEY，该字段仅作为占位符
    base_url="http://172.17.160.46:8080/v1",
    # base_url="http://127.0.0.1:3002/v1",
    timeout=1200,
    http_client=AsyncClient(
        base_url="http://172.17.160.46:8080/v1",
        # base_url="http://127.0.0.1:3002/v1",
        limits=Limits(
            max_connections=1024,  # 最大连接数
        ),
        timeout=1200,
    ),
)

# client = AsyncOpenAI(
#     api_key="sk-00000000000000000000",  # 目前VLLM没有设定KEY，该字段仅作为占位符
#     base_url=os.environ["OPENAI_BASE_URL"],
#     timeout=1200,
#     http_client=AsyncClient(
#         base_url=os.environ["OPENAI_BASE_URL"],
#         limits=Limits(
#             max_connections=1024,  # 最大连接数
#         ),
#         timeout=1200,
#     ),
# )
# client_for_embedding
client_ = AsyncOpenAI(
    api_key=os.environ["EMBEDDING_API_KEY"],
    base_url=os.environ["EMBEDDING_BASE_URL"],
)


class MockLlmClient(AbstractLlmClient):
    def __init__(self) -> None:
        super().__init__()

    async def invoke_stream(self, message: List[Dict[str, str]], prompt: ChatPromptClient | None = None):
        content = """
以下是财务数据表格形式的展示  

|  左对齐 | 右对齐  | 居中对齐  |
| ------| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |

好的，我给你**指南针**的k线图  
今天的走势是：
"""
        for token in content:
            await asyncio.sleep(0.01)
            yield token
import math
from typing import List, Dict, Union, Any

import math
from typing import List, Dict, Union, Any

def process_intent_conformal_prediction(
    top_logprobs_objects: List[Any], 
    valid_keywords: List[str],
    confidence_threshold: float = 0.95,
    noise_threshold: float = 0.05,
    dominance_ratio: float = 5.0,
    max_output_num: int = 3  # 【新增】最大允许输出的意图数量
) -> Dict[str, Any]:
    """
    处理意图识别的共形预测 + 优势判别 + 熔断风控
    """
    candidates = []
    
    # 1. 数据清洗与概率转换
    for item in top_logprobs_objects:
        token_text = item.token.strip().replace('"', '')
        matched_intent = None
        for keyword in valid_keywords:
            if token_text and keyword.startswith(token_text):
                matched_intent = keyword
                break
        
        if matched_intent:
            prob = math.exp(item.logprob)
            candidates.append({
                "intent": matched_intent,
                "prob": prob
            })

    # 排序
    candidates.sort(key=lambda x: x["prob"], reverse=True)

    # ---------------------------------------------------------
    # 【新增风控 1】总量检查：如果所有识别出的意图加起来都不到阈值，说明模型彻底不知道在说什么
    # ---------------------------------------------------------
    total_valid_prob = sum(x["prob"] for x in candidates)
    if not candidates or total_valid_prob < confidence_threshold:
        return {
            "decision": "unknown", 
            "reason": "low_total_confidence", # 拒绝原因：总置信度不足
            "top1_conf": candidates[0]["prob"] if candidates else 0.0,
            "is_ambiguous": False,
            "final_set_details": []
        }

    # 2. 共形预测：累加概率
    prediction_set = []
    cumulative_prob = 0.0
    
    for cand in candidates:
        prediction_set.append(cand)
        cumulative_prob += cand["prob"]
        if cumulative_prob >= confidence_threshold:
            break
            
    # 3. 工程化去噪 (Dominance Check)
    final_output = []
    if len(prediction_set) == 1:
        final_output = prediction_set
    else:
        top1 = prediction_set[0]
        final_output.append(top1) 
        
        for i in range(1, len(prediction_set)):
            current = prediction_set[i]
            ratio = top1["prob"] / current["prob"]
            if current["prob"] > noise_threshold and ratio < dominance_ratio:
                final_output.append(current)

    # ---------------------------------------------------------
    # 【新增风控 2】数量检查：如果最终保留的意图超过 max_output_num (比如2个)
    # 说明问题太模糊，强行回答不如不回答
    # ---------------------------------------------------------
    if len(final_output) > max_output_num:
        return {
            "decision": "unknown",
            "reason": "too_ambiguous", # 拒绝原因：歧义太大（候选项太多）
            "top1_conf": candidates[0]["prob"],
            "is_ambiguous": False,
            "final_set": [], 
            "final_set_details": [
                {"intent": x["intent"], "prob": round(x["prob"], 4)} 
                for x in final_output
            ]
        }

    # 4. 封装最终结果 (能走到这里说明：置信度够高，且选项在 1~2 个之间)
    result = {
        "top1_intent": candidates[0]["intent"],     
        "top1_conf": candidates[0]["prob"],         
        
        "final_set": [x["intent"] for x in final_output], 
        
        "final_set_details": [
            {"intent": x["intent"], "prob": round(x["prob"], 4)} 
            for x in final_output
        ],
        
        "is_ambiguous": len(final_output) > 1,     
    }
    
    if len(final_output) == 1:
        result["decision"] = final_output[0]["intent"]
    else:
        result["decision"] = [x["intent"] for x in final_output]
        
    return result


class LlmClient(AbstractLlmClient):
    def __init__(self, temperature: Optional[float] = None) -> None:
        self.char_threshold = 5000
        self.max_token_limit = 10000
        self.temperature = temperature




    # @observe(name="大模型调用", capture_input=False, as_type="generation")
    async def invoke_stream(
        self, message: List[Any], prompt: ChatPromptClient | None = None, run_metadata: Dict[str, Any] = None
    ) -> AsyncGenerator[str, None]:
        langfuse_context.update_current_observation(
            input=message,
            # model="/Data1/wyf_workspace/Qwen2.5-7B-Instruct",
            model=os.environ["OPENAI_MODEL_BASE"],
            prompt=prompt,
        )
        total_chars = sum(len(m.get("content", "")) for m in message)
        if total_chars > self.char_threshold:
            pass
            # print(
            #     f" 本次普通模型输入上下文字符数为 {total_chars}，超过阈值 {self.char_threshold}"
            # )
            # for msg in message:
            #     print(msg["content"])
        if run_metadata is None:
            run_metadata = {}
        full_response_content = ""
        diagnostic_info: LlmCallDiagnosticInfo = {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
            "cached_prompt_token": -1,
            "TTFT": -1,
        }
        chat_completion = await client.chat.completions.create(
            # model="/Data1/wyf_workspace/Qwen2.5-7B-Instruct",
            model=os.environ["OPENAI_MODEL_BASE"],
            messages=message,
            stream=True,
            stream_options={
                "include_usage": True,
            },
            logprobs=True,
            temperature=0,
            top_logprobs=20,
            # temperature=self.temperature,
        )
        print(client)

        INTENT_KEYWORDS = ["新闻类", "知识库类", "选股类", "K线图类", "分时图类", "诊股类", "推荐类", "预测类", "身份类", "策略类", "指标查询类", "通用类"]

        # 用于存储最终结果的变量
        intent_calculated = False
        final_intent = None
        final_confidence = 0.0


        # 更新 Langfuse 观察数据
        start_time = time.time()
        token_count = 0

        async for chat_chunk in chat_completion:
            if chat_chunk.choices:
                if diagnostic_info["TTFT"] == -1:
                    diagnostic_info["TTFT"] = time.time() - start_time
                chat_chunk: ChatCompletionChunk = chat_chunk
                
                
                delta_content = chat_chunk.choices[0].delta.content
                if delta_content:
                    full_response_content += delta_content

                # --- 核心修改逻辑开始 ---
                # 只有当我们还没计算过意图，且当前 chunk 包含 logprobs 时才进入
                if not intent_calculated and chat_chunk.choices[0].logprobs and chat_chunk.choices[0].logprobs.content:
                    try:
                        token_data = chat_chunk.choices[0].logprobs.content[0]
                        token_text = token_data.token
                        
                        # 粗筛：当前 token 是否看起来像是一个意图的开始
                        clean_token = token_text.strip().replace('"', '')
                        
                        # 只要命中任意一个关键词的首字，我们就认为这是“决策点”
                        is_hit = False
                        for kw in INTENT_KEYWORDS:
                            if clean_token and kw.startswith(clean_token):
                                is_hit = True
                                break
                        
                        if is_hit:
                            print(f"--- 捕获意图决策点: {token_text} ---")
                            
                            # 【修改点2】调用外部独立函数处理逻辑
                            # 注意：我们需要传入 top_logprobs 列表
                            if token_data.top_logprobs:
                                analysis_result = process_intent_conformal_prediction(
                                    top_logprobs_objects=token_data.top_logprobs,
                                    valid_keywords=INTENT_KEYWORDS,
                                    confidence_threshold=0.95,
                                    noise_threshold=0.05,
                                    dominance_ratio=5.0
                                )
                                
                                # 【修改点3】将结果写入 metadata
                                # 你的上层代码可以检查 'decision' 是字符串还是列表
                                run_metadata["intent_analysis"] = analysis_result
                                run_metadata["predicted_intent"] = analysis_result["decision"] 
                                run_metadata["confidence"] = analysis_result["top1_conf"]
                                
                                print(f"分析结果: {analysis_result['decision']}")
                                print(f"是否模糊: {analysis_result['is_ambiguous']}")
                                
                            intent_calculated = True # 标记已计算，后续 Token 不再处理
                    except Exception as e: # <--- 【新增】加上 Except
                        print(f"警告：意图识别逻辑计算出错，但不影响正常生成。错误信息: {e}")
                token_count += 1  # 每来一段，就累加一次
                if token_count > self.max_token_limit:
                    await chat_completion.close()
                    raise Exception(f"累计接受 token 数超过 {self.max_token_limit}，大模型服务出错")
                if delta_content is not None:
                    yield delta_content
            elif chat_chunk.usage:
                diagnostic_info["prompt_tokens"] = chat_chunk.usage.prompt_tokens
                diagnostic_info["completion_tokens"] = chat_chunk.usage.completion_tokens
                diagnostic_info["total_tokens"] = chat_chunk.usage.total_tokens
                detail = chat_chunk.usage.prompt_tokens_details
                if detail and detail.cached_tokens is not None:
                    diagnostic_info["cached_prompt_token"] = detail.cached_tokens
                else:
                    diagnostic_info["cached_prompt_token"] = -1
        
        langfuse_context.update_current_observation(metadata=diagnostic_info, prompt=prompt)
        from langfuse.types import ModelUsage

        model_usage: ModelUsage = {
            "input": diagnostic_info["prompt_tokens"],
            "output": diagnostic_info["completion_tokens"],
            "total": diagnostic_info["total_tokens"],
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
            "unit": "tokens",
        }
        # langfuse_context.update_current_observation(
        #     usage=model_usage,
        #     metadata=diagnostic_info,
        # )

    @observe(name="深度思考大模型调用", capture_input=False, as_type="generation")
    async def invoke_stream_reasoning(self, message: List[Any], prompt: ChatPromptClient | None = None):
        langfuse_context.update_current_observation(
            input=message,
            model=os.environ["OPENAI_INFERENCE"],
            prompt=prompt,
        )
        total_chars = sum(len(m.get("content", "")) for m in message)
        if total_chars > self.char_threshold:
            pass
            # print(f" 本次深度思考模型输入上下文字符数为 {total_chars}，超过阈值 {self.char_threshold}")

        diagnostic_info: LlmCallDiagnosticInfo = {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
            "cached_prompt_token": -1,
            "TTFT": -1,
        }

        chat_completion = await client.chat.completions.create(
            model=os.environ["OPENAI_INFERENCE"],
            messages=message,
            stream=True,
            stream_options={"include_usage": True},
            temperature=self.temperature,
        )
        start_time = time.time()
        time_to_first_token = -1
        think_finished = False
        token_count = 0
        async for chat_chunk in chat_completion:
            if chat_chunk.choices:
                reasoning_content = ""
                content = ""
                if diagnostic_info["TTFT"] == -1:
                    diagnostic_info["TTFT"] = time.time() - start_time

                # 关键修复：先判断 choices 存在，再取 [0]，杜绝 IndexError
                delta = chat_chunk.choices[0].delta
                delta_content = delta.content if delta.content is not None else ""

                token_count += 1  # 每来一段，就累加一次
                if token_count > self.max_token_limit:
                    await chat_completion.close()
                    raise Exception(f"累计接受 token 数超过 {self.max_token_limit}，大模型服务出错")

                if delta_content is not None:
                    if not think_finished:
                        if delta_content.strip() == "</think>":
                            think_finished = True
                            token_count = 0
                            continue
                        reasoning_content = delta_content
                    else:
                        content = delta_content

                    yield reasoning_content, content

            elif chat_chunk.usage:
                diagnostic_info["prompt_tokens"] = chat_chunk.usage.prompt_tokens
                diagnostic_info["completion_tokens"] = chat_chunk.usage.completion_tokens
                diagnostic_info["total_tokens"] = chat_chunk.usage.total_tokens
                detail = chat_chunk.usage.prompt_tokens_details
                if detail and detail.cached_tokens is not None:
                    diagnostic_info["cached_prompt_token"] = detail.cached_tokens
                else:
                    diagnostic_info["cached_prompt_token"] = -1

        langfuse_context.update_current_observation(
            input=message,
            model=os.environ["OPENAI_INFERENCE"],
            metadata=diagnostic_info,
        )


class DeepSeekCloudClient(AbstractLlmClient):
    def __init__(self) -> None:
        super().__init__()

    async def invoke_stream(self, message: List[Any], prompt: ChatPromptClient | None = None):
        deepseek_client = AsyncOpenAI(
            api_key="325e800c-2231-4497-8c83-e468d8ad4654",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=1200,
            http_client=AsyncClient(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                limits=Limits(
                    max_connections=1024,  # 最大连接数
                ),
                timeout=1200,
            ),
        )
        chat_completion = await deepseek_client.chat.completions.create(
            model="ep-20250210161946-5fsv6",
            messages=message,
            stream=True,
            temperature=0.0,
        )
        # 更新 Langfuse 观察数据
        start_time = time.time()
        time_to_first_token = -1
        token_count = 0
        async for chat_chunk in chat_completion:
            if time_to_first_token == -1:
                time_to_first_token = time.time() - start_time
            chat_chunk: ChatCompletionChunk = chat_chunk
            delta_content = chat_chunk.choices[0].delta.content
            token_count += 1  # 每来一段，就累加一次
            if delta_content is not None:
                yield delta_content

class GLMClient(AbstractLlmClient):
    def __init__(self,temperature: Optional[float] = None) -> None:
        super().__init__()
        self.temperature = temperature
        self.glm_client = AsyncOpenAI(
            api_key="fc24b20331d04a2f87cd8f4ea0943dde.Y38Zww13OkP25KVP",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            timeout=1200,
            http_client=AsyncClient(
                base_url="https://open.bigmodel.cn/api/coding/paas/v4",
                limits=Limits(
                    max_connections=1024,  # 最大连接数
                ),
                timeout=1200,
            ),
        )

    async def invoke_stream(self, message: List[Any], prompt: ChatPromptClient | None = None):
        chat_completion = await self.glm_client.chat.completions.create(
            model="glm-4.7",
            messages=message,
            stream=True,
            temperature=self.temperature,
            extra_body={
                "thinking": {"type": "disabled"},
            },
        )
        # 更新 Langfuse 观察数据
        async for chat_chunk in chat_completion:
            chat_chunk: ChatCompletionChunk = chat_chunk
            delta_content = chat_chunk.choices[0].delta.content
            if delta_content is not None:
                yield delta_content
    
    async def invoke_stream_reasoning(self, message: List[Any], prompt: ChatPromptClient | None = None):
        chat_completion = await self.glm_client.chat.completions.create(
            model="glm-4.7",
            messages=message,
            stream=True,
            temperature=self.temperature
        )
        think_finished = False
        async for chat_chunk in chat_completion:
            if chat_chunk.choices:
                reasoning_content = ""
                content = ""

                delta = chat_chunk.choices[0].delta
                delta_content = delta.content if delta.content is not None else ""

                if delta_content is not None:
                    if not think_finished:
                        if delta_content.strip() == "</think>":
                            think_finished = True
                            continue
                        reasoning_content = delta_content
                    else:
                        content = delta_content
                    yield reasoning_content, content

class GLMClient_Local(AbstractLlmClient):
    def __init__(self,temperature: Optional[float] = None) -> None:
        super().__init__()
        self.temperature = temperature
        self.glm_client = AsyncOpenAI(
            api_key="123456",
            base_url="http://172.17.160.47:8080/v1",
            timeout=1200,
            http_client=AsyncClient(
                base_url="http://172.17.160.47:8080/v1",
                limits=Limits(
                    max_connections=1024,  # 最大连接数
                ),
                timeout=1200,
            ),
        )

    async def invoke_stream(self, message: List[Any], prompt: ChatPromptClient | None = None):
        chat_completion = await self.glm_client.chat.completions.create(
            model="DeepSeek-R1-Distill-Qwen-32B",
            messages=message,
            stream=True,
            temperature=self.temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        async for chat_chunk in chat_completion:
            if chat_chunk.choices:
                content = getattr(chat_chunk.choices[0].delta, "reasoning_content", None)
                if content:
                    yield content
    
    async def invoke_stream_reasoning(self, message: List[Any], prompt: ChatPromptClient | None = None):
        chat_completion = await self.glm_client.chat.completions.create(
            model="DeepSeek-R1-Distill-Qwen-32B",
            messages=message,
            stream=True,
            temperature=self.temperature
        )
        async for chat_chunk in chat_completion:
            if chat_chunk.choices:
                reasoning_content = getattr(chat_chunk.choices[0].delta, "reasoning_content", None)
                content = getattr(chat_chunk.choices[0].delta, "content", None)
                if reasoning_content:
                    yield reasoning_content,""
                elif content:
                    yield "",content


# OPENAI_BASE_URL_STATIC = "http://172.17.160.41:8001/v1"
# OPENAI_MODEL_BASE_STATIC = "/Data5/models/Qwen2.5-14B-Instruct/"
# client_static = AsyncOpenAI(
#     api_key="sk-00000000000000000000", # 目前VLLM没有设定KEY，该字段仅作为占位符
#     base_url=OPENAI_BASE_URL_STATIC,
# )


class EmbeddingClient:
    def __init__(self) -> None:
        self.char_threshold = 512

    async def embedding(self, input: List[str] | str) -> np.ndarray:
        """
        生成嵌入向量。

        Args:
            texts (list): 需要生成嵌入的文本列表。

        Returns:
            list: 嵌入向量列表。
        """
        if isinstance(input, str):
            input = [input]

        # 检查每个文本的字符长度是否超过阈值
        for i, text in enumerate(input):
            if len(text) > self.char_threshold:
                print(f"第 {i + 1} 个文本字符数为 {len(text)}，超过阈值 {self.char_threshold}")
        try:
            completion = await client_.embeddings.create(
                model=os.environ["EMBEDDING_MODEL"],
                input=input,
                encoding_format="float",
            )

            # 提取嵌入向量
            embeddings = [item.embedding for item in completion.data]

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"调用 OpenAI Embedding API 失败: {e}")
            raise e


class BaiduTextCensorClient:
    def __init__(self):
        self.client_id = "KXFJEXS6VPRU69QGmyBkgb4a"
        self.client_secret = "PyhigAk1bdOlXzOG6CKs3DXIwg3xezwd"
        self.access_token = None
        self.token_expiry_time = None

    async def get_access_token(self):
        """获取百度API的access_token"""
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.client_id}&client_secret={self.client_secret}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as response:
                data = await response.json()
                self.access_token = data["access_token"]
                # 设置token的过期时间，通常是 30 天
                self.token_expiry_time = time.time() + data["expires_in"]
                return self.access_token

    async def check_text(self, text: str):
        """检查文本是否合法"""
        # 如果 access_token 过期，重新获取
        if self.access_token is None or (self.token_expiry_time is not None and time.time() > self.token_expiry_time):
            await self.get_access_token()

        censor_url = f"https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined?access_token={self.access_token}"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"text": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(censor_url, data=data, headers=headers) as response:
                result = await response.text()
                return result


class AsyncNoticeClient(AbstractLlmClient):
    def __init__(self) -> None:
        pass

    async def invoke_stream(self, message: List[Any], prompt: ChatPromptClient | None = None):
        aynsc_client = AsyncOpenAI(
            api_key="sk-00000000000000000000",  # 目前VLLM没有设定KEY，该字段仅作为占位符
            base_url=os.environ.get("ASYNC_OPENAI_BASE_URL", "http://172.17.160.46:8080/v1"),
            timeout=1200,
            http_client=AsyncClient(
                base_url=os.environ.get("ASYNC_OPENAI_BASE_URL", "http://172.17.160.46:8080/v1"),
                limits=Limits(
                    max_connections=1024,  # 最大连接数
                ),
                timeout=1200,
            ),
        )
        chat_completion = await aynsc_client.chat.completions.create(
            model=os.environ.get("ASYNC_MODEL", "Qwen3-30B-A3B"),
            messages=message,
            stream=True,
            temperature=0.01,
            top_p=0.01,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        # 更新 Langfuse 观察数据
        start_time = time.time()
        time_to_first_token = -1
        token_count = 0
        async for chat_chunk in chat_completion:
            if time_to_first_token == -1:
                time_to_first_token = time.time() - start_time
            chat_chunk: ChatCompletionChunk = chat_chunk
            delta_content = chat_chunk.choices[0].delta.content
            token_count += 1  # 每来一段，就累加一次
            if delta_content is not None:
                yield delta_content


llm_client = LlmClient()
embedding_client = EmbeddingClient()
baidu_text_censor_client = BaiduTextCensorClient()
