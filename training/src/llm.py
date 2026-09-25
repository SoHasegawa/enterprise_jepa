import torch
import os
import base64
import time
import math
import re
import ast
import json
import requests
import pandas as pd
import ollama
import anthropic

from io import BytesIO
from PIL import Image
from ollama import chat, Client
from copy import deepcopy
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer


def _openai_reasoning_effort_for_model(model: str) -> str | None:
    configured = os.environ.get("OPENAI_REASONING_EFFORT")
    if configured:
        return configured
    if model.lower().startswith("gpt-5.1"):
        return "none"
    return None


def _strip_model_thinking_output(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(
        r"<\|channel>thought\s*.*?<channel\|>\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


class LLM:
    def __init__(self, method="llama", multimodal=False):
        self.method = method
        self.ollama = False
        if "/" in method:
            if multimodal:
                self.model, self.text_tokenizer = self._qwen_vl_initialize(method)
            elif method.split("/")[0] == "ollama":
                self.ollama_endpoint = "http://localhost:11434"
                self.client = Client(self.ollama_endpoint)
                self.model = method.split("/")[1]
                self.client.pull(self.model)
                self.ollama = True
            elif method.split("/")[0].split(":")[0] == "vllm":
                # `vllm/<model>`             -> port 9000 (default) or $VLLM_ENDPOINT
                # `vllm:<port>/<model>`      -> that port on 127.0.0.1
                # $VLLM_ENDPOINT overrides both (full chat/completions URL).
                prefix, self.vllm_model = method.split("/", 1)
                port = prefix.split(":", 1)[1] if ":" in prefix else "9000"
                self.vllm_endpoint = os.environ.get(
                    "VLLM_ENDPOINT",
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                )
                self.vllm_api_key = os.environ.get("VLLM_API_KEY")
                self.method = "vllm"
            else:
                self.model, self.tokenizer = self._llm_initialize(method)
                #self.model = method
            self.multimodal = multimodal
        else:
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            self.gpt4_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            self.client = OpenAI()
            # self.gpt4_mini_endpoint = os.environ.get("AZURE_OPENAI_MINI_ENDPOINT")
            self.gemini_endpoint = os.environ.get("GEMINI_ENDPOINT")
            self.gpt5_endpoint = os.environ.get("GPT5_ENDPOINT")
            self.gpt5_mini_endpoint = os.environ.get("GPT5_MINI_ENDPOINT")
            self.gpt51_model = os.environ.get("GPT51_MODEL", "gpt-5.1")
            self.vllm_api_key = os.environ.get("VLLM_API_KEY")
            self.qwen3_vllm_endpoint_9000 = os.environ.get(
                "QWEN3_VLLM_ENDPOINT_9000",
                "http://127.0.0.1:9000/v1/chat/completions",
            )
            self.qwen3_vllm_endpoint_9001 = os.environ.get(
                "QWEN3_VLLM_ENDPOINT_9001",
                "http://127.0.0.1:9001/v1/chat/completions",
            )
            self.qwen3_vllm_endpoint_9002 = os.environ.get(
                "QWEN3_VLLM_ENDPOINT_9002",
                "http://127.0.0.1:9002/v1/chat/completions",
            )
            self.qwen3_vllm_model_9000 = os.environ.get(
                "QWEN3_VLLM_MODEL_9000",
                "Qwen/Qwen3-4B",
            )
            self.qwen3_vllm_model_9001 = os.environ.get(
                "QWEN3_VLLM_MODEL_9001",
                "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            )
            self.qwen3_vl_vllm_model_9002 = os.environ.get(
                "QWEN3_VLLM_MODEL_9002",
                "Qwen/Qwen3-VL-4B-Instruct",
            )
            # self.gemini_pro_endpoint = os.environ.get("GEMINI_PRO_ENDPOINT")
            # self.cohere_api_key = os.environ.get("COHERE_API_KEY")
            # self.co = cohere.ClientV2(self.api_key)

        self.system_prompt_enable = False
        self.system_prompt = None

    @staticmethod
    def _strip_json_fence(response):
        if not isinstance(response, str):
            return response
        response = response.strip()
        fence_match = re.match(r"^```(?:json|python)?\s*(.*?)\s*```$", response, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        return response

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"yes", "true", "1"}

    @staticmethod
    def _extract_json_candidates(response):
        if not isinstance(response, str):
            return [response]

        response = LLM._strip_json_fence(response)
        candidates = []

        def add_candidate(value):
            if value is None:
                return
            value = value.strip() if isinstance(value, str) else value
            if value not in candidates:
                candidates.append(value)

        add_candidate(response)

        for opener, closer in [("{", "}"), ("[", "]")]:
            start = response.find(opener)
            end = response.rfind(closer)
            if start != -1 and end != -1 and end > start:
                add_candidate(response[start:end + 1])

        return candidates

    @staticmethod
    def _parse_json_candidate(candidate):
        if not isinstance(candidate, str):
            return candidate

        normalized = candidate.strip()
        normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

        parsers = [
            json.loads,
            ast.literal_eval,
        ]

        fallback_candidates = [
            normalized,
            normalized.replace("\n", " "),
            re.sub(r",\s*([}\]])", r"\1", normalized),
            normalized.replace("None", "null").replace("True", "true").replace("False", "false"),
        ]

        for text in fallback_candidates:
            for parser in parsers:
                try:
                    return parser(text)
                except Exception:
                    continue

        raise ValueError("Unable to parse JSON candidate")

    @staticmethod
    def _coerce_json_to_schema(value, schema):
        if schema is None:
            return value

        if isinstance(schema, dict):
            result = {}
            value = value if isinstance(value, dict) else {}
            for key, item_schema in schema.items():
                result[key] = LLM._coerce_json_to_schema(value.get(key), item_schema)
            return result

        if isinstance(schema, list):
            item_schema = schema[0] if schema else None
            if isinstance(value, dict):
                if item_schema and isinstance(item_schema, dict):
                    normalized_items = []
                    for key, item_value in value.items():
                        if isinstance(item_value, dict):
                            candidate = dict(item_value)
                            candidate.setdefault("trajectory_index", key)
                        else:
                            candidate = {"trajectory_index": key, "question": item_value}
                        normalized_items.append(candidate)
                    value = normalized_items
                else:
                    value = list(value.values())
            elif value is None:
                value = []
            elif not isinstance(value, list):
                value = [value]

            if item_schema is None:
                return value
            return [LLM._coerce_json_to_schema(item, item_schema) for item in value]

        if isinstance(schema, bool):
            return LLM._parse_bool(value)

        if isinstance(schema, int) and not isinstance(schema, bool):
            try:
                return int(value)
            except Exception:
                return int(schema)

        if isinstance(schema, float):
            try:
                return float(value)
            except Exception:
                return float(schema)

        if isinstance(schema, str):
            if value is None:
                return deepcopy(schema)
            if isinstance(value, str):
                return value.strip()
            return str(value)

        if schema is None:
            return value

        return deepcopy(schema)

    @staticmethod
    def _wrap_json(response, schema=None):
        try:
            for candidate in LLM._extract_json_candidates(response):
                try:
                    parsed = LLM._parse_json_candidate(candidate)
                    return LLM._coerce_json_to_schema(parsed, schema)
                except Exception:
                    continue
            print(f"JSON Convert Error: {response}")
            return LLM._coerce_json_to_schema(response, schema) if schema is not None else response
        except Exception:
            print(f"JSON Convert Error: {response}")
            return LLM._coerce_json_to_schema(response, schema) if schema is not None else response

    @staticmethod
    def _wrap_code(response):
        try:
            if response.startswith("```python"):
                response = response.lstrip("```python").rstrip("```")
            elif "```python" in response:
                pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
                result = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
                response = result[0]
                response = response.lstrip("```python").rstrip("```")
            elif response.startswith("<think>"):
                pattern = r"<answer>(.*?)</answer>"
                matches = re.findall(pattern, response)
                response = matches[0]
            # response = json.loads(response)
            # if response.get("code") is not None:
            #     return response["code"]
            # else:
            return response
        except Exception as e:
            return response

    @staticmethod
    def _is_rate_limit_error(exc) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True

        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True

        message = str(exc).lower()
        return any(
            pattern in message
            for pattern in [
                "rate limit",
                "too many requests",
                "429",
                "please try again in",
                "request limit",
            ]
        )

    @staticmethod
    def _extract_retry_delay_seconds(exc):
        headers = {}
        response = getattr(exc, "response", None)
        if response is not None:
            raw_headers = getattr(response, "headers", None)
            if raw_headers is not None:
                try:
                    headers = {str(k).lower(): str(v) for k, v in dict(raw_headers).items()}
                except Exception:
                    try:
                        headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}
                    except Exception:
                        headers = {}

        for key in (
            "retry-after",
            "retry-after-ms",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-reset",
        ):
            value = headers.get(key)
            if not value:
                continue
            try:
                numeric = float(value)
                if "ms" in key:
                    return max(numeric / 1000.0, 1.0)
                if "reset" in key:
                    return max(numeric, 1.0)
                return max(numeric, 1.0)
            except Exception:
                match = re.search(r"(\d+(?:\.\d+)?)", value)
                if match:
                    numeric = float(match.group(1))
                    if "ms" in key:
                        return max(numeric / 1000.0, 1.0)
                    return max(numeric, 1.0)

        message = str(exc)
        for pattern, multiplier in (
            (r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds)", 0.001),
            (r"try again in\s+(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds)", 1.0),
            (r"try again in\s+(\d+(?:\.\d+)?)\s*(m|min|mins|minutes)", 60.0),
            (r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds)", 0.001),
            (r"retry after\s+(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds)", 1.0),
            (r"retry after\s+(\d+(?:\.\d+)?)\s*(m|min|mins|minutes)", 60.0),
        ):
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return max(float(match.group(1)) * multiplier, 1.0)

        return None

    def _wait_for_rate_limit(self, exc, provider: str, attempt: int):
        delay = self._extract_retry_delay_seconds(exc)
        if delay is None:
            delay = min(60.0 * max(1, attempt), 900.0)
        delay = max(delay, 1.0)
        delay = math.ceil(delay)
        print(f"{provider} rate limit reached. Waiting {delay} seconds before retrying.")
        time.sleep(delay)

    def _ovis_initialize(self, model):
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                    #  load_in_8bit=True,
                                                    #  low_cpu_mem_usage=True,
                                                     device_map="auto",
                                                     multimodal_max_length=32768,
                                                     trust_remote_code=True)
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        return model, text_tokenizer, visual_tokenizer
    
    def _qwen_vl_initialize(self, model_path):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(model_path, min_pixels=1280*28*28, max_pixels=16384*28*28)

        return model, processor
    
    def _llm_initialize(self, model):
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     trust_remote_code=True
                                                     )

        return model, tokenizer

    @staticmethod
    def _extract_text_block(payload):
        if isinstance(payload, str):
            value = payload.strip()
            return value or None
        if not isinstance(payload, dict):
            return None

        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            content_text = []
            for item in content:
                extracted = LLM._extract_text_block(item)
                if extracted:
                    content_text.append(extracted)
            if content_text:
                return "\n".join(content_text)

        parts = payload.get("parts")
        if isinstance(parts, list):
            part_text = []
            for item in parts:
                extracted = LLM._extract_text_block(item)
                if extracted:
                    part_text.append(extracted)
            if part_text:
                return "\n".join(part_text)

        return None

    @staticmethod
    def _extract_response_text(response):
        if not isinstance(response, dict):
            return None

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            extracted = LLM._extract_text_block(message)
            if extracted:
                return extracted

        candidates = response.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            extracted = LLM._extract_text_block(content)
            if extracted:
                return extracted

        return None

    @staticmethod
    def _summarize_empty_response(response):
        if not isinstance(response, dict):
            return str(response)

        error = response.get("error")
        if error is not None:
            try:
                return json.dumps({"error": error}, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(error)

        summary = {}
        prompt_feedback = response.get("promptFeedback")
        if prompt_feedback is not None:
            summary["promptFeedback"] = prompt_feedback

        candidates = response.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            candidate = candidates[0]
            candidate_summary = {}
            for key in ("finishReason", "finishMessage", "safetyRatings"):
                if candidate.get(key) is not None:
                    candidate_summary[key] = candidate.get(key)
            if candidate_summary:
                summary["candidate"] = candidate_summary

        if not summary:
            summary = response

        try:
            serialized = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        except Exception:
            serialized = str(summary)
        return serialized[:1000]
    
    def _execute(self, data, index, api_key, endpoint):
        headers = {
                'Content-type': 'application/json',
                'api-key': api_key,
        }

        generated_text = None
        limit_count = 0
        last_error = None

        while generated_text is None and limit_count < 5:
            try:
                response = requests.post(endpoint,
                                         headers=headers,
                                         json=data,
                                         timeout=300)
                response.raise_for_status()
                response_payload = response.json()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(60)
                limit_count += 1
                continue

            generated_text = self._extract_response_text(response_payload)
            if generated_text is None:
                last_error = self._summarize_empty_response(response_payload)
                time.sleep(60)
                limit_count += 1

        if generated_text is None:
            raise RuntimeError(
                "LLM endpoint did not return any response text after "
                f"{limit_count} attempts. Last response summary: {last_error}"
            )

        response_dict = {
            "generated_text": generated_text,
            "index": index
        }

        return response_dict

    def _generate_ovis(self, prompt, image_paths=None):
        if image_paths is not None:
            images = [Image.open(image_path) for image_path in image_paths]
            query = f'<image>\n{prompt}'
        else:
            images = None
            query = f'{prompt}'
        max_partition = 9

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=0.0,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

        return output

    def _generate_qwen_vl(self, prompt, image_paths):
        system_prompt = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since 1+1=2, so the answer is 2. </think><answer> 2 </answer>, which means assistant's output should start with <think> and end with </answer>."

        generate_kwargs = dict(
            max_new_tokens=2048,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0
        )

        # Prepare input with image and text
        messages = [{"role": "system", "content": system_prompt}]
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                data.append({"type": "image", "image": image_path})
        message = {"role": "user", "content": data}
        messages.append(message)

        # Preparation for inference
        text = self.text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.text_tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def _generate_local_llm(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=8192, temperature=1e-9)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(response)

        return response
    
    def _generate_ollama(self, prompt, image_paths=None):
        if image_paths is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            print(prompt, image_paths)
            messages = [{"role": "user", "content": prompt, "images": image_paths}]

        response = self.client.chat(model=self.model, messages=messages, options={"temperature": 0.0, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1, "max_tokens": 1500})
        response = response['message']['content']
        if "</think>" in response:
            response = response.split("</think>")[1]
        if "</response>" in response:
            response = response.split("</response>")[1].replace("  ", "")

        return response

    def _generate_vllm(self, prompt, system_prompt=None, image_paths=None, temperature=0.0, endpoint=None, model=None):
        user_content = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                if isinstance(image_path, str):
                    with open(image_path, "rb") as image_file:
                        d = base64.b64encode(image_file.read()).decode("utf-8")
                elif isinstance(image_path, Image.Image):
                    buffered = BytesIO()
                    image_path.save(buffered, format="JPEG")
                    d = base64.b64encode(buffered.getvalue()).decode("utf-8")
                else:
                    continue
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}}
                )

        messages = []
        if system_prompt not in [None, ""]:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        headers = {"Content-type": "application/json"}
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        # `max_tokens` is a hard latency knob, not just a safety cap: single-stream decode on a
        # 9B-class model runs ~20-25 ms/token, so an unbounded 2000-token budget lets one
        # rambling reply cost tens of seconds. Callers that know their output size (the agent
        # generators pass their --max-new-tokens) set `vllm_max_tokens`; 2000 stays the default
        # for direct LLM(...) users.
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": int(getattr(self, "vllm_max_tokens", 0) or 2000),
            "temperature": temperature,
        }

        generated_text = None
        limit_count = 0
        while generated_text is None and limit_count < 5:
            response = requests.post(
                endpoint,
                headers=headers,
                json=body,
                timeout=300,
            ).json()
            if response.get("choices") is not None:
                generated_text = response["choices"][0]["message"]["content"]
            else:
                time.sleep(5)
                limit_count += 1

        return generated_text

    def _generate_qwen3_vllm_9000(self, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        return self._generate_vllm(
            prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            temperature=temperature,
            endpoint=self.qwen3_vllm_endpoint_9000,
            model=self.qwen3_vllm_model_9000,
        )

    def _generate_qwen3_vllm_9001(self, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        return self._generate_vllm(
            prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            temperature=temperature,
            endpoint=self.qwen3_vllm_endpoint_9001,
            model=self.qwen3_vllm_model_9001,
        )

    def _generate_qwen3_vl_vllm_9002(self, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        return self._generate_vllm(
            prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            temperature=temperature,
            endpoint=self.qwen3_vllm_endpoint_9002,
            model=self.qwen3_vl_vllm_model_9002,
        )

    def _generate_gemini(self, prompt, system_prompt=None, image_paths=None, endpoint=None):
        data = [{"text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                if isinstance(image_path, str):
                    with open(image_path, "rb") as image_file:
                        d = base64.b64encode(image_file.read()).decode("utf-8")
                elif isinstance(image_path, Image.Image):
                    buffered = BytesIO()
                    image_path.save(buffered, format="JPEG")
                    d = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data.append({"inlineData": {"mimeType": "image/jpeg", "data": f"{d}"}})
        message = {}
        if system_prompt is not None:
            data_system = {"parts": [{"text": system_prompt}]}
            message["system_instruction"] = data_system
        message["contents"] = [{"role": "user", "parts": data}]
        message["generation_config"] = {"temperature": 0.0, "max_output_tokens": 5000, "presence_penalty": 1.0}

        generated_text = []
        indices = []

        for _ in range(1):
            response = self._execute(message, 0, self.api_key, endpoint)
            generated_text.append(response["generated_text"])
        output = generated_text[0]

        return output
    
    def _generate_gpt4(self, prompt, system_prompt, image_paths=None, temperature=0.0, endpoint=None):
        system_data = [{"type": "text", "text": system_prompt}]
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                if isinstance(image_path, str):
                    with open(image_path, "rb") as image_file:
                        d = base64.b64encode(image_file.read()).decode("utf-8")
                elif isinstance(image_path, Image.Image):
                    buffered = BytesIO()
                    image_path.save(buffered, format="JPEG")
                    d = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}})
        message = [{"role": "system", "content": system_data}, {"role": "user", "content": data}]
        body = {
            "messages": message,
            "max_tokens": 200,
            "presence_penalty": 1.0,
            "temperature": temperature
        }
        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "60"))
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=message,
                    temperature=temperature,
                    max_tokens=1000,
                    presence_penalty=1.0,
                )
                output = response.choices[0].message.content
                return output
            except Exception as exc:
                attempt += 1
                if not self._is_rate_limit_error(exc):
                    raise
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"GPT-4 rate limit did not clear after {max_attempts} retries."
                    ) from exc
                self._wait_for_rate_limit(exc, "GPT-4", attempt)
 
    def _generate_gpt5(self, prompt, image_paths=None, temperature=0.0, endpoint=None):
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            if len(image_paths) > 50: image_paths = image_paths[:50]
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    d = base64.b64encode(image_file.read()).decode("utf-8")
                data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}})
        if self.system_prompt_enable:
            message = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": data}]
        else:
            message = [{"role": "user", "content": data}]
        body = {
            "messages": message,
            "max_completion_tokens": 4000
        }

        generated_text = []
        indices = []

        for _ in range(1):
            response = self._execute(body, 0, self.api_key, endpoint)
            generated_text.append(response["generated_text"])
        output = generated_text[0]

        print(output)

        return output

    def _generate_openai_chat(self, prompt, system_prompt=None, image_paths=None, temperature=0.0, model=None):
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            if len(image_paths) > 50:
                image_paths = image_paths[:50]
            for image_path in image_paths:
                if isinstance(image_path, str):
                    with open(image_path, "rb") as image_file:
                        d = base64.b64encode(image_file.read()).decode("utf-8")
                elif isinstance(image_path, Image.Image):
                    buffered = BytesIO()
                    image_path.save(buffered, format="JPEG")
                    d = base64.b64encode(buffered.getvalue()).decode("utf-8")
                else:
                    continue
                data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": data})

        request_kwargs = {
            "model": model or self.gpt51_model,
            "messages": messages,
            "max_completion_tokens": 4000,
        }
        reasoning_effort = _openai_reasoning_effort_for_model(request_kwargs["model"])
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
        if temperature and temperature > 0:
            request_kwargs["temperature"] = temperature

        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "60"))
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                return _strip_model_thinking_output(response.choices[0].message.content or "")
            except Exception as exc:
                attempt += 1
                if not self._is_rate_limit_error(exc):
                    raise
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"{request_kwargs['model']} rate limit did not clear after {max_attempts} retries."
                    ) from exc
                self._wait_for_rate_limit(exc, request_kwargs["model"], attempt)

    def _generate_claude(self, prompt, image_paths=None, temperature=0.0, endpoint=None):
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    d = base64.b64encode(image_file.read()).decode("utf-8")
                data.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": d}})
        
        mess = [{"role": "user", "content": data}]

        max_attempts = int(os.environ.get("LLM_RATE_LIMIT_MAX_RETRIES", "60"))
        attempt = 0
        while True:
            try:
                message = anthropic.Anthropic().messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4000,
                    messages=mess,
                    temperature=0.0,
                ).content[0].text
                return message
            except Exception as exc:
                attempt += 1
                if not self._is_rate_limit_error(exc):
                    raise
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Claude rate limit did not clear after {max_attempts} retries."
                    ) from exc
                self._wait_for_rate_limit(exc, "Claude", attempt)
    
    def _generate_cohere(self, prompt, image_paths=None):
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    d = base64.b64encode(image_file.read()).decode("utf-8")
                data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}})

        count = 0
        response = None

        while count < 5:
            try:
                response = self.co.chat(
                    model="command-a-vision-07-2025",
                    messages=[
                        {
                            "role": "user",
                            "content": data
                        }
                    ],
                    temperature=0.0,
                )
                break
            except Exception as e:
                print(e)
                count += 1

        if response is not None:
            generated_text = response.message.content[0].text
        else:
            generated_text = None

        return generated_text

    def __call__(self, prompt, system_prompt, image_paths=None, temperature=0.0):
        if "/" in self.method:
            if self.multimodal:
                generated_text = self._generate_qwen_vl(prompt, image_paths=image_paths)
            elif self.ollama:
                generated_text = self._generate_ollama(prompt)
            else:
                generated_text = self._generate_local_llm(prompt)
        elif self.method == "gpt4":
            generated_text = self._generate_gpt4(prompt, system_prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_endpoint)
        elif self.method == "gpt4-mini":
            generated_text = self._generate_gpt4(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_mini_endpoint)
        elif self.method == "gpt5":
            generated_text = self._generate_gpt5(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt5_endpoint)
        elif self.method == "gpt5-mini":
            generated_text = self._generate_gpt5(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt5_mini_endpoint)
        elif self.method in {"gpt5.1", "gpt-5.1"}:
            generated_text = self._generate_openai_chat(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature, model=self.gpt51_model)
        elif self.method == "qwen3-vllm-9000":
            generated_text = self._generate_qwen3_vllm_9000(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature)
        elif self.method == "qwen3-vllm-9001":
            generated_text = self._generate_qwen3_vllm_9001(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature)
        elif self.method == "qwen3-vl-vllm-9002":
            generated_text = self._generate_qwen3_vl_vllm_9002(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature)
        elif self.method == "vllm":
            generated_text = self._generate_vllm(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature, endpoint=self.vllm_endpoint, model=self.vllm_model)
        elif self.method == "gemini":
            generated_text = self._generate_gemini(prompt, system_prompt, image_paths=image_paths, endpoint=self.gemini_endpoint)
        elif self.method == "gemini-pro":
            generated_text = self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_pro_endpoint)
        elif self.method == "claude":
            generated_text = self._generate_claude(prompt, image_paths=image_paths)
        elif self.method == "cohere":
            generated_text = self._generate_cohere(prompt, image_paths=image_paths)

        return generated_text
    
    def generate_format(self, prompt, system_prompt, image_paths=None, temperature=0.0, format="code", schema=None):
        if "/" in self.method:
            if self.multimodal:
                generated_text = self._generate_qwen_vl(prompt, image_paths=image_paths)
            elif self.ollama:
                generated_text = self._generate_ollama(prompt)
            else:
                generated_text = self._generate_local_llm(prompt)
        elif self.method == "gpt4":
            generated_text = self._generate_gpt4(prompt, system_prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_endpoint)
        elif self.method == "gpt4-mini":
            generated_text = self._generate_gpt4(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_mini_endpoint)
        elif self.method == "gpt5":
            generated_text = self._generate_gpt5(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt5_endpoint)
        elif self.method == "gpt5-mini":
            generated_text = self._generate_gpt5(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt5_mini_endpoint)
        elif self.method in {"gpt5.1", "gpt-5.1"}:
            generated_text = self._generate_openai_chat(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature, model=self.gpt51_model)
        elif self.method == "qwen3-vllm-9000":
            generated_text = self._generate_qwen3_vllm_9000(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature)
        elif self.method == "qwen3-vllm-9001":
            generated_text = self._generate_qwen3_vllm_9001(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature)
        elif self.method == "qwen3-vl-vllm-9002":
            generated_text = self._generate_qwen3_vl_vllm_9002(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature)
        elif self.method == "vllm":
            generated_text = self._generate_vllm(prompt, system_prompt=system_prompt, image_paths=image_paths, temperature=temperature, endpoint=self.vllm_endpoint, model=self.vllm_model)
        elif self.method == "gemini":
            generated_text = self._generate_gemini(prompt, system_prompt, image_paths=image_paths, endpoint=self.gemini_endpoint)
        elif self.method == "gemini-pro":
            generated_text = self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_pro_endpoint)
        elif self.method == "claude":
            generated_text = self._generate_claude(prompt, image_paths=image_paths)
        elif self.method == "cohere":
            generated_text = self._generate_cohere(prompt, image_paths=image_paths)

        if generated_text is None: return generated_text
        if generated_text.startswith("-") or generated_text == "OK":
            return generated_text
        
        if format == "code":
            generated_text = self._wrap_code(generated_text)
        elif format == "json":
            generated_text = self._wrap_json(generated_text, schema=schema)

        return generated_text
    
    def ensemble(self, prompt, image_paths=None):
        results = {}
        results["gpt4"] = self._wrap_json(self._generate_gpt4(prompt, image_paths=image_paths, endpoint=self.gpt4_endpoint))
        results["gpt4-mini"] = self._wrap_json(self._generate_gpt4(prompt, image_paths=image_paths, endpoint=self.gpt4_mini_endpoint))
        results["gemini"] = self._wrap_json(self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_endpoint))
        results["gemini-pro"] = self._wrap_json(self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_pro_endpoint))

        return results


class ClouseSourcedLLM:
    """Route hosted LLM calls across GPT-5, Claude, and Gemini on rate limits."""

    DEFAULT_METHODS = ("gpt5", "claude", "gemini")

    def __init__(self, methods=None, max_rounds=None):
        self.methods = tuple(methods or self.DEFAULT_METHODS)
        if not self.methods:
            raise ValueError("ClouseSourcedLLM requires at least one provider method.")
        unsupported = [method for method in self.methods if method not in self.DEFAULT_METHODS]
        if unsupported:
            raise ValueError(f"Unsupported cloud-sourced LLM methods: {unsupported}")
        self.max_rounds = int(max_rounds or os.environ.get("CLOUD_SOURCED_LLM_MAX_ROUNDS", "3"))
        self.backends = {method: LLM(method) for method in self.methods}
        self.next_method_index = 0
        self.last_method = None

    @staticmethod
    def _is_rate_limit_error(exc) -> bool:
        return LLM._is_rate_limit_error(exc)

    @staticmethod
    def _image_to_base64(image_path):
        if isinstance(image_path, str):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        if isinstance(image_path, Image.Image):
            buffered = BytesIO()
            image_path.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        return None

    def _ordered_methods(self):
        return [
            self.methods[(self.next_method_index + offset) % len(self.methods)]
            for offset in range(len(self.methods))
        ]

    def _mark_success(self, method):
        self.last_method = method
        self.next_method_index = self.methods.index(method)

    def _mark_rate_limited(self, method):
        self.next_method_index = (self.methods.index(method) + 1) % len(self.methods)

    def _call_gpt5_once(self, backend, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths[:50]:
                encoded = self._image_to_base64(image_path)
                if encoded:
                    data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": data})
        body = {
            "messages": messages,
            "max_completion_tokens": 4000,
        }
        headers = {
            "Content-type": "application/json",
            "api-key": backend.api_key,
        }
        response = requests.post(backend.gpt5_endpoint, headers=headers, json=body, timeout=300)
        response.raise_for_status()
        payload = response.json()
        generated_text = backend._extract_response_text(payload)
        if generated_text is None:
            raise RuntimeError(f"GPT-5 endpoint returned no text: {backend._summarize_empty_response(payload)}")
        return generated_text

    def _call_claude_once(self, backend, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        content = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                encoded = self._image_to_base64(image_path)
                if encoded:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        }
                    )
        kwargs = {
            "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        return anthropic.Anthropic().messages.create(**kwargs).content[0].text

    def _call_gemini_once(self, backend, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        parts = [{"text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                encoded = self._image_to_base64(image_path)
                if encoded:
                    parts.append({"inlineData": {"mimeType": "image/jpeg", "data": encoded}})
        body = {"contents": [{"role": "user", "parts": parts}]}
        if system_prompt:
            body["system_instruction"] = {"parts": [{"text": system_prompt}]}
        body["generation_config"] = {
            "temperature": temperature,
            "max_output_tokens": 5000,
            "presence_penalty": 1.0,
        }
        headers = {
            "Content-type": "application/json",
            "api-key": backend.api_key,
        }
        response = requests.post(backend.gemini_endpoint, headers=headers, json=body, timeout=300)
        response.raise_for_status()
        payload = response.json()
        generated_text = backend._extract_response_text(payload)
        if generated_text is None:
            raise RuntimeError(f"Gemini endpoint returned no text: {backend._summarize_empty_response(payload)}")
        return generated_text

    def _call_method_once(self, method, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        backend = self.backends[method]
        if method == "gpt5":
            return self._call_gpt5_once(
                backend,
                prompt,
                system_prompt=system_prompt,
                image_paths=image_paths,
                temperature=temperature,
            )
        if method == "claude":
            return self._call_claude_once(
                backend,
                prompt,
                system_prompt=system_prompt,
                image_paths=image_paths,
                temperature=temperature,
            )
        if method == "gemini":
            return self._call_gemini_once(
                backend,
                prompt,
                system_prompt=system_prompt,
                image_paths=image_paths,
                temperature=temperature,
            )
        raise ValueError(f"Unsupported cloud-sourced LLM method: {method}")

    def __call__(self, prompt, system_prompt=None, image_paths=None, temperature=0.0):
        last_rate_limit_error = None
        max_attempts = max(1, self.max_rounds) * len(self.methods)
        for _ in range(max_attempts):
            for method in self._ordered_methods():
                try:
                    output = self._call_method_once(
                        method,
                        prompt,
                        system_prompt=system_prompt,
                        image_paths=image_paths,
                        temperature=temperature,
                    )
                    self._mark_success(method)
                    return output
                except Exception as exc:
                    if not self._is_rate_limit_error(exc):
                        raise
                    last_rate_limit_error = exc
                    print(f"{method} rate limit reached; switching provider.")
                    self._mark_rate_limited(method)
                    break
        raise RuntimeError(
            f"All cloud-sourced LLM providers rate-limited after {max_attempts} attempts."
        ) from last_rate_limit_error

    def generate_format(self, prompt, system_prompt, image_paths=None, temperature=0.0, format="code", schema=None):
        generated_text = self(
            prompt,
            system_prompt=system_prompt,
            image_paths=image_paths,
            temperature=temperature,
        )
        if generated_text is None:
            return generated_text
        if generated_text.startswith("-") or generated_text == "OK":
            return generated_text
        if format == "code":
            return LLM._wrap_code(generated_text)
        if format == "json":
            return LLM._wrap_json(generated_text, schema=schema)
        return generated_text


CloudSourcedLLM = ClouseSourcedLLM
ClousedSourceLLM = ClouseSourcedLLM


class TextEncoder:
    def __init__(self, device='cpu'):
        self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

    def __call__(self, text):
        with torch.no_grad():
            outputs = self.model.encode([text])
        emb = outputs[0].tolist()

        return emb
