import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image


CONFIG_PATH = Path(__file__).with_name("rh_models.json")


def _load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = _load_config()
DEFAULT_API_BASE = CONFIG.get("plugin", {}).get("default_api_base", "http://124.221.138.114:8001")
IMAGE_MODEL_NAME_TO_ID = {item.get("display_name") or item["id"]: item["id"] for item in CONFIG.get("image_models", [])}
VISION_MODEL_NAME_TO_ID = {item.get("display_name") or item["id"]: item["id"] for item in CONFIG.get("vision_models", [])}
MODEL_ID_TO_DISPLAY = {
    **{model_id: display for display, model_id in IMAGE_MODEL_NAME_TO_ID.items()},
    **{model_id: display for display, model_id in VISION_MODEL_NAME_TO_ID.items()},
}
IMAGE_MODELS = list(IMAGE_MODEL_NAME_TO_ID.keys())
VISION_MODELS = list(VISION_MODEL_NAME_TO_ID.keys())
IMAGE_SPECS = CONFIG.get("image_specs", ["1K", "2K", "4K"])
ASPECT_RATIOS = CONFIG.get("aspect_ratios", ["智能比例", "1:1", "4:3", "3:4", "16:9", "9:16"])
IMAGE_QUALITIES = CONFIG.get("image_qualities", ["低画质", "标准画质", "高画质"])
RUNTIME_PROFILES = [item["id"] for item in CONFIG.get("runtime_profiles", [])] or ["Lite", "Standard", "Plus"]

TEMPLATE_OPTIONS = ["产品视觉", "高级画册", "固定场景模特换装", "人像摄影视觉", "人像视觉系列衍生"]
MODE_OPTIONS = ["统一视觉战役", "多维创意宇宙"]
OUTPUT_LANGUAGES = ["中文", "英文"]
REFERENCE_MODES = ["完整参考", "主体参考", "风格参考", "构图参考", "局部重绘"]
STYLE_OPTIONS = [
    "高级极简商业摄影",
    "自然光生活化实拍",
    "高级画册留白排版",
    "VOGUE 杂志大片",
    "高端电商详情页",
    "电影感户外人像",
    "奢侈品棚拍质感",
    "未来感玻璃拟态",
    "小红书高级生活方式",
    "清透 3D 产品渲染",
]


def _model_id(display_name, model_map):
    return model_map.get(display_name, display_name)


def _model_display(model_id):
    return MODEL_ID_TO_DISPLAY.get(model_id, model_id)


def _api_base(api_base):
    base = (api_base or DEFAULT_API_BASE).strip().rstrip("/")
    if not base:
        return DEFAULT_API_BASE
    if base.startswith(("sk_", "sk-", "sk.")):
        return DEFAULT_API_BASE
    return base


def _headers(api_token):
    token = (api_token or "").strip()
    if not token:
        raise RuntimeError("请填写你发给买家的 Jojoagent API 口令")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _request(method, url, **kwargs):
    session = requests.Session()
    session.trust_env = False
    return session.request(method, url, **kwargs)


def _tensor_to_pil(image):
    if len(image.shape) > 3:
        image = image[0]
    arr = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _tensor_to_base64(image, max_side=1600, quality=92):
    img = _tensor_to_pil(image)
    width, height = img.size
    longest = max(width, height)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image(image_base64):
    raw = base64.b64decode(str(image_base64).split(",", 1)[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _split_prompts(text):
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict):
            for key in ("prompt_list", "prompts", "screens"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass
    quoted_screens = re.findall(r'"([^"]*第\s*\d+\s*屏提示词[：:].*?)"', raw, flags=re.DOTALL)
    if quoted_screens:
        return [item.strip() for item in quoted_screens if item.strip()]
    screen_matches = list(re.finditer(r"第\s*\d+\s*屏提示词[：:]", raw))
    if len(screen_matches) > 1:
        prompts = []
        for index, match in enumerate(screen_matches):
            end = screen_matches[index + 1].start() if index + 1 < len(screen_matches) else len(raw)
            block = raw[match.start():end].strip().strip('"')
            if block:
                prompts.append(block)
        return prompts
    parts = [item.strip() for item in re.split(r"\n+|---+", raw) if item.strip()]
    if len(parts) > 1:
        return parts
    numbered = re.split(r"(?:^|\n|\s)(?:第\s*\d+\s*[屏幕张幅]|Screen\s*\d+|\d+\s*[.、:：])", raw, flags=re.IGNORECASE)
    numbered = [item.strip(" \n\t:：。") for item in numbered if item.strip(" \n\t:：。")]
    return numbered or [raw]


def _images_from_optional(*images, limit=8):
    result = []
    for image in images[:limit]:
        if image is not None:
            result.append(_tensor_to_base64(image))
    return result


def _image_ratio_from_optional(*images):
    for image in images:
        if image is None:
            continue
        try:
            shape = list(image.shape)
            if len(shape) >= 3:
                return float(shape[-2]) / max(float(shape[-3]), 1.0)
        except Exception:
            pass
    return 1.0


def _closest_aspect_ratio(width_to_height):
    candidates = [item for item in ASPECT_RATIOS if re.match(r"^\d+:\d+$", str(item))]
    if not candidates:
        candidates = ["1:1", "4:3", "3:4", "16:9", "9:16"]

    def score(label):
        w, h = [float(part) for part in str(label).split(":", 1)]
        return abs((w / h) - width_to_height)

    return min(candidates, key=score)


def _normalize_aspect_ratio(aspect_ratio, *images):
    value = str(aspect_ratio or "").strip()
    if value in {"智能比例", "自动", "自适应", "auto", "Auto"}:
        return _closest_aspect_ratio(_image_ratio_from_optional(*images))
    return value or "1:1"


def _coerce_prompt_list(data, expected_count):
    expected_count = max(1, int(expected_count or 1))
    prompt_list = data.get("prompt_list") if isinstance(data, dict) else None
    if not isinstance(prompt_list, list):
        prompt_list = _split_prompts(data.get("optimized_prompt", "") if isinstance(data, dict) else data)
    prompt_list = [str(item).strip() for item in prompt_list if str(item).strip()]
    if len(prompt_list) >= expected_count:
        return prompt_list[:expected_count]
    base = prompt_list[0] if prompt_list else str(data.get("optimized_prompt", "") if isinstance(data, dict) else "").strip()
    if not base:
        base = "高质量商业视觉图像，保持主体一致，画面精致完整。"
    while len(prompt_list) < expected_count:
        index = len(prompt_list) + 1
        prompt_list.append(f"第{index}屏：{base} 作为连续系列的第{index}个画面，保持主体、品牌调性和视觉风格一致，但使用不同构图、景别和画面重点。")
    return prompt_list


def _prompt_list_text(prompt_list):
    blocks = []
    for index, item in enumerate(prompt_list, 1):
        text = str(item).strip().strip('"')
        text = re.sub(r"^第\s*\d+\s*屏提示词[：:]\s*", "", text).strip()
        if text:
            blocks.append(f'"第{index}屏提示词：{text}"')
    return "\n\n".join(blocks)


def _post_json(api_base, api_token, path, payload, timeout=600):
    res = _request("POST", f"{_api_base(api_base)}{path}", headers=_headers(api_token), json=payload, timeout=timeout)
    if res.status_code != 200:
        raise RuntimeError(f"请求失败: HTTP {res.status_code} {res.text[:1000]}")
    body = res.json()
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


def _build_optimizer_instruction(template, mode, screens, brand_name, base_instruction, output_language):
    brand = "" if not brand_name or brand_name == "选填" else str(brand_name).strip()
    return f"""
你是 Jojoagent RH V2 提示词优化器。
模板：{template}
模式：{mode}
屏次：{screens}
品牌名：{brand or "不强制出现品牌文字"}
输出语言：{output_language}
基础指令：{base_instruction or "根据参考图生成高质量、可直接用于图像生成的提示词。"}

要求：
1. 输出适合商业生产的完整图像提示词。
2. 如果屏次大于 1，请生成连续系列，不要重复同一个构图。
3. 保留参考图中重要主体、材质、颜色、比例、风格和场景信息。
4. 不要输出解释过程，不要输出 Markdown。
5. 返回 JSON，字段包含 optimized_prompt、prompt_list、image_analysis、negative_prompt、model_notes。
6. prompt_list 必须是长度严格等于 {int(screens)} 的字符串数组；每一项对应一屏完整生图提示词，不能只给 1 条总提示。
""".strip()


def _build_ecommerce_instruction(product_name, selling_points, style, screens, output_language):
    return f"""
你是 Jojoagent RH V2 电商详情页提示词总监。
商品名称：{product_name}
核心卖点/材质细节/使用场景：{selling_points}
视觉风格：{style}
生成屏数：{screens}
输出语言：{output_language}

请结合正面图、背面图、侧面图、风格参考图，生成 {screens} 条详情页连续分屏提示词。
每一屏都要包含：主文案、副文案、版式与字体、画面主体构图、画质质感细节。
只返回 JSON 字符串数组，数组长度必须等于生成屏数。
""".strip()


class JojoV2CostDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "刷新触发": ("*", {}),
                "api_base": ("STRING", {"default": DEFAULT_API_BASE}),
                "api_token": ("STRING", {"default": ""}),
                "显示条数": ("INT", {"default": 10, "min": 1, "max": 100}),
                "只看成功扣费": (["否", "是"], {"default": "否"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("扣费详情", "合计扣除", "记录数量")
    FUNCTION = "run"
    CATEGORY = "Jojoagent V2/RH"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(self, 刷新触发, api_base, api_token, 显示条数, 只看成功扣费):
        headers = _headers(api_token)
        usage_url = f"{_api_base(api_base)}/api/usage?limit={int(显示条数)}"
        usage_res = _request("GET", usage_url, headers=headers, timeout=60)
        if usage_res.status_code != 200:
            raise RuntimeError(f"扣费详情获取失败: HTTP {usage_res.status_code} {usage_res.text[:800]}")
        rows = usage_res.json().get("data", [])

        balance_text = "未知"
        balance_res = _request("GET", f"{_api_base(api_base)}/api/balance", headers=headers, timeout=60)
        if balance_res.status_code == 200:
            balance_data = balance_res.json().get("data", {})
            if "balance" in balance_data:
                balance_text = str(round(float(balance_data.get("balance") or 0), 4))

        if 只看成功扣费 == "是":
            rows = [row for row in rows if row.get("status") == "success"]
        total = round(sum(float(row.get("cost") or 0) for row in rows), 4)
        lines = ["Jojo RH V2 扣费", f"当前余额：{balance_text}", f"本次显示合计扣除：{total}", ""]
        for row in rows:
            model = _model_display(str(row.get("model", "")))
            cost = round(float(row.get("cost") or 0), 4)
            lines.append(f"- 模型：{model} | 扣除：{cost}")
        return ("\n".join(lines), total, len(rows))


class JojoV2ReversePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": ("STRING", {"default": DEFAULT_API_BASE}),
                "api_token": ("STRING", {"default": ""}),
                "识图模型": (VISION_MODELS, {"default": VISION_MODELS[0]}),
                "分析指令": ("STRING", {"multiline": True, "default": "分析图片并反推适合生图的高质量提示词"}),
                "参考模式": (REFERENCE_MODES, {"default": "完整参考"}),
                "目标画幅": (["自动"] + ASPECT_RATIOS, {"default": "自动"}),
                "输出语言": (OUTPUT_LANGUAGES, {"default": "中文"}),
                "创造性": ("FLOAT", {"default": 0.35, "min": 0, "max": 2, "step": 0.05}),
                "最大输出": ("INT", {"default": 4096, "min": 256, "max": 32768}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "image_analysis", "negative_prompt", "run_info")
    FUNCTION = "run"
    CATEGORY = "Jojoagent V2/RH"

    def run(self, api_base, api_token, 识图模型, 分析指令, 参考模式, 目标画幅, 输出语言, 创造性, 最大输出, **kwargs):
        images = _images_from_optional(*(kwargs.get(f"image{i}") for i in range(1, 9)), limit=8)
        model_id = _model_id(识图模型, VISION_MODEL_NAME_TO_ID)
        payload = {
            "model": model_id,
            "images": images,
            "instruction": 分析指令,
            "reference_mode": 参考模式,
            "target_ratio": 目标画幅,
            "output_language": 输出语言,
            "temperature": float(创造性),
            "max_tokens": int(最大输出),
        }
        data = _post_json(api_base, api_token, "/api/rh/reverse-prompt", payload)
        analysis = data.get("image_analysis", [])
        if isinstance(analysis, list):
            analysis = "\n".join(str(item) for item in analysis)
        run_info = f"模型：{识图模型}\n实际调用：{data.get('model', model_id)}\n扣除：{data.get('cost')}\n请求ID：{data.get('request_id')}"
        return (str(data.get("optimized_prompt", "")), str(analysis), str(data.get("negative_prompt", "")), run_info)


class JojoV2ImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": ("STRING", {"default": DEFAULT_API_BASE}),
                "api_token": ("STRING", {"default": ""}),
                "批量模式": (["多提示词生图", "单提示词多次"], {"default": "多提示词生图"}),
                "模型": (IMAGE_MODELS, {"default": IMAGE_MODELS[0]}),
                "提示词列表": ("STRING", {"multiline": True, "default": ""}),
                "单提示词": ("STRING", {"multiline": True, "default": ""}),
                "生成次数": ("INT", {"default": 1, "min": 1, "max": 100}),
                "图像规格": (IMAGE_SPECS, {"default": "1K"}),
                "图片比例": (ASPECT_RATIOS, {"default": "1:1"}),
                "图像质量": (IMAGE_QUALITIES, {"default": "标准画质"}),
                "运行机型": (RUNTIME_PROFILES, {"default": RUNTIME_PROFILES[0]}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "参考图4": ("IMAGE",),
                "参考图5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "batch_info")
    FUNCTION = "generate_batch"
    CATEGORY = "Jojoagent V2/RH"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate_batch(self, api_base, api_token, 批量模式, 模型, 提示词列表, 单提示词, 生成次数, 图像规格, 图片比例, 图像质量, 运行机型, 随机种子, **kwargs):
        if 批量模式 == "多提示词生图":
            prompts = _split_prompts(提示词列表)
        else:
            prompts = [单提示词.strip() or "高质量商业视觉图像"] * int(生成次数)
        if not prompts:
            prompts = [单提示词.strip() or "高质量商业视觉图像"]
        ref_tensors = [kwargs.get(f"参考图{i}") for i in range(1, 6)]
        refs = _images_from_optional(*ref_tensors, limit=5)
        normalized_ratio = _normalize_aspect_ratio(图片比例, *ref_tensors)
        tensors = []
        lines = []
        total = 0.0
        for index, prompt in enumerate(prompts, 1):
            seed = int(随机种子)
            current_seed = seed + index - 1 if seed >= 0 else -1
            model_id = _model_id(模型, IMAGE_MODEL_NAME_TO_ID)
            payload = {
                "model": model_id,
                "prompt": prompt,
                "images": refs,
                "output_size": 图像规格,
                "aspect_ratio": normalized_ratio,
                "image_quality": 图像质量,
                "runtime_profile": 运行机型,
                "seed": current_seed,
            }
            data = _post_json(api_base, api_token, "/api/rh/generate-image", payload, timeout=700)
            tensors.append(_decode_image(data["image_base64"]))
            cost = float(data.get("cost") or 0)
            total += cost
            lines.append(f"{index}. {模型} -> {data.get('model', model_id)} | {图片比例}->{normalized_ratio} | 扣除 {cost} | {data.get('message', '')}")
        try:
            image_batch = torch.cat(tensors, dim=0)
        except Exception:
            image_batch = tensors[0]
            lines.append("提示：返回图片尺寸不一致，仅输出第一张图片。")
        info = "Jojo RH V2 批量图像生成\n" + "\n".join(lines) + f"\n合计扣除：{round(total, 4)}"
        return (image_batch, info)


class JojoV2PromptOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模板": (TEMPLATE_OPTIONS, {"default": "产品视觉"}),
                "模式": (MODE_OPTIONS, {"default": "统一视觉战役"}),
                "屏次": ("INT", {"default": 1, "min": 1, "max": 100}),
                "品牌名": ("STRING", {"default": "选填"}),
                "输出语言": (OUTPUT_LANGUAGES, {"default": "中文"}),
                "识图模型": (VISION_MODELS, {"default": VISION_MODELS[0]}),
                "api_base": ("STRING", {"default": DEFAULT_API_BASE}),
                "api_token": ("STRING", {"default": ""}),
                "最大输出": ("INT", {"default": 4096, "min": 256, "max": 32768}),
                "基础指令": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "图像1": ("IMAGE",),
                "图像2": ("IMAGE",),
                "图像3": ("IMAGE",),
                "图像4": ("IMAGE",),
                "图像5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("optimized_prompt", "prompts_list", "prompts_count")
    FUNCTION = "run"
    CATEGORY = "Jojoagent V2/RH"

    def run(self, 模板, 模式, 屏次, 品牌名, 输出语言, 识图模型, api_base, api_token, 最大输出, 基础指令, **kwargs):
        images = _images_from_optional(*(kwargs.get(f"图像{i}") for i in range(1, 6)), limit=5)
        instruction = _build_optimizer_instruction(模板, 模式, 屏次, 品牌名, 基础指令, 输出语言)
        model_id = _model_id(识图模型, VISION_MODEL_NAME_TO_ID)
        payload = {
            "model": model_id,
            "images": images,
            "instruction": instruction,
            "reference_mode": "提示词优化器",
            "target_ratio": "自动",
            "output_language": 输出语言,
            "temperature": 0.35,
            "max_tokens": int(最大输出),
        }
        data = _post_json(api_base, api_token, "/api/rh/reverse-prompt", payload)
        prompt_list = _coerce_prompt_list(data, 屏次)
        return (str(data.get("optimized_prompt", "")), _prompt_list_text(prompt_list), len(prompt_list))


class JojoV2EcommerceDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "商品名称": ("STRING", {"default": "商品"}),
                "核心卖点": ("STRING", {"multiline": True, "default": "核心卖点\n材质细节\n使用场景"}),
                "视觉风格": (STYLE_OPTIONS, {"default": "高级极简商业摄影"}),
                "生成屏数": ("INT", {"default": 3, "min": 1, "max": 100}),
                "输出语言": (OUTPUT_LANGUAGES, {"default": "中文"}),
                "识图模型": (VISION_MODELS, {"default": VISION_MODELS[0]}),
                "api_base": ("STRING", {"default": DEFAULT_API_BASE}),
                "api_token": ("STRING", {"default": ""}),
                "最大输出": ("INT", {"default": 8192, "min": 256, "max": 32768}),
            },
            "optional": {
                "正面图": ("IMAGE",),
                "背面图": ("IMAGE",),
                "侧面图": ("IMAGE",),
                "风格参考1": ("IMAGE",),
                "风格参考2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts_list", "prompts_count")
    FUNCTION = "run"
    CATEGORY = "Jojoagent V2/RH"

    def run(self, 商品名称, 核心卖点, 视觉风格, 生成屏数, 输出语言, 识图模型, api_base, api_token, 最大输出, **kwargs):
        images = _images_from_optional(kwargs.get("正面图"), kwargs.get("背面图"), kwargs.get("侧面图"), kwargs.get("风格参考1"), kwargs.get("风格参考2"), limit=5)
        instruction = _build_ecommerce_instruction(商品名称, 核心卖点, 视觉风格, 生成屏数, 输出语言)
        model_id = _model_id(识图模型, VISION_MODEL_NAME_TO_ID)
        payload = {
            "model": model_id,
            "images": images,
            "instruction": instruction,
            "reference_mode": "电商详情页",
            "target_ratio": "自动",
            "output_language": 输出语言,
            "temperature": 0.35,
            "max_tokens": int(最大输出),
        }
        data = _post_json(api_base, api_token, "/api/rh/reverse-prompt", payload)
        prompt_list = _coerce_prompt_list(data, 生成屏数)
        return (_prompt_list_text(prompt_list), len(prompt_list))


NODE_CLASS_MAPPINGS = {
    "JojoV2CostDetail": JojoV2CostDetail,
    "JojoV2ReversePrompt": JojoV2ReversePrompt,
    "JojoV2ImageBatch": JojoV2ImageBatch,
    "JojoV2PromptOptimizer": JojoV2PromptOptimizer,
    "JojoV2EcommerceDetail": JojoV2EcommerceDetail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JojoV2CostDetail": "Jojo V2 扣费详情",
    "JojoV2ReversePrompt": "Jojo V2 识图提示词反推",
    "JojoV2ImageBatch": "Jojo V2 AI 批量图像生成",
    "JojoV2PromptOptimizer": "Jojo V2 提示词优化器",
    "JojoV2EcommerceDetail": "Jojo V2 电商详情页提示词",
}
