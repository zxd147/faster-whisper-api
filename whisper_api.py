import asyncio
import base64
import gc
import json
import logging
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from io import BytesIO
from typing import Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from faster_whisper import WhisperModel
from utils.log_utils import logger


def torch_gc():
    """释放内存"""
    # 垃圾回收操作执行垃圾回收和 CUDA 缓存清空
    # Prior inference run might have large variables not cleaned up due to exception during the run.
    # Free up as much memory as possible to allow this run to be successful.
    gc.collect()
    if torch.cuda.is_available():  # 检查是否可用CUDA
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集CUDA内存碎片


def init_app():
    if torch.cuda.is_available():
        log = f'本次加载模型的设备为GPU: {torch.cuda.get_device_name(0)}'
    else:
        log = '本次加载模型的设备为CPU.'
    whisper_logger.info(log)
    log = f"Service started!"
    whisper_logger.info(log)


def configure_logging():
    log_file = 'logs/api.log'
    logger = logging.getLogger('whisper')
    logger.setLevel(logging.INFO)
    handel_format = '%(asctime)s - %(levelname)s - %(message)s'
    # 设置 propagate 为 False
    # propagate 用于控制日志消息的传播行为，如果设置为 True（默认值），那么该 logger 记录的消息会向上层的 logger 传播，导致记录两次日志。
    logger.propagate = False
    # 移除现有的处理器（防止重复）
    if logger.hasHandlers():
        logger.handlers.clear()
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter(handel_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    return logger


async def run_in_threadpool(func, audio, language, task, beam_size, initial_prompt, vad_parameters,
                            vad_filter=True):
    loop = asyncio.get_event_loop()
    # Use partial to create a callable that includes combined_args
    partial_func = partial(func, audio, language, task, beam_size, initial_prompt=initial_prompt,
                           vad_filter=vad_filter, vad_parameters=vad_parameters)
    result = await loop.run_in_executor(executor, partial_func)
    return result


async def get_audio(audio_file, request_data):
    audio_info = None
    audio_contents = None
    content_type = request_data['content_type']
    audio_format = request_data['audio_format']
    audio_path = request_data['audio_path']
    audio_base64 = request_data['audio_base64']
    # 检查是否至少提供了一个字段
    if not (audio_file or audio_path or audio_base64):
        raise ValueError("ERROR: No file, file path, base64 audio or text provided")
    if audio_file:
        # audio_contents = await audio_contents.read()
        # size = len(audio_contents)
        audio_contents = audio_file.file  # 直接使用 `UploadFile` 的文件对象
        content_type = audio_file.content_type or content_type
        audio_contents.seek(0, 2)  # 移动到文件末尾
        size = audio_contents.tell()  # 获取当前位置，即文件大小
        audio_contents.seek(0)  # 将指针重置到文件开头
        audio_info = f"Audio type: file, Format: {content_type}, Size: {size} bytes."
    elif audio_path:
        # 从文件路径读取文件内容
        if os.path.exists(audio_path):
            # with open(audio_path, "rb") as f:
            #     audio_contents = f.read()  # 读取文件内容
            audio_contents = audio_path
            audio_extension = os.path.splitext(audio_path)[1] or 'null'  # 获取文件后缀，例如 ".mp3"
            size = len(audio_contents)
            audio_info = f"File type: file_path, Format: {audio_extension}, Size: {size} bytes."
        else:
            raise FileNotFoundError(f"File not found: {audio_path}.")
    elif audio_base64:
        if audio_base64.startswith("data:"):
            # 使用正则表达式提取音频格式
            match = re.match(r'data:(.*?);base64,(.*)', audio_base64)
            if match:
                mime_type = match.group(1)
                audio_base64 = match.group(2)
                # 从MIME类型中提取音频格式
                audio_format = mimetypes.guess_extension(mime_type, strict=False) or audio_format
        # 解码 base64 编码的音频
        audio_bytes = base64.b64decode(audio_base64)
        audio_contents = BytesIO(audio_bytes)
        size = len(audio_bytes)
        audio_info = f"File type: file_base64, Format: {audio_format}, Size: {size} bytes."
    if not audio_contents or not audio_info:
        raise ValueError("No valid audio content found.")
    audio_info = f'Success load audio contents, {audio_info}'
    return audio_info, audio_contents


def filter_keywords(text):
    # 定义关键词列表
    keywords = ["社群", "社区", "赞助", "订阅", "关注", "平台", "精英", "字幕", "点赞", "栏目", "谢谢观看",
                "感谢观看", "工作室", "业务联系"]
    # 检查文本中是否包含任何关键词
    if any(keyword in text for keyword in keywords):
        text = ""
        code = 1
        messages = "Warning: Transcribe result is empty due to filtered keywords!"
        # 如果包含，返回空字符串
        return text, code, messages
    else:
        code = 0
        messages = "Transcribe audio successfully!"
        # 如果不包含，返回原始文本
        return text, code, messages


# whisper_logger = configure_logging()
whisper_logger = logger
model_dir = "models/faster-whisper-large-v2"
default_audio_dir = './audio'
model = WhisperModel(model_dir, device="cuda", num_workers=4, compute_type="float16")
whisper_app = FastAPI()
# 创建一个线程池
executor = ThreadPoolExecutor(max_workers=10)


class TranscribeRequest(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))  # 动态生成时间戳
    uid: Union[int, str] = 'admin'
    language: str = 'zh'  # 默认语言 "zh"
    initial_prompt: str = '你是劲童，这是简体中文的句子。'  # 初始提示，中文句子
    # initial_prompt: str = '你是劲童，这是 such as 简体中文的句子 and English.'  # 初始提示，中文句子
    # initial_prompt: str = '什么是龋齿，牙周炎，智齿？'  # 初始提示，中文句子
    audio_format: str = ".wav"  # 默认值 ".wav"
    content_type: str = 'audio/wav'  # 文件格式默认值 'audio/wav'
    audio_file: Optional[UploadFile] = File(None)
    audio_path: Optional[str] = None  # 文件路径
    audio_base64: Optional[str] = None  # 音频 base64 编码


class TranscribeResponse(BaseModel):
    code: int
    messages: str
    sno: Optional[Union[int, str]] = None
    text: Optional[str] = None


@whisper_app.get("/")
async def index():
    service_name = """
        <html> <head> <title>whisper_service_api</title> </head>
            <body style="display: flex; justify-content: center;"> <h1>whisper_service_api</h1></body> </html>
        """
    return HTMLResponse(status_code=200, content=service_name)


@whisper_app.get("/http_check")
@whisper_app.get("/health")
async def health():
    """Health check."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    health_data = {"status": "healthy", "timestamp": timestamp}
    # 返回JSON格式的响应
    return JSONResponse(status_code=200, content=health_data)


@whisper_app.post("/v1/asr")
async def transcribe_audio(
        request: Request,
        audio_file: Optional[UploadFile] = File(None),  # 上传的文件
):
    try:
        # 判断请求的内容类型
        if request.headers.get('content-type') == 'application/json':
            json_data = await request.json()
            request = TranscribeRequest(**json_data)
        else:
            # 解析表单数据
            form_data = await request.form()
            request = TranscribeRequest(**form_data)
        task = 'transcribe'
        request_data = request.model_dump()
        sno = request_data['sno']
        uid = request_data['uid']
        language = request_data['language']
        initial_prompt = request_data['initial_prompt']
        audio_format = request_data['audio_format']

        audio_info, audio = await get_audio(audio_file, request_data)
        logs = f"ASR request param: sno={sno}, uid={uid}, initial_prompt={initial_prompt}, audio_format={audio_format}, audio info: {audio_info} "
        whisper_logger.info(logs)
        with torch.no_grad():
            beam_size = 5
            vad_parameters = dict(min_silence_duration_ms=500)
            # segments, _ = model.transcribe(audio, language, task, beam_size, initial_prompt=initial_prompt,
            #                                vad_parameters=vad_parameters, vad_filter=True)
            # 将GPU推理任务放入线程池中以实现异步处理
            segments, _ = await run_in_threadpool(model.transcribe, audio, language, task, beam_size,
                                                  initial_prompt=initial_prompt, vad_parameters=vad_parameters,
                                                  vad_filter=True)
        torch_gc()
        # 将所有 segment.text 拼接成一个完整的字符串
        text = "".join([segment.text for segment in segments])
        logs = text
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # logs = f'[{timestamp}]\n{text}'
        whisper_logger.info(logs)
        text, code, messages = filter_keywords(text)
        results = TranscribeResponse(
            code=code,
            sno=sno,
            messages=messages,
            text=text
        )
        logs = f"ASR response results: {results}\n"
        whisper_logger.info(logs)
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = TranscribeResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Transcribe audio  error: {error_message}\n"
        whisper_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except Exception as e:
        error_message = TranscribeResponse(
            code=-1,
            messages=f"Exception: {str(e)} "
        )
        logs = f"Transcribe audio  error: {error_message}\n"
        whisper_logger.error(logs)
        return JSONResponse(status_code=500, content=error_message.model_dump())


if __name__ == "__main__":
    init_app()
    uvicorn.run(whisper_app, host="0.0.0.0", port=8001)
