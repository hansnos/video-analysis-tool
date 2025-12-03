import streamlit as st
import cv2
import numpy as np
import base64
from openai import OpenAI
import tempfile
import os
from PIL import Image
import io
import json
# 确保安装的是 moviepy==1.0.3
from moviepy.editor import VideoFileClip

# --- 1. 配置与密钥加载 ---
st.set_page_config(
    page_title="视听语言分析工作站", 
    page_icon="🎬", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

try:
    VISION_API_KEY = st.secrets["vision"]["api_key"]
    VISION_BASE_URL = st.secrets["vision"]["base_url"]
    VISION_MODEL = st.secrets["vision"]["model"]
    
    AUDIO_API_KEY = st.secrets["audio"]["api_key"]
    AUDIO_BASE_URL = st.secrets["audio"]["base_url"]
    AUDIO_MODEL = st.secrets["audio"]["model"]
except Exception as e:
    st.error(f"⚠️ 配置缺失: {e}。请检查 secrets.toml")
    st.stop()

# --- 2. 顶级 UI 设计 (复刻参考图风格) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

    /* === 全局深色主题重置 === */
    .stApp {
        background-color: #0B0E14; /* 深邃黑蓝背景 */
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    /* 强制所有文字颜色，解决看不清的问题 */
    h1, h2, h3, p, div, span, label {
        color: #FFFFFF !important;
    }
    .stMarkdown p {
        color: #B0B6BE !important; /* 正文稍微灰一点，形成层次 */
    }

    /* === 标题区域 === */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(41, 121, 255, 0.3); /* 蓝色微光 */
    }
    .subtitle {
        text-align: center;
        color: #8E95A3 !important;
        font-size: 1rem;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* === Tab 导航栏 (复刻胶囊风格) === */
    /* 容器调整：去除底线，居中 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: none !important;
        display: flex;
        flex-wrap: nowrap; /* 禁止换行 */
        white-space: nowrap;
        margin-bottom: 30px;
    }
    
    /* 单个 Tab 按钮 (未选中状态) */
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 22px; /* 胶囊圆角 */
        background-color: #1E232E; /* 深灰底色 */
        color: #B0B6BE !important;
        border: 1px solid #2D3342;
        font-size: 14px;
        font-weight: 500;
        padding: 0 16px; /* 压缩内边距，防止溢出 */
        flex-grow: 1; /* 自动撑满宽度 */
        justify-content: center;
        transition: all 0.2s;
    }
    
    /* 鼠标悬停 */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2D3342;
        color: #FFFFFF !important;
    }

    /* 选中状态 (高亮蓝) */
    .stTabs [aria-selected="true"] {
        background-color: #2979FF !important; /* 参考图的亮蓝 */
        color: #FFFFFF !important;
        border: none;
        box-shadow: 0 4px 15px rgba(41, 121, 255, 0.4); /* 发光效果 */
    }

    /* === 上传框美化 === */
    [data-testid='stFileUploader'] {
        background-color: rgba(30, 35, 46, 0.6);
        border: 2px dashed #444C5C;
        border-radius: 20px;
        padding: 40px 20px;
        text-align: center;
        transition: all 0.3s;
    }
    [data-testid='stFileUploader']:hover {
        border-color: #2979FF;
        background-color: rgba(41, 121, 255, 0.05);
    }
    [data-testid='stFileUploader'] section { background-color: transparent !important; }
    /* 隐藏多余小字 */
    [data-testid='stFileUploader'] small { display: none; }

    /* === 按钮样式 === */
    .stButton > button {
        background: linear-gradient(135deg, #2979FF, #1565C0);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 0;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(41, 121, 255, 0.4);
    }

    /* === 结果卡片系统 === */
    .info-card {
        background-color: #161920;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #2A2F3A;
        position: relative;
        overflow: hidden;
    }
    
    /* 装饰性左边框 */
    .card-style { border-left: 6px solid #FF4081; }
    .card-shot  { border-left: 6px solid #FFD740; }
    .card-prompt{ border-left: 6px solid #448AFF; }
    .card-audio { border-left: 6px solid #00E676; }
    .card-ocr   { border-left: 6px solid #FF6E40; }

    /* 卡片标题 */
    .card-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* 颜色定义 */
    .pink { color: #FF4081 !important; }
    .yellow { color: #FFD740 !important; }
    .blue { color: #448AFF !important; }
    .green { color: #00E676 !important; }
    .orange { color: #FF6E40 !important; }

    /* 内容文本 */
    .card-content {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #D1D5DB !important;
        background: rgba(255,255,255,0.03);
        padding: 12px;
        border-radius: 8px;
    }

    /* 图片容器圆角 */
    img { border-radius: 12px; }

</style>
""", unsafe_allow_html=True)

# --- 3. 逻辑函数 (保持功能不变) ---

def get_image_base64(image_array):
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_frame_at_time(video_path, time_sec=1.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    frame_id = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def detect_scenes_ignore_subtitles(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    prev_hist = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 15 == 0: 
            height, width, _ = frame.shape
            crop_h = int(height * 0.8) 
            cropped_frame = frame[0:crop_h, :] 
            hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            if prev_hist is None:
                frames.append(frame)
                timestamps.append(frame_count / fps)
                prev_hist = hist
            else:
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if (1 - score) > (threshold / 100.0) and (frame_count / fps - timestamps[-1] > 1.5):
                    frames.append(frame)
                    timestamps.append(frame_count / fps)
                    prev_hist = hist
        frame_count += 1
    cap.release()
    return frames, timestamps

def analyze_image_reverse_engineering(image_base64):
    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    system_prompt = """
    请分析图片，严格输出 JSON 格式（不要 Markdown）：
    {
        "style": "风格提示词...",
        "shot": "镜头与景别...",
        "prompt": "英文生成提示词..."
    }
    """
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
            ], max_tokens=800,
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except:
        return {"style": "Error", "shot": "Error", "prompt": "Error"}

def analyze_video_frame_dual(image_base64):
    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    system_prompt = """
    分析视频帧，忽略字幕。请严格按照 JSON 格式输出两部分内容：
    {
        "cn_desc": "中文画面描述（包含环境、主体、动作、氛围）",
        "en_prompt": "High quality English prompt for Sora/Runway video generation"
    }
    """
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
            ], max_tokens=500,
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        return {"cn_desc": "解析失败", "en_prompt": str(e)}

def analyze_ocr_text(image_base64):
    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    system_prompt = "你是一个专业的 OCR 文字识别助手。请识别画面中出现的所有【固定中文文字】，忽略底部的即时字幕。直接输出内容。"
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
            ], max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OCR Error: {str(e)}"

def transcribe_audio_api(video_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_path = temp_audio.name
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3', logger=None, ffmpeg_params=["-ac", "1"])
        video.close()
        client = OpenAI(api_key=AUDIO_API_KEY, base_url=AUDIO_BASE_URL)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
        os.remove(audio_path)
        if isinstance(transcript, str):
            try:
                data = json.loads(transcript)
                if "text" in data: return data["text"]
            except: pass
            return transcript
        return transcript.text
    except Exception as e:
        return f"Audio Error: {str(e)}"

# --- 4. 界面渲染 ---

# 标题区
st.markdown("<h1>视听语言分析工作站</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Visual Intelligence Analysis Workstation</div>", unsafe_allow_html=True)

# Tab 导航区 (名称简化，防止溢出)
tab1, tab2, tab3, tab4 = st.tabs(["图生文反推", "视频拆解", "口播扒取", "文字提取"])

# === Tab 1: 图生文 ===
with tab1:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>上传参考图片，AI 将分别反推其风格、镜头语言及完整的生图提示词。</div>", unsafe_allow_html=True)
    
    # 居中布局
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded_img = st.file_uploader(" ", type=["jpg", "png"], key="img_up")

    if uploaded_img:
        st.write("")
        c_disp, c_a
