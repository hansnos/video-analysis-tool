import streamlit as st
import cv2
import numpy as np
import base64
from openai import OpenAI
import tempfile
import os
from PIL import Image
import io
import time
from moviepy.editor import VideoFileClip

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="Video Analysis Platform", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. 瑞士平面设计风格 (Swiss Style CSS) ---
# 特点：高对比度、巨型字体、黑白分明、绝对清晰
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Now+Display:wght@400;900&display=swap');

    /* 全局重置 */
    .stApp {
        background-color: #FFFFFF; /* 纯白背景 */
        color: #000000; /* 纯黑文字 */
        font-family: 'Helvetica Now Display', 'Arial', sans-serif;
    }

    /* 侧边栏：极简黑 */
    [data-testid="stSidebar"] {
        background-color: #F4F4F4;
        border-right: 3px solid #000000;
    }

    /* 标题系统：巨型、加粗 */
    h1 {
        font-size: 4rem !important;
        font-weight: 900 !important;
        letter-spacing: -2px;
        line-height: 1;
        text-transform: uppercase;
        margin-bottom: 20px;
        color: #000000;
    }
    h5 {
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        border-bottom: 3px solid #000000;
        padding-bottom: 5px;
        margin-top: 30px !important;
        margin-bottom: 15px !important;
        color: #000000 !important;
    }

    /* === 核心交互组件 === */

    /* 1. 上传框 (File Uploader) - 绝对高亮 */
    [data-testid='stFileUploader'] {
        background-color: #FFF000; /* 亮黄色背景，绝对醒目 */
        border: 4px solid #000000; /* 极粗黑边框 */
        padding: 30px;
        border-radius: 0px; /* 直角 */
        text-align: center;
    }
    /* 强制上传框内的文字为纯黑粗体 */
    [data-testid='stFileUploader'] div, 
    [data-testid='stFileUploader'] span, 
    [data-testid='stFileUploader'] small,
    [data-testid='stFileUploader'] label {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
    }
    [data-testid='stFileUploader'] button {
        border: 2px solid #000000 !important;
        color: #000000 !important;
        background-color: #FFFFFF !important;
        font-weight: 900;
    }

    /* 2. 单选框 (Radio) */
    [role="radiogroup"] {
        background-color: #000000;
        padding: 15px;
        color: #FFFFFF;
    }
    .stRadio label {
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* 3. 按钮 (Primary Button) */
    .stButton > button {
        background-color: #000000;
        color: #FFFFFF !important;
        border: none;
        border-radius: 0px;
        font-weight: 900;
        font-size: 1.5rem;
        padding: 15px 30px;
        text-transform: uppercase;
        border: 3px solid #000000;
        transition: all 0.1s;
    }
    .stButton > button:hover {
        background-color: #FFFFFF;
        color: #000000 !important;
        transform: translate(-4px, -4px);
        box-shadow: 6px 6px 0px #000000;
    }

    /* 4. 输入框 */
    .stTextInput input {
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: bold;
    }

    /* 5. 结果卡片 */
    .result-card {
        border: 3px solid #000000;
        background-color: #FFFFFF;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 8px 8px 0px #EEEEEE;
    }
    .result-label {
        background-color: #000000;
        color: #FFFFFF;
        padding: 5px 10px;
        font-weight: 900;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 1.1rem;
        line-height: 1.5;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心逻辑函数 ---

def get_image_base64(image_array):
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def detect_scenes(video_path, threshold=30.0):
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
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            if prev_hist is None:
                frames.append(frame)
                timestamps.append(frame_count / fps)
                prev_hist = hist
            else:
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if (1 - score) > (threshold / 100.0) and (frame_count / fps - timestamps[-1] > 2.0):
                    frames.append(frame)
                    timestamps.append(frame_count / fps)
                    prev_hist = hist
        frame_count += 1
    cap.release()
    return frames, timestamps

def analyze_image(image_base64, api_key, base_url, model):
    """视觉分析：使用 Vision Key"""
    if not api_key: return "Error: Missing Vision API Key"
    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt = "分析画面。输出：1. English Prompt. 2. Chinese Description. 纯文本。"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Vision Error: {str(e)}"

def transcribe_audio(video_path, api_key, base_url, model):
    """音频转写：使用 Audio Key"""
    if not api_key: return "Error: Missing Audio API Key"
    
    try:
        # 1. 提取音频并转单声道 (修复 code 1214)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_path = temp_audio.name
        
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            audio_path, 
            codec='mp3', 
            logger=None, 
            ffmpeg_params=["-ac", "1"] # 强制单声道
        )
        video.close()
        
        # 2. 调用 API
        client = OpenAI(api_key=api_key, base_url=base_url)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model, 
                file=audio_file,
                response_format="text",
                prompt="中文口播内容。"
            )
        os.remove(audio_path) 
        return transcript
        
    except Exception as e:
        error_msg = str(e)
        if "model_not_found" in error_msg or "1211" in error_msg:
            return f"❌ 配置错误：模型 '{model}' 不存在。\n\n原因：你可能在用智谱的 Key 调用 OpenAI 的模型。\n解决：请在左侧 'AUDIO SETTINGS' 中填入正确的 OpenAI Key，或者将模型改为智谱支持的名称（如果支持）。推荐使用 OpenAI 官方 Key 或 Groq。"
        return f"Audio Error: {error_msg}"

# --- 4. 侧边栏：双通道配置 (关键修改) ---
with st.sidebar:
    st.markdown("<h5>1. VISION SETTINGS (视觉)</h5>", unsafe_allow_html=True)
    st.caption("用于画面拆解 (推荐智谱免费版)")
    v_key = st.text_input("Vision API Key", type="password", help="智谱 Key")
    v_url = st.text_input("Vision Base URL", value="https://open.bigmodel.cn/api/paas/v4/")
    v_model = st.text_input("Vision Model", value="glm-4v-flash")
    
    st.markdown("<h5>2. AUDIO SETTINGS (听觉)</h5>", unsafe_allow_html=True)
    st.caption("用于口播提取 (智谱不支持 whisper-1，请用 OpenAI/Groq)")
    a_key = st.text_input("Audio API Key", type="password", help="OpenAI / Groq Key")
    a_url = st.text_input("Audio Base URL", value="https://api.openai.com/v1")
    a_model = st.text_input("Audio Model", value="whisper-1")

# --- 5. 主界面 ---
st.markdown("<h1>VIDEO ANALYSIS<br>PLATFORM</h1>", unsafe_allow_html=True)

mode = st.radio(
    "SELECT MODE",
    ["VISUAL ANALYSIS (画面拆解)", "AUDIO DECRYPT (口播提取)"],
    horizontal=True
)

st.write("") # Spacer

if "VISUAL" in mode:
    # ---------------- 画面拆解 ----------------
    col1, col2 = st.columns([1, 1.2]) 
    
    with col1:
        st.markdown("<h5>INPUT SOURCE</h5>", unsafe_allow_html=True)
        v_file = st.file_uploader("DROP VIDEO HERE (MP4)", type=["mp4", "mov"], key="v_up")
        
        st.markdown("<h5>SENSITIVITY</h5>", unsafe_allow_html=True)
        threshold = st.slider("CUT THRESHOLD", 10, 60, 30)
        
        if v_file:
            st.video(v_file)
            st.write("")
            if st.button("START VISUAL ANALYSIS"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(v_file.read())
                
                with st.status("PROCESSING VISUAL DATA...", expanded=True) as status:
                    frames, tstamps = detect_scenes(tfile.name, threshold)
                    
                    res_area = st.container()
                    for i, (frm, ts) in enumerate(zip(frames, tstamps)):
                        b64 = get_image_base64(frm)
                        # 使用 Vision 配置
                        txt = analyze_image(b64, v_key, v_url, v_model)
                        
                        with res_area:
                            c1, c2 = st.columns([1, 2])
                            c1.image(frm, channels="BGR", use_container_width=True)
                            with c2:
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="result-label">{ts:.2f}s</div>
                                    <div class="result-text">{txt}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    status.update(label="DONE", state="complete", expanded=False)

    with col2:
        st.markdown("<h5>OUTPUT LOG</h5>", unsafe_allow_html=True)
        if not v_file:
            st.info("WAITING FOR FILE...")

elif "AUDIO" in mode:
    # ---------------- 口播提取 ----------------
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<h5>AUDIO SOURCE</h5>", unsafe_allow_html=True)
        a_file = st.file_uploader("DROP MEDIA HERE (MP4/MP3)", type=["mp4", "mp3", "wav"], key="a_up")
        
        if a_file:
            st.audio(a_file)
            st.write("")
            if st.button("EXTRACT TRANSCRIPT"):
                tfile_a = tempfile.NamedTemporaryFile(delete=False)
                tfile_a.write(a_file.read())
                
                with col2:
                    with st.spinner("TRANSCRIBING..."):
                        # 使用 Audio 配置
                        txt = transcribe_audio(tfile_a.name, a_key, a_url, a_model)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">FULL TRANSCRIPT</div>
                        <div class="result-text">{txt}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text_area("COPY TEXT", value=txt, height=400)
            
    with col2:
        st.markdown("<h5>TEXT RESULT</h5>", unsafe_allow_html=True)
        if not a_file:

            st.info("WAITING FOR AUDIO...")
