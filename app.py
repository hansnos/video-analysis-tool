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
    layout="wide", # <--- 改为 wide，为了让结果展示更宽阔
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

# --- 2. 样式微调 (适配 Wide 模式但保持输入居中) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

    .stApp { background-color: #0B0E14; font-family: 'Noto Sans SC', sans-serif; }
    h1, h2, h3, p, div, span, label { color: #FFFFFF !important; }
    .stMarkdown p { color: #B0B6BE !important; }

    /* 标题 */
    h1 {
        font-size: 2.8rem !important; font-weight: 900 !important; text-align: center;
        margin-top: 20px; margin-bottom: 10px; letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(41, 121, 255, 0.3);
    }
    .subtitle { text-align: center; color: #8E95A3 !important; font-size: 1rem; margin-bottom: 40px; }

    /* Tab 导航栏 (居中) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: transparent; border-bottom: none !important;
        display: flex; justify-content: center; /* 强制居中 */
        flex-wrap: nowrap; margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 22px; background-color: #1E232E; color: #B0B6BE !important;
        border: 1px solid #2D3342; font-size: 14px; font-weight: 500; padding: 0 30px;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #2D3342; color: #FFFFFF !important; }
    .stTabs [aria-selected="true"] {
        background-color: #2979FF !important; color: #FFFFFF !important; border: none;
        box-shadow: 0 4px 15px rgba(41, 121, 255, 0.4);
    }

    /* 上传框 (黑底蓝框) */
    [data-testid='stFileUploader'] {
        background-color: rgba(30, 35, 46, 0.6); border: 2px dashed #444C5C; border-radius: 20px;
        padding: 40px 20px; text-align: center; transition: all 0.3s;
    }
    [data-testid='stFileUploader']:hover { border-color: #2979FF; background-color: rgba(41, 121, 255, 0.05); }
    [data-testid='stFileUploader'] section { background-color: transparent !important; }
    [data-testid='stFileUploader'] small { display: none; }

    /* 结果卡片 */
    .info-card {
        background-color: #161920; border-radius: 16px; padding: 20px; margin-bottom: 20px;
        border: 1px solid #2A2F3A; position: relative;
    }
    .card-style { border-left: 6px solid #FF4081; }
    .card-shot  { border-left: 6px solid #FFD740; }
    .card-prompt{ border-left: 6px solid #448AFF; }
    .card-audio { border-left: 6px solid #00E676; }
    .card-ocr   { border-left: 6px solid #FF6E40; }
    .card-cn    { border-left: 6px solid #9C27B0; }

    .card-header { font-size: 1.1rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
    .pink { color: #FF4081 !important; }
    .yellow { color: #FFD740 !important; }
    .blue { color: #448AFF !important; }
    .green { color: #00E676 !important; }
    .orange { color: #FF6E40 !important; }
    .purple { color: #9C27B0 !important; }

    .card-content {
        font-family: 'JetBrains Mono', monospace; font-size: 1rem; line-height: 1.6;
        color: #D1D5DB !important; background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;
    }
    img { border-radius: 12px; }
    
    /* 下载按钮美化 */
    .stDownloadButton button {
        background-color: transparent !important;
        border: 1px solid #444 !important;
        color: #888 !important;
        font-size: 12px;
        padding: 5px 15px;
    }
    .stDownloadButton button:hover {
        border-color: #2979FF !important;
        color: #2979FF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心逻辑函数 ---

def get_image_base64(image_array):
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# 新增：用于下载图片的转换函数
def convert_frame_to_bytes(frame_array):
    # OpenCV BGR -> RGB -> Bytes
    img = Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

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
    """
    图生文反推模式：升级版 System Prompt，追求 95% 还原度
    """
    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    
    # === 核心修改：赋予 AI 专家人设，要求极度精准的关键词 ===
    system_prompt = """
    你是一位顶级的 AI 绘画提示词工程师（Prompt Engineer），精通 Midjourney、Stable Diffusion 和 Flux 的提示词逻辑。
    请深度剖析这张图片，反推出能完美还原该画面的提示词。
    
    请严格按照以下 JSON 格式输出（不要 Markdown）：
    {
        "style": "这里列出核心艺术风格。例如：Cyberpunk, Ukiyo-e, Oil Painting, 3D Render (Octane), Pixar Style, Matte Painting...",
        "shot": "这里列出镜头与光影。例如：Wide angle, Telephoto lens, Dutch angle, Volumetric lighting, Rim light, Bokeh...",
        "prompt": "这里编写一段高质量的英文 Prompt。必须包含：
                   1. 主体细节（五官、衣着材质、表情）。
                   2. 环境细节（背景元素、天气）。
                   3. 技术参数（如：8k, photorealistic, masterpiece, highly detailed, unreal engine 5）。
                   请使用逗号分隔的关键词形式。"
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=800,
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        return {"style": "Error", "shot": "Error", "prompt": str(e)}

def analyze_video_frame_reconstruction(image_base64):
    """
    针对 90% 还原度的画面帧反推 Prompt (升级版：增强风格与身份识别)
    """
    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    
    # === 这里是核心修改：大幅增强了提示词的要求 ===
    system_prompt = """
    你是一个顶级的 AI 艺术导演和提示词专家。
    请深度分析这张视频截图，目标是生成一段能让 Midjourney/Sora 完美还原画面神韵的英文 Prompt。
    
    请特别注意以下细节的提取：
    1. **人物身份与特征**：不要只说 "Person"。请仔细观察衣着（如长袍、斗笠、破旧衣物），判断是否为 Monk (僧人), Daoist (道士), Wanderer (流浪者) 或 Elder (老者)。
    2. **摄影与艺术风格**：这是写实照片、CG渲染还是黑白电影？如果是黑白的，请加上 "Black and white photography, vintage style, film grain" 等关键词。
    3. **环境与氛围**：描述天气（阴沉、迷雾）、光影（柔光、逆光）及画面的情绪（孤独、史诗感）。
    
    请严格按照 JSON 格式输出：
    {
        "cn_desc": "中文深度画面描述（必须明确写出人物身份，如：背负行囊的苦行僧/老道士，以及画面的黑白复古质感）",
        "en_prompt": "High-fidelity English text-to-image prompt. Include keywords for: Subject Identity (e.g., old monk, ascetic), Clothing (traditional robes), Art Style (e.g., 1920s vintage photography, black and white, grainy film), Lighting, and Atmosphere."
    }
    不要输出 Markdown 标记。
    """
    
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=800, # 稍微增加了 token 限制以允许更详细的描述
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

st.markdown("<h1>视听语言分析工作站</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Visual Intelligence Analysis Workstation</div>", unsafe_allow_html=True)

# Tab 导航区
tab1, tab2, tab3, tab4 = st.tabs(["图生文反推", "视频拆解", "口播扒取", "文字提取"])

# === Tab 1: 图生文 ===
with tab1:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>AI 反推风格、镜头语言及生图提示词</div>", unsafe_allow_html=True)
    
    # 输入区域居中 (1:2:1 布局)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded_img = st.file_uploader(" ", type=["jpg", "png"], key="img_up")

    if uploaded_img:
        # 自动化处理：不需要按钮，直接开始
        with st.spinner("AI 视觉引擎正在解析..."):
            image = Image.open(uploaded_img)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result = analyze_image_reverse_engineering(img_b64)
            
            # 结果展示：左图右文布局 (1:2)
            st.write("")
            r1, r2 = st.columns([1, 2])
            with r1:
                st.image(uploaded_img, caption="原始图片", use_container_width=True)
            with r2:
                st.markdown(f"""
                <div class="info-card card-style">
                    <div class="card-header pink">🎨 风格提示词 (Style)</div>
                    <div class="card-content">{result.get('style', 'N/A')}</div>
                </div>
                <div class="info-card card-shot">
                    <div class="card-header yellow">📷 镜头与景别 (Shot)</div>
                    <div class="card-content">{result.get('shot', 'N/A')}</div>
                </div>
                <div class="info-card card-prompt">
                    <div class="card-header blue">✨ AI 生图提示词 (Prompt)</div>
                    <div class="card-content" style="user-select: all;">{result.get('prompt', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

# === Tab 2: 视频拆解 (核心修改区) ===
with tab2:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>生成画面帧双语提示词 (适用于即梦/NanoBanana画面还原)</div>", unsafe_allow_html=True)
    
    # 输入区域居中
    t2_c1, t2_c2, t2_c3 = st.columns([1, 2, 1])
    with t2_c2:
        v_file = st.file_uploader(" ", type=["mp4", "mov"], key="v_up")
        threshold = st.slider("切镜灵敏度", 10, 60, 25)

    if v_file:
        # 自动化处理
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(v_file.read())
        
        with st.status("正在逐帧分析与生成提示词...", expanded=True) as status:
            frames, tstamps = detect_scenes_ignore_subtitles(tfile.name, threshold)
            st.write(f"检测到 {len(frames)} 个关键镜头，正在生成还原 Prompt...")
            
            res_container = st.container()
            for i, (frm, ts) in enumerate(zip(frames, tstamps)):
                b64 = get_image_base64(frm)
                res = analyze_video_frame_reconstruction(b64)
                
                with res_container:
                    # 结果布局：图片变大 (2:3 布局)
                    res_c1, res_c2 = st.columns([2, 3])
                    
                    with res_c1:
                        st.image(frm, channels="BGR", use_container_width=True)
                        # 下载按钮逻辑
                        img_bytes = convert_frame_to_bytes(frm)
                        st.download_button(
                            label="📥 下载该帧",
                            data=img_bytes,
                            file_name=f"frame_{ts:.2f}.png",
                            mime="image/png",
                            key=f"dl_{ts}"
                        )
                        st.caption(f"⏱️ 时间点: {ts:.2f}s")
                        
                    with res_c2:
                        st.markdown(f"""
                        <div class="info-card card-cn" style="margin-bottom:10px;">
                            <div class="card-header purple">📝 中文画面描述</div>
                            <div class="card-content">{res.get('cn_desc', '...')}</div>
                        </div>
                        <div class="info-card card-prompt">
                            <div class="card-header blue">✨ 画面还原 Prompt (Image Gen)</div>
                            <div class="card-content" style="user-select: all;">{res.get('en_prompt', '...')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.divider()
            status.update(label="✅ 分析完成", state="complete", expanded=False)

# === Tab 3: 口播扒取 ===
with tab3:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>提取语音，转换为逐字稿</div>", unsafe_allow_html=True)
    
    t3_c1, t3_c2, t3_c3 = st.columns([1, 2, 1])
    with t3_c2:
        a_file = st.file_uploader(" ", type=["mp4", "mp3", "wav"], key="a_up")
    
    if a_file:
        # 自动化处理
        tfile_a = tempfile.NamedTemporaryFile(delete=False)
        tfile_a.write(a_file.read())
        with st.spinner("AI 听写中..."):
            txt = transcribe_audio_api(tfile_a.name)
            
            # 结果展示居中
            r3_c1, r3_c2, r3_c3 = st.columns([1, 6, 1])
            with r3_c2:
                st.audio(a_file)
                st.markdown(f"""
                <div class="info-card card-audio">
                    <div class="card-header green">🎙️ 逐字稿 (Transcript)</div>
                    <div class="card-content" style="user-select: all;">{txt}</div>
                </div>
                """, unsafe_allow_html=True)

# === Tab 4: 文字提取 ===
with tab4:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>识别大字报、包装文字及关键信息</div>", unsafe_allow_html=True)
    
    t4_c1, t4_c2, t4_c3 = st.columns([1, 2, 1])
    with t4_c2:
        ocr_file = st.file_uploader(" ", type=["mp4", "mov"], key="ocr_up")
    
    if ocr_file:
        # 自动化处理
        tfile_ocr = tempfile.NamedTemporaryFile(delete=False)
        tfile_ocr.write(ocr_file.read())
        frame = get_frame_at_time(tfile_ocr.name, time_sec=1.5)
        
        if frame is not None:
            with st.spinner("OCR 识别中..."):
                b64 = get_image_base64(frame)
                ocr_text = analyze_ocr_text(b64)
                
                # 结果展示 (1:1 布局)
                ocr_c1, ocr_c2 = st.columns([1, 1])
                with ocr_c1:
                    st.image(frame, channels="BGR", caption="识别帧", use_container_width=True)
                with ocr_c2:
                    st.markdown(f"""
                    <div class="info-card card-ocr">
                        <div class="card-header orange">🔠 提取结果 (OCR)</div>
                        <div class="card-content" style="white-space: pre-line; user-select: all;">{ocr_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

