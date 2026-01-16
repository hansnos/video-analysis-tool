import streamlit as st
import cv2
import numpy as np
import base64
from openai import OpenAI
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import io
import json
import subprocess
from datetime import datetime
import zipfile

# 确保安装的是 moviepy==1.0.3
from moviepy.editor import VideoFileClip

# --- 1. 配置与密钥加载 ---
st.set_page_config(
    page_title="视听语言分析工作站", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API 配置
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

# Google Sheets 配置（可选）
try:
    GSHEET_CREDENTIALS = st.secrets["gsheet"]["credentials"]
    GSHEET_URL = st.secrets["gsheet"]["sheet_url"]
    GSHEET_ENABLED = True
except:
    GSHEET_ENABLED = False

# --- 用户账号系统 ---
USERS = {
    "Baihe123": "Hengxing666",
    "Shujun123": "Hengxing666",
    "Hans123": "Hengxing666",
    "Heixin123": "Hengxing666",
}

def check_login():
    """检查登录状态"""
    return st.session_state.get("logged_in", False)

def get_current_user():
    """获取当前登录用户"""
    return st.session_state.get("username", None)

def log_usage(username, feature, options=""):
    """记录使用日志到 Google Sheets"""
    if not GSHEET_ENABLED:
        return
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds_dict = json.loads(GSHEET_CREDENTIALS)
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        sheet = gc.open_by_url(GSHEET_URL).sheet1
        
        # 记录：时间、用户、功能、选项
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, username, feature, options])
        
    except Exception as e:
        st.warning(f"日志记录失败: {e}")

# --- 2. 样式 ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

    .stApp { background-color: #0B0E14; font-family: 'Noto Sans SC', sans-serif; }
    h1, h2, h3, p, div, span, label { color: #FFFFFF !important; }
    .stMarkdown p { color: #B0B6BE !important; }

    h1 {
        font-size: 2.8rem !important; font-weight: 900 !important; text-align: center;
        margin-top: 20px; margin-bottom: 10px; letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(41, 121, 255, 0.3);
    }
    .subtitle { text-align: center; color: #8E95A3 !important; font-size: 1rem; margin-bottom: 40px; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: transparent; border-bottom: none !important;
        display: flex; justify-content: center;
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

    [data-testid='stFileUploader'] {
        background-color: rgba(30, 35, 46, 0.6); border: 2px dashed #444C5C; border-radius: 20px;
        padding: 40px 20px; text-align: center; transition: all 0.3s;
    }
    [data-testid='stFileUploader']:hover { border-color: #2979FF; background-color: rgba(41, 121, 255, 0.05); }
    [data-testid='stFileUploader'] section { background-color: transparent !important; }
    [data-testid='stFileUploader'] small { display: none; }

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
    .card-poster { border-left: 6px solid #00BCD4; }

    .card-header { font-size: 1.1rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
    .pink { color: #FF4081 !important; }
    .yellow { color: #FFD740 !important; }
    .blue { color: #448AFF !important; }
    .green { color: #00E676 !important; }
    .orange { color: #FF6E40 !important; }
    .purple { color: #9C27B0 !important; }
    .cyan { color: #00BCD4 !important; }

    .card-content {
        font-family: 'JetBrains Mono', monospace; font-size: 1rem; line-height: 1.6;
        color: #D1D5DB !important; background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;
    }
    img { border-radius: 12px; }
    
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
    
    /* 登录框样式 */
    .login-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 40px;
        max-width: 400px;
        margin: 50px auto;
    }
    .login-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 30px;
        color: #00BCD4 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心逻辑函数 ---

def get_image_base64(image_array):
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def convert_frame_to_bytes(frame_array):
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

def get_video_dimensions(video_path):
    """获取视频尺寸"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

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
    client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
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
            max_tokens=800,
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

# --- 大字报生成函数 ---

def load_font(size, weight="Regular"):
    """加载思源黑体（从 Google Fonts CDN 或本地）"""
    # 尝试加载本地字体文件
    font_paths = [
        "NotoSansSC-Bold.ttf",
        "NotoSansSC-Regular.ttf", 
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    
    # 如果没有本地字体，尝试下载
    try:
        import urllib.request
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Bold.otf"
        font_path = "/tmp/NotoSansSC-Bold.otf"
        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)
        return ImageFont.truetype(font_path, size)
    except:
        # 最后使用默认字体
        return ImageFont.load_default()

def generate_poster_v1(width, height, line1, line2, line3):
    """
    V1 样式：标准居中布局
    - 标题：大号黄色
    - 副标题：中号白色
    - 评论：中号黄色
    """
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 字体大小（根据视频宽度自适应）
    title_size = int(width * 0.08)
    subtitle_size = int(width * 0.045)
    comment_size = int(width * 0.05)
    
    font_title = load_font(title_size)
    font_subtitle = load_font(subtitle_size)
    font_comment = load_font(comment_size)
    
    # 颜色
    yellow = (255, 255, 0, 255)
    white = (255, 255, 255, 255)
    
    # 计算位置
    margin_top = int(height * 0.05)
    line_spacing = int(height * 0.02)
    
    # 第1行：黄色大标题
    bbox1 = draw.textbbox((0, 0), line1, font=font_title)
    x1 = (width - (bbox1[2] - bbox1[0])) // 2
    y1 = margin_top
    draw.text((x1, y1), line1, font=font_title, fill=yellow)
    
    # 第2行：白色副标题
    bbox2 = draw.textbbox((0, 0), line2, font=font_subtitle)
    x2 = (width - (bbox2[2] - bbox2[0])) // 2
    y2 = y1 + (bbox1[3] - bbox1[1]) + line_spacing
    draw.text((x2, y2), line2, font=font_subtitle, fill=white)
    
    # 第3行：黄色评论
    bbox3 = draw.textbbox((0, 0), line3, font=font_comment)
    x3 = (width - (bbox3[2] - bbox3[0])) // 2
    y3 = y2 + (bbox2[3] - bbox2[1]) + line_spacing * 1.5
    draw.text((x3, y3), line3, font=font_comment, fill=yellow)
    
    return img

def generate_poster_v2(width, height, line1, line2, line3):
    """
    V2 样式：大行距 + 较小字体
    - 整体更舒朗
    - 副标题用浅灰色
    """
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 字体大小（比V1小）
    title_size = int(width * 0.07)
    subtitle_size = int(width * 0.038)
    comment_size = int(width * 0.042)
    
    font_title = load_font(title_size)
    font_subtitle = load_font(subtitle_size)
    font_comment = load_font(comment_size)
    
    # 颜色
    yellow = (255, 220, 0, 255)  # 偏暖黄
    light_gray = (200, 200, 200, 255)
    orange_yellow = (255, 180, 0, 255)
    
    # 计算位置（更大的边距和行距）
    margin_top = int(height * 0.06)
    line_spacing = int(height * 0.035)
    
    # 第1行
    bbox1 = draw.textbbox((0, 0), line1, font=font_title)
    x1 = (width - (bbox1[2] - bbox1[0])) // 2
    y1 = margin_top
    draw.text((x1, y1), line1, font=font_title, fill=yellow)
    
    # 第2行
    bbox2 = draw.textbbox((0, 0), line2, font=font_subtitle)
    x2 = (width - (bbox2[2] - bbox2[0])) // 2
    y2 = y1 + (bbox1[3] - bbox1[1]) + line_spacing
    draw.text((x2, y2), line2, font=font_subtitle, fill=light_gray)
    
    # 第3行
    bbox3 = draw.textbbox((0, 0), line3, font=font_comment)
    x3 = (width - (bbox3[2] - bbox3[0])) // 2
    y3 = y2 + (bbox2[3] - bbox2[1]) + line_spacing * 2
    draw.text((x3, y3), line3, font=font_comment, fill=orange_yellow)
    
    return img

def generate_poster_v3(width, height, line1, line2, line3):
    """
    V3 样式：超大标题 + 紧凑布局
    - 标题特别大
    - 整体更紧凑有冲击力
    """
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 字体大小（标题超大）
    title_size = int(width * 0.10)
    subtitle_size = int(width * 0.04)
    comment_size = int(width * 0.055)
    
    font_title = load_font(title_size)
    font_subtitle = load_font(subtitle_size)
    font_comment = load_font(comment_size)
    
    # 颜色（高对比度）
    bright_yellow = (255, 255, 50, 255)
    white = (255, 255, 255, 255)
    gold = (255, 215, 0, 255)
    
    # 计算位置（紧凑）
    margin_top = int(height * 0.04)
    line_spacing = int(height * 0.015)
    
    # 第1行
    bbox1 = draw.textbbox((0, 0), line1, font=font_title)
    x1 = (width - (bbox1[2] - bbox1[0])) // 2
    y1 = margin_top
    draw.text((x1, y1), line1, font=font_title, fill=bright_yellow)
    
    # 第2行
    bbox2 = draw.textbbox((0, 0), line2, font=font_subtitle)
    x2 = (width - (bbox2[2] - bbox2[0])) // 2
    y2 = y1 + (bbox1[3] - bbox1[1]) + line_spacing
    draw.text((x2, y2), line2, font=font_subtitle, fill=white)
    
    # 第3行
    bbox3 = draw.textbbox((0, 0), line3, font=font_comment)
    x3 = (width - (bbox3[2] - bbox3[0])) // 2
    y3 = y2 + (bbox2[3] - bbox2[1]) + line_spacing
    draw.text((x3, y3), line3, font=font_comment, fill=gold)
    
    return img

def process_video_with_effects(video_path, mirror=False, high_saturation=False):
    """
    处理视频：镜像 / 高饱和度高亮度
    返回处理后的视频路径
    """
    if not mirror and not high_saturation:
        return video_path
    
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    
    filters = []
    if mirror:
        filters.append("hflip")
    if high_saturation:
        filters.append("eq=saturation=1.5:brightness=0.1")
    
    filter_str = ",".join(filters)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filter_str,
        "-c:a", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"视频处理失败: {e}")
        return video_path

def overlay_png_on_video(video_path, png_image, output_path):
    """
    使用 FFmpeg 将 PNG 叠加到视频上
    """
    # 保存 PNG 到临时文件
    png_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png_image.save(png_temp.name, "PNG")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", png_temp.name,
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
        "-c:a", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(png_temp.name)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"叠加失败: {e.stderr.decode()}")
        os.remove(png_temp.name)
        return False

def generate_all_videos(video_path, line1, line2, line3, use_mirror, use_saturation):
    """
    生成所有版本的视频
    返回: [(文件名, 文件路径), ...]
    """
    results = []
    width, height = get_video_dimensions(video_path)
    
    # 生成三个版本的 PNG
    posters = {
        "V1": generate_poster_v1(width, height, line1, line2, line3),
        "V2": generate_poster_v2(width, height, line1, line2, line3),
        "V3": generate_poster_v3(width, height, line1, line2, line3),
    }
    
    # 确定要处理的效果组合
    effect_combinations = []
    
    if not use_mirror and not use_saturation:
        # 无特效：只生成原版
        effect_combinations.append(("原版", video_path))
    else:
        if use_mirror:
            mirror_video = process_video_with_effects(video_path, mirror=True, high_saturation=False)
            effect_combinations.append(("镜像", mirror_video))
        if use_saturation:
            sat_video = process_video_with_effects(video_path, mirror=False, high_saturation=True)
            effect_combinations.append(("高饱和", sat_video))
    
    # 为每个效果组合生成 V1/V2/V3
    for effect_name, processed_video in effect_combinations:
        for version, poster_img in posters.items():
            output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = output_file.name
            
            if overlay_png_on_video(processed_video, poster_img, output_path):
                if effect_name == "原版":
                    filename = f"大字报_{version}.mp4"
                else:
                    filename = f"大字报_{effect_name}_{version}.mp4"
                results.append((filename, output_path))
    
    return results

# --- 4. 界面渲染 ---

st.markdown("<h1>视听语言分析工作站</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Visual Intelligence Analysis Workstation</div>", unsafe_allow_html=True)

# Tab 导航区
tab1, tab2, tab3, tab4, tab5 = st.tabs(["图生文反推", "视频拆解", "口播扒取", "文字提取", "🔒 大字报生成"])

# === Tab 1: 图生文 ===
with tab1:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>AI 反推风格、镜头语言及生图提示词</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded_img = st.file_uploader(" ", type=["jpg", "png"], key="img_up")

    if uploaded_img:
        with st.spinner("AI 视觉引擎正在解析..."):
            image = Image.open(uploaded_img)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result = analyze_image_reverse_engineering(img_b64)
            
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

# === Tab 2: 视频拆解 ===
with tab2:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>生成画面帧双语提示词 (适用于即梦/NanoBanana画面还原)</div>", unsafe_allow_html=True)
    
    t2_c1, t2_c2, t2_c3 = st.columns([1, 2, 1])
    with t2_c2:
        v_file = st.file_uploader(" ", type=["mp4", "mov"], key="v_up")
        threshold = st.slider("切镜灵敏度", 10, 60, 25)

    if v_file:
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
                    res_c1, res_c2 = st.columns([2, 3])
                    
                    with res_c1:
                        st.image(frm, channels="BGR", use_container_width=True)
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
        tfile_a = tempfile.NamedTemporaryFile(delete=False)
        tfile_a.write(a_file.read())
        with st.spinner("AI 听写中..."):
            txt = transcribe_audio_api(tfile_a.name)
            
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
        tfile_ocr = tempfile.NamedTemporaryFile(delete=False)
        tfile_ocr.write(ocr_file.read())
        frame = get_frame_at_time(tfile_ocr.name, time_sec=1.5)
        
        if frame is not None:
            with st.spinner("OCR 识别中..."):
                b64 = get_image_base64(frame)
                ocr_text = analyze_ocr_text(b64)
                
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

# === Tab 5: 大字报生成（需登录） ===
with tab5:
    st.markdown("<div style='text-align:center; color:#888; margin-bottom:10px;'>🔐 团队专用功能 - 自动生成大字报视频</div>", unsafe_allow_html=True)
    
    # 检查登录状态
    if not check_login():
        # 显示登录框
        login_c1, login_c2, login_c3 = st.columns([1, 1.5, 1])
        with login_c2:
            st.markdown("""
            <div style="text-align:center; margin: 30px 0;">
                <span style="font-size: 3rem;">🔐</span>
                <h3 style="margin-top: 10px;">团队成员登录</h3>
            </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("账号", placeholder="请输入账号")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            
            if st.button("登 录", use_container_width=True):
                if username in USERS and USERS[username] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success(f"✅ 欢迎回来，{username}！")
                    st.rerun()
                else:
                    st.error("❌ 账号或密码错误")
    else:
        # 已登录，显示功能界面
        current_user = get_current_user()
        
        # 顶部显示用户信息和退出按钮
        user_col1, user_col2 = st.columns([6, 1])
        with user_col1:
            st.markdown(f"<div style='color:#00BCD4;'>👤 当前用户: <b>{current_user}</b></div>", unsafe_allow_html=True)
        with user_col2:
            if st.button("退出登录"):
                st.session_state["logged_in"] = False
                st.session_state["username"] = None
                st.rerun()
        
        st.divider()
        
        # 主功能区
        main_c1, main_c2 = st.columns([1, 1])
        
        with main_c1:
            st.markdown("### 📤 上传视频")
            poster_video = st.file_uploader("拖入 MP4 文件", type=["mp4", "mov"], key="poster_video")
            
            if poster_video:
                st.video(poster_video)
        
        with main_c2:
            st.markdown("### ✏️ 输入文字")
            line1 = st.text_input("第1行（黄色大标题）", value="三国&模拟&经营", placeholder="例：三国&模拟&经营")
            line2 = st.text_input("第2行（白色副标题）", value="一款以模拟经营为核心的现代三国手游", placeholder="例：一款以模拟经营为核心的...")
            line3 = st.text_input("第3行（黄色评论）", value="玩家：玩了三天还在新手村经营木材厂", placeholder="例：玩家：玩了三天...")
            
            st.markdown("### ⚙️ 特效选项")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                use_mirror = st.checkbox("🔄 镜像处理", help="水平翻转视频")
            with col_opt2:
                use_saturation = st.checkbox("🌈 高饱和高亮度", help="提升画面鲜艳度")
        
        st.divider()
        
        # 预览区域
        if poster_video and line1:
            st.markdown("### 👁️ 样式预览")
            
            # 临时保存视频获取尺寸
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(poster_video.read())
            poster_video.seek(0)  # 重置读取位置
            
            width, height = get_video_dimensions(temp_video.name)
            
            # 生成预览图
            preview_cols = st.columns(3)
            
            posters = [
                ("V1 标准", generate_poster_v1(width, height, line1, line2, line3)),
                ("V2 舒朗", generate_poster_v2(width, height, line1, line2, line3)),
                ("V3 冲击", generate_poster_v3(width, height, line1, line2, line3)),
            ]
            
            # 获取视频第一帧作为背景
            bg_frame = get_frame_at_time(temp_video.name, 0.5)
            
            for i, (name, poster) in enumerate(posters):
                with preview_cols[i]:
                    # 合成预览图
                    if bg_frame is not None:
                        bg_img = Image.fromarray(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
                        bg_img = bg_img.resize((width, height))
                        preview = Image.alpha_composite(bg_img, poster)
                        st.image(preview, caption=name, use_container_width=True)
                    else:
                        # 纯黑背景预览
                        black_bg = Image.new('RGBA', (width, height), (0, 0, 0, 255))
                        preview = Image.alpha_composite(black_bg, poster)
                        st.image(preview, caption=name, use_container_width=True)
        
        st.divider()
        
        # 生成按钮
        gen_c1, gen_c2, gen_c3 = st.columns([1, 2, 1])
        with gen_c2:
            generate_btn = st.button("🚀 生成大字报视频", use_container_width=True, type="primary")
        
        if generate_btn and poster_video:
            # 记录使用日志
            options_str = []
            if use_mirror: options_str.append("镜像")
            if use_saturation: options_str.append("高饱和")
            log_usage(current_user, "大字报生成", ", ".join(options_str) if options_str else "无特效")
            
            with st.status("正在生成视频...", expanded=True) as status:
                # 保存上传的视频
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                poster_video.seek(0)
                temp_input.write(poster_video.read())
                temp_input.close()
                
                st.write("📐 读取视频信息...")
                
                st.write("🎨 生成大字报 PNG...")
                
                st.write("🎬 合成视频...")
                
                results = generate_all_videos(
                    temp_input.name, 
                    line1, line2, line3,
                    use_mirror, use_saturation
                )
                
                status.update(label=f"✅ 生成完成！共 {len(results)} 个视频", state="complete")
            
            # 显示下载按钮
            if results:
                st.markdown("### 📥 下载生成的视频")
                
                download_cols = st.columns(3)
                for i, (filename, filepath) in enumerate(results):
                    with download_cols[i % 3]:
                        with open(filepath, "rb") as f:
                            st.download_button(
                                label=f"📥 {filename}",
                                data=f.read(),
                                file_name=filename,
                                mime="video/mp4",
                                key=f"dl_poster_{i}"
                            )
                
                # 提供打包下载
                if len(results) > 1:
                    st.divider()
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for filename, filepath in results:
                            zf.write(filepath, filename)
                    
                    st.download_button(
                        label="📦 打包下载全部",
                        data=zip_buffer.getvalue(),
                        file_name="大字报视频合集.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                
                # 清理临时文件
                for _, filepath in results:
                    try:
                        os.remove(filepath)
                    except:
                        pass
