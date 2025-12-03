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
