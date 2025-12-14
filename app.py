"""
情感分析 Streamlit 应用
优化版本 - 使用统一配置和工具函数
"""
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt
from pathlib import Path
import sys

# 导入项目配置和工具
import config
import utils

# 设置页面配置
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT
)

# ============= 模型加载 =============
@st.cache_resource
def load_model():
    """
    加载模型和tokenizer（带缓存）

    Returns:
        tuple: (tokenizer, model, device) 或 (None, None, None) 如果失败
    """
    try:
        model_path = config.MODEL_DIR

        # 检查模型目录是否存在
        if not Path(model_path).exists():
            st.error(
                f"❌ 模型目录不存在: {model_path}\n\n"
                "请先运行 04_BERT_Finetune.ipynb 训练模型"
            )
            return None, None, None

        # 使用 CPU 进行推理（单条数据推理时 CPU 已足够快）
        device = torch.device("cpu")

        # 加载 tokenizer 和模型
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        st.success("✅ 模型加载成功")
        return tokenizer, model, device

    except Exception as e:
        st.error(
            f"❌ 模型加载失败！\n\n"
            f"错误信息: {e}\n\n"
            f"请检查:\n"
            f"1. 模型路径 '{model_path}' 是否存在\n"
            f"2. 是否已运行训练脚本生成模型文件"
        )
        return None, None, None


# ============= 侧边栏 =============
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
        st.title("🤖 AI 舆情分析助手")
        st.markdown("---")

        st.markdown(f"""
        **项目信息:**
        - **课程:** {config.MODEL_INFO['course']}
        - **模型:** {config.MODEL_INFO['model_name']}
        - **准确率:** {config.MODEL_INFO['accuracy']}

        **业务价值:**

        该系统帮助产品经理从海量评论中：
        1. 🔍 **自动识别** 用户痛点 (Negative)
        2. ⚖️ **精准判断** 中立建议 (Neutral)
        3. 🚀 **量化** 用户满意度

        **技术亮点:**
        - 基于 DistilBERT 预训练模型
        - 随机种子控制确保可复现
        - 统一配置管理
        - 完善的错误处理
        """)

        st.markdown("---")
        st.info("💡 Tip: 尝试输入复杂句子，测试模型的语义理解能力。")

        # 显示配置信息
        with st.expander("⚙️ 配置信息"):
            st.text(f"设备: {utils.get_device_info()}")
            st.text(f"随机种子: {config.RANDOM_SEED}")
            st.text(f"最大序列长度: {config.MAX_SEQ_LENGTH}")


# ============= 主界面 =============
def render_main_interface(tokenizer, model, device):
    """
    渲染主界面

    Args:
        tokenizer: BERT tokenizer
        model: 训练好的模型
        device: 运行设备
    """
    st.title("Voice of User Analysis Dashboard")
    st.markdown("### 🗣️ 实时评论情感检测 (Real-time Sentiment Detection)")

    # 输入区
    text_input = st.text_area(
        "请输入用户评论 (Enter user review):",
        height=100,
        placeholder="例如: I love the design but the price is too high.",
        max_chars=config.MAX_TEXT_LENGTH,
        help=f"最大长度: {config.MAX_TEXT_LENGTH} 字符"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🚀 开始分析", use_container_width=True)

    # ============= 推理逻辑 =============
    if analyze_btn:
        if not text_input or not text_input.strip():
            st.warning("⚠️ 请先输入一段文字")
            return

        # 输入验证
        is_valid, cleaned_text, warning = utils.validate_input_text(text_input)

        if not is_valid:
            st.error(f"❌ {warning}")
            return

        if warning:
            st.warning(f"⚠️ {warning}")
            text_input = cleaned_text

        # 进行推理
        with st.spinner("正在调用 BERT 模型进行分析..."):
            try:
                # 预处理
                inputs = tokenizer(
                    text_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.MAX_SEQ_LENGTH,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 推理
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                # 结果映射
                labels = ['Bad 😡', 'Neutral 😐', 'Good 😍']
                pred_idx = probs.argmax()
                pred_label = labels[pred_idx]
                confidence = probs[pred_idx]

                # ============= 结果展示 =============
                st.markdown("---")

                # 关键词提取
                keywords = utils.extract_keywords(text_input)
                if keywords:
                    kw_list = [k[0] for k in keywords]
                    st.info(f"🔍 **用户关注点 (Key Topics):** {', '.join(kw_list)}")
                else:
                    st.info("🔍 **用户关注点:** (句子太短，无法提取)")

                # 第一行：大卡片展示结果
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("情感倾向 (Sentiment)", pred_label)
                with r2:
                    st.metric("置信度 (Confidence)", f"{confidence:.2%}")
                with r3:
                    # 根据结果给出建议
                    if pred_idx == 0:
                        st.error("🚨 建议：优先处理！这是负面反馈，可能包含 Bug 或核心痛点。")
                    elif pred_idx == 1:
                        st.warning("⚠️ 建议：持续关注。用户可能有功能建议或部分不满。")
                    else:
                        st.success("✅ 建议：保持现状。用户对产品非常满意。")

                # 第二行：可视化概率分布
                st.markdown("#### 📊 模型概率分布 (Model Probability)")

                prob_df = pd.DataFrame({
                    'Sentiment': ['Bad', 'Neutral', 'Good'],
                    'Probability': probs
                })

                # 使用 Altair 画柱状图
                chart = alt.Chart(prob_df).mark_bar().encode(
                    x=alt.X('Sentiment', sort=None),
                    y=alt.Y('Probability', axis=alt.Axis(format='%')),
                    color=alt.Color(
                        'Sentiment',
                        scale=alt.Scale(
                            domain=['Bad', 'Neutral', 'Good'],
                            range=[
                                config.SENTIMENT_COLORS['Bad'],
                                config.SENTIMENT_COLORS['Neutral'],
                                config.SENTIMENT_COLORS['Good']
                            ]
                        ),
                        legend=None
                    ),
                    tooltip=['Sentiment', alt.Tooltip('Probability', format='.2%')]
                ).properties(height=300)

                st.altair_chart(chart, use_container_width=True)

                # 详细概率信息
                with st.expander("📋 查看详细概率"):
                    for sent, prob in zip(['Bad', 'Neutral', 'Good'], probs):
                        st.text(f"{sent:10s}: {prob:.4f} ({prob*100:.2f}%)")

            except Exception as e:
                st.error(f"❌ 推理失败: {e}")
                st.exception(e)


# ============= 批量分析功能 =============
def render_batch_analysis(tokenizer, model, device):
    """
    渲染批量分析功能

    Args:
        tokenizer: BERT tokenizer
        model: 训练好的模型
        device: 运行设备
    """
    st.markdown("---")
    st.markdown("### 📁 批量分析 (Batch Analysis)")

    uploaded_file = st.file_uploader(
        "上传 CSV 文件（需包含 'text' 列）",
        type=['csv'],
        help="CSV 文件必须包含名为 'text' 的列"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if 'text' not in df.columns:
                st.error("❌ CSV 文件必须包含 'text' 列")
                return

            st.info(f"📊 上传了 {len(df)} 条记录")

            if st.button("🚀 开始批量分析"):
                with st.spinner(f"正在分析 {len(df)} 条记录..."):
                    predictions = []
                    confidences = []

                    progress_bar = st.progress(0)

                    for idx, text in enumerate(df['text']):
                        # 验证输入
                        is_valid, cleaned_text, _ = utils.validate_input_text(str(text))
                        if not is_valid:
                            predictions.append('Error')
                            confidences.append(0.0)
                            continue

                        # 推理
                        inputs = tokenizer(
                            cleaned_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=config.MAX_SEQ_LENGTH,
                            padding=True
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                        pred_idx = probs.argmax()
                        predictions.append(list(config.LABEL_MAP.keys())[pred_idx])
                        confidences.append(probs[pred_idx])

                        # 更新进度
                        progress_bar.progress((idx + 1) / len(df))

                    # 添加结果到 dataframe
                    df['predicted_sentiment'] = predictions
                    df['confidence'] = confidences

                    st.success("✅ 批量分析完成！")

                    # 显示结果
                    st.dataframe(df)

                    # 统计信息
                    st.markdown("#### 📊 统计结果")
                    col1, col2, col3 = st.columns(3)

                    sentiment_counts = pd.Series(predictions).value_counts()
                    with col1:
                        st.metric("Bad", sentiment_counts.get('bad', 0))
                    with col2:
                        st.metric("Neutral", sentiment_counts.get('neutral', 0))
                    with col3:
                        st.metric("Good", sentiment_counts.get('good', 0))

                    # 下载结果
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 下载结果 CSV",
                        csv,
                        "sentiment_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )

        except Exception as e:
            st.error(f"❌ 处理文件失败: {e}")


# ============= 主函数 =============
def main():
    """主函数"""
    # 渲染侧边栏
    render_sidebar()

    # 加载模型
    tokenizer, model, device = load_model()

    # 如果模型加载失败，停止运行
    if tokenizer is None or model is None:
        st.stop()

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["🔍 单条分析", "📁 批量分析", "📖 使用说明"])

    with tab1:
        render_main_interface(tokenizer, model, device)

    with tab2:
        render_batch_analysis(tokenizer, model, device)

    with tab3:
        st.markdown("""
        ## 📖 使用说明

        ### 单条分析
        1. 在文本框中输入用户评论
        2. 点击"开始分析"按钮
        3. 查看情感分类结果和置信度
        4. 根据建议采取相应行动

        ### 批量分析
        1. 准备包含 'text' 列的 CSV 文件
        2. 上传文件
        3. 点击"开始批量分析"
        4. 下载分析结果

        ### 情感分类说明
        - **Bad (负面)**: 用户不满、投诉、批评
        - **Neutral (中性)**: 客观陈述、功能建议
        - **Good (正面)**: 用户赞扬、满意表达

        ### 注意事项
        - 输入文本最大长度: {0} 字符
        - 支持中英文混合输入
        - 模型基于训练数据，可能存在偏差
        """.format(config.MAX_TEXT_LENGTH))

        st.markdown("---")
        st.markdown("""
        ### 🛠️ 技术栈
        - **模型**: DistilBERT (Hugging Face)
        - **框架**: PyTorch, Transformers
        - **前端**: Streamlit
        - **数据处理**: Pandas, NumPy
        - **可视化**: Altair

        ### 📞 支持
        如有问题，请联系开发团队或查看项目 README。
        """)


if __name__ == '__main__':
    main()
