import sys
sys.dont_write_bytecode = True  # 禁止 .pyc 缓存

for mod in list(sys.modules.keys()):
    if mod.startswith('src.') or mod == 'app_streamlit':
        del sys.modules[mod]

import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
import json
import traceback
import pandas as pd
import time

# ---- 初始化 ----
root_path = Path("data/stock_data")
pipeline = Pipeline(root_path, run_config=max_config)

@st.cache_data
def load_subset_mapping():
    subset_path = root_path / "subset.csv"
    try:
        df = pd.read_csv(subset_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(subset_path, encoding='gbk')
    return dict(zip(df['sha1'], df['file_name']))

sha1_to_filename = load_subset_mapping()

# ---- Session State ----
if 'history' not in st.session_state:
    st.session_state.history = []  # [{question, answer_dict, timestamp, elapsed}]
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = None

# ---- 页面配置 ----
st.set_page_config(page_title="RAG 年报问答", layout="wide")

# ---- 顶部标题 ----
st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 24px; border-radius: 12px; text-align: center; margin-bottom: 20px;'>
    <h2 style='color: white; margin: 0;'>📊 年报智能问答系统</h2>
    <div style='color: #e0d5ff; font-size: 14px; margin-top: 6px;'>向量检索 + LLM 推理 + 重排 | 中芯国际 2024 年度报告</div>
</div>
""", unsafe_allow_html=True)

# ---- 左侧栏 ----
with st.sidebar:
    st.header("🔍 查询设置")
    user_question = st.text_area(
        "输入问题",
        "中芯国际在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？",
        height=90,
        placeholder="请输入关于中芯国际的问题..."
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_btn = st.button("🚀 生成答案", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ 清除历史", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.session_state.selected_idx = None
        st.rerun()

    # 历史记录
    st.markdown("---")
    st.markdown("### 📝 历史记录")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            q_preview = h['question'][:30] + ("..." if len(h['question']) > 30 else "")
            ts = h.get('timestamp', '')
            if st.button(f"⏱ {ts}\n{q_preview}", key=f"hist_{idx}", use_container_width=True):
                st.session_state.selected_idx = idx
                st.rerun()
    else:
        st.caption("暂无历史记录")

# ---- 主区域 ----
# 如果点选了历史记录，显示历史答案
if st.session_state.selected_idx is not None:
    h = st.session_state.history[st.session_state.selected_idx]
    answer_dict = h['answer_dict']
    elapsed = h.get('elapsed', 0)
    st.caption(f"⏱ 耗时 {elapsed:.1f} 秒 | 📅 {h.get('timestamp', '')}")

    step_by_step = answer_dict.get("step_by_step_analysis", "-")
    reasoning_summary = answer_dict.get("reasoning_summary", "-")
    final_answer = answer_dict.get("final_answer", "-")
    references = answer_dict.get("references", [])

    with st.expander("📋 分步推理", expanded=False):
        st.info(step_by_step)
    with st.expander("💡 推理摘要", expanded=False):
        st.success(reasoning_summary)
    with st.expander("📄 参考页面", expanded=False):
        seen = set()
        for ref in references:
            sha1 = ref.get("pdf_sha1", "")
            page = ref.get("page_index", 1)
            fname = sha1_to_filename.get(sha1, sha1)
            display_name = fname.replace('【财报】', '').replace('.pdf', '')
            key = (display_name, page)
            if key not in seen:
                seen.add(key)
                st.markdown(f"- {display_name}，第 {page} 页")

    st.markdown("### 📌 最终答案")
    st.markdown(f"""
    <div style='background:#f6f8fa;padding:20px;border-radius:10px;border-left:4px solid #7b2ff2;font-size:16px;line-height:1.8;'>
    {final_answer}
    </div>
    """, unsafe_allow_html=True)
    st.button("📋 复制答案", key="copy_hist", help="点击后手动 Ctrl+C 复制", use_container_width=True)

# 新提问
if submit_btn and user_question.strip():
    with st.spinner("正在检索并生成答案，请稍候..."):
        t_start = time.time()
        try:
            answer = pipeline.answer_single_question(user_question, kind="string")
            elapsed = time.time() - t_start

            if isinstance(answer, str):
                try:
                    answer_dict = json.loads(answer)
                except Exception:
                    st.error("返回内容无法解析：" + str(answer))
                    answer_dict = {}
            else:
                answer_dict = answer

            # 存入历史
            from datetime import datetime
            st.session_state.history.append({
                'question': user_question,
                'answer_dict': answer_dict,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'elapsed': elapsed
            })
            st.session_state.selected_idx = len(st.session_state.history) - 1

            # 显示结果
            step_by_step = answer_dict.get("step_by_step_analysis", "-")
            reasoning_summary = answer_dict.get("reasoning_summary", "-")
            final_answer = answer_dict.get("final_answer", "-")
            references = answer_dict.get("references", [])

            st.caption(f"⏱ 耗时 {elapsed:.1f} 秒 | 📅 {st.session_state.history[-1]['timestamp']}")

            with st.expander("📋 分步推理", expanded=True):
                st.info(step_by_step)
            with st.expander("💡 推理摘要", expanded=True):
                st.success(reasoning_summary)
            with st.expander("📄 参考页面", expanded=False):
                seen = set()
                for ref in references:
                    sha1 = ref.get("pdf_sha1", "")
                    page = ref.get("page_index", 1)
                    fname = sha1_to_filename.get(sha1, sha1)
                    display_name = fname.replace('【财报】', '').replace('.pdf', '')
                    key = (display_name, page)
                    if key not in seen:
                        seen.add(key)
                        st.markdown(f"- {display_name}，第 {page} 页")
                if not references:
                    pages = answer_dict.get("relevant_pages", [])
                    for p in pages:
                        st.markdown(f"- 第 {p} 页")

            st.markdown("### 📌 最终答案")
            st.markdown(f"""
            <div style='background:#f6f8fa;padding:20px;border-radius:10px;border-left:4px solid #7b2ff2;font-size:16px;line-height:1.8;'>
            {final_answer}
            </div>
            """, unsafe_allow_html=True)

            # 复制按钮（用 text_area 实现可复制的区域）
            st.text_area("📋 复制答案", value=final_answer, height=120, key=f"copy_{len(st.session_state.history)}")

        except Exception as e:
            tb = traceback.format_exc()
            st.error(f"❌ 生成答案时出错: {e}")
            with st.expander("查看详细错误追踪"):
                st.code(tb)

else:
    if st.session_state.selected_idx is None:
        st.info("👈 请在左侧输入问题并点击「生成答案」，或查看历史记录")
