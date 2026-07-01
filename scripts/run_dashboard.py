#!/usr/bin/env python3
"""
启动 Streamlit 监控面板
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import datetime
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit plotly")
    sys.exit(1)


def main():
    """主函数"""
    st.set_page_config(
        page_title="足球滚球预测系统",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ 足球滚球预测系统 - 监控面板")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.header("导航")
    page = st.sidebar.selectbox(
        "选择页面",
        ["概览", "Live 比赛", "预测分析", "资金管理", "设置"]
    )
    
    # 概览页面
    if page == "概览":
        show_overview()
    elif page == "Live 比赛":
        show_live_matches()
    elif page == "预测分析":
        show_predictions()
    elif page == "资金管理":
        show_bankroll()
    elif page == "设置":
        show_settings()


def show_overview():
    """显示概览信息"""
    st.header("📊 系统概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Bankroll", "$1,000.00", "+5.2%")
    
    with col2:
        st.metric("Total Bets", "24", "-2")
    
    with col3:
        st.metric("Win Rate", "58.3%", "+2.1%")
    
    with col4:
        st.metric("ROI", "+12.5%", "+1.8%")
    
    st.markdown("---")
    
    # 最近活动
    st.subheader("🕐 最近活动")
    
    activity_data = pd.DataFrame({
        "时间": ["10:30", "10:15", "09:45", "09:30"],
        "类型": ["赢利", "损失", "赢利", "下注"],
        "金额": ["+$52.00", "-$20.00", "+$38.50", "-$25.00"],
        "比赛": ["Man Utd vs Chelsea", "Arsenal vs Liverpool", "Real Madrid vs Barcelona", "Bayern vs Dortmund"]
    })
    
    st.dataframe(activity_data, use_container_width=True)


def show_live_matches():
    """显示 live 比赛"""
    st.header("🔴 Live 比赛")
    
    st.info("暂无 live 比赛数据。请连接数据源 API。")
    
    # 示例比赛卡片
    st.subheader("示例比赛")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Man Utd vs Chelsea
        **比分**: 1-1  
        **时间**: 67'  
        **控球率**: 52% - 48%  
        **射门**: 8 - 6  
        **射正**: 3 - 2
        """)
        
        st.success("**预测**: 下一球 - Man Utd (45%)")
    
    with col2:
        st.markdown("""
        ### Arsenal vs Liverpool
        **比分**: 2-0  
        **时间**: 78'  
        **控球率**: 45% - 55%  
        **射门**: 5 - 12  
        **射正**: 3 - 4
        """)
        
        st.warning("**预测**: 下一球 - Liverpool (52%)")


def show_predictions():
    """显示预测分析"""
    st.header("🔮 预测分析")
    
    st.subheader("模型性能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 进球预测模型
        - 准确率：62.5%
        - AUC: 0.68
        - 训练样本：15,000
        """)
    
    with col2:
        st.markdown("""
        #### 比赛结果预测
        - 准确率：54.2%
        - AUC: 0.61
        - 训练样本：20,000
        """)
    
    st.markdown("---")
    
    st.subheader("实时预测")
    
    pred_data = pd.DataFrame({
        "比赛": ["Man Utd vs Chelsea", "Arsenal vs Liverpool"],
        "主胜概率": ["42%", "28%"],
        "平局概率": ["31%", "25%"],
        "客胜概率": ["27%", "47%"],
        "Over 2.5": ["58%", "42%"],
        "建议": ["主胜", "客胜"]
    })
    
    st.dataframe(pred_data, use_container_width=True)


def show_bankroll():
    """显示资金管理"""
    st.header("💰 资金管理")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("当前余额", "$1,125.00")
    
    with col2:
        st.metric("初始资金", "$1,000.00")
    
    with col3:
        st.metric("总盈利", "+$125.00")
    
    st.markdown("---")
    
    # 资金曲线图
    st.subheader("资金曲线")
    
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    balance = 1000 + np.cumsum(np.random.randn(30) * 20)
    balance = np.maximum(balance, 100)  # 确保不低于 100
    
    chart_data = pd.DataFrame({"日期": dates, "余额": balance})
    st.line_chart(chart_data.set_index("日期"))
    
    st.markdown("---")
    
    st.subheader("投注统计")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("总投注数", "24")
    
    with stats_col2:
        st.metric("赢利", "14")
    
    with stats_col3:
        st.metric("损失", "10")
    
    with stats_col4:
        st.metric("胜率", "58.3%")


def show_settings():
    """显示设置"""
    st.header("⚙️ 设置")
    
    st.subheader("API 配置")
    
    api_url = st.text_input("数据 API URL", "https://api-football.com")
    api_key = st.text_input("API Key", type="password")
    
    st.subheader("投注策略")
    
    strategy = st.selectbox(
        "选择策略",
        ["价值投注", "保守投注", "激进投注"]
    )
    
    max_stake = st.slider("最大投注比例 (%)", 1, 10, 5)
    min_confidence = st.slider("最小置信度 (%)", 30, 80, 50)
    
    st.subheader("通知设置")
    
    email_notifications = st.checkbox("邮件通知", value=False)
    push_notifications = st.checkbox("推送通知", value=True)
    
    if st.button("保存设置"):
        st.success("设置已保存！")


if __name__ == "__main__":
    main()
