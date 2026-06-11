import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai

st.set_page_config(
    page_title="Stock & AI Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Key: Streamlit Secrets에서 로드 ───────────────────
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = None

# ── 사이드바 ──────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
page = st.sidebar.radio("페이지 선택", ["📈 주식 데이터", "🤖 AI 챗봇"])

st.sidebar.markdown("---")
if api_key:
    st.sidebar.success("Gemini API Key 연결됨 ✅")
else:
    st.sidebar.error("Gemini API Key 미설정 ❌")

# ── 주식 데이터 페이지 ────────────────────────────────────
STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "Samsung (KRX)": "005930.KS",
    "SK Hynix (KRX)": "000660.KS",
    "KOSPI ETF": "069500.KS",
}

if page == "📈 주식 데이터":
    st.title("📈 글로벌 주식 데이터")
    st.caption("미국 7대 빅테크 + 국내 대표 종목 10개")

    period_map = {"1주": "5d", "1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_period_label = st.select_slider("조회 기간", options=list(period_map.keys()), value="1개월")
    period = period_map[selected_period_label]

    # 현재가 요약 카드
    st.subheader("현재가 요약")
    with st.spinner("데이터 불러오는 중..."):
        summary_rows = []
        for name, ticker in STOCKS.items():
            try:
                info = yf.Ticker(ticker).fast_info
                price = info.last_price
                prev = info.previous_close
                change = price - prev
                pct = change / prev * 100
                summary_rows.append({
                    "종목": name,
                    "티커": ticker,
                    "현재가": price,
                    "전일대비": change,
                    "등락률(%)": pct,
                })
            except Exception:
                pass

    df_summary = pd.DataFrame(summary_rows)

    # 카드 레이아웃
    cols = st.columns(5)
    for i, row in df_summary.iterrows():
        c = cols[i % 5]
        arrow = "▲" if row["등락률(%)"] >= 0 else "▼"
        color = "green" if row["등락률(%)"] >= 0 else "red"
        c.metric(
            label=f"{row['종목']} ({row['티커']})",
            value=f"{row['현재가']:,.2f}",
            delta=f"{arrow} {abs(row['등락률(%)']):.2f}%",
        )

    st.markdown("---")

    # 등락률 바 차트
    st.subheader("등락률 비교")
    fig_bar = px.bar(
        df_summary,
        x="종목",
        y="등락률(%)",
        color="등락률(%)",
        color_continuous_scale=["red", "lightgrey", "green"],
        color_continuous_midpoint=0,
        text="등락률(%)",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_bar.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # 개별 종목 캔들차트
    st.subheader("개별 종목 차트")
    selected_stock = st.selectbox("종목 선택", list(STOCKS.keys()))
    ticker_sym = STOCKS[selected_stock]

    with st.spinner(f"{selected_stock} 차트 불러오는 중..."):
        hist = yf.Ticker(ticker_sym).history(period=period)

    if hist.empty:
        st.error("데이터를 불러올 수 없습니다.")
    else:
        fig_candle = go.Figure(data=[
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                increasing_line_color="red",
                decreasing_line_color="blue",
            )
        ])
        fig_candle.update_layout(
            title=f"{selected_stock} ({ticker_sym}) — {selected_period_label}",
            xaxis_title="날짜",
            yaxis_title="가격",
            height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # 거래량
        fig_vol = px.bar(hist, x=hist.index, y="Volume", title="거래량", height=250)
        fig_vol.update_layout(xaxis_title="날짜", yaxis_title="거래량")
        st.plotly_chart(fig_vol, use_container_width=True)

        # 원시 데이터 테이블
        with st.expander("원시 데이터 보기"):
            st.dataframe(
                hist[["Open", "High", "Low", "Close", "Volume"]].sort_index(ascending=False),
                use_container_width=True,
            )

# ── AI 챗봇 페이지 ─────────────────────────────────────────
elif page == "🤖 AI 챗봇":
    st.title("🤖 Gemini AI 챗봇")
    st.caption("Google Gemini 2.0 Flash 기반 대화형 AI")

    if not api_key:
        st.info("👈 왼쪽 사이드바에서 Gemini API Key를 먼저 입력해주세요.")
        st.stop()

    # 모델 초기화
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        st.error(f"API 초기화 실패: {e}")
        st.stop()

    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])

    # 히스토리 초기화 버튼
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ 초기화"):
            st.session_state.messages = []
            st.session_state.chat = model.start_chat(history=[])
            st.rerun()

    # 기존 메시지 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 입력창
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    response = st.session_state.chat.send_message(prompt)
                    reply = response.text
                except Exception as e:
                    reply = f"❌ 오류 발생: {e}"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
