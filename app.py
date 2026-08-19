import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(
    page_title="부산 은퇴자 거주지 추천 AI 대시보드", layout="wide"
)

st.sidebar.title("📌 메뉴 선택")
menu = st.sidebar.radio(
    "분석 수준 선택",
    ["16개 구·군 광역 분석", "구별 아파트 단지 AI 맞춤 추천"],
)

# -------------------------------------------------------------------
# 1. 16개 구·군 광역 분석
# -------------------------------------------------------------------
if menu == "16개 구·군 광역 분석":
  st.title("🏆 부산광역시 16개 구·군 은퇴자 최적 거주지 추천 대시보드")
  st.markdown(
      "사용자가 선호하는 가중치를 직접 조절하면 실시간으로 최적 구·군"
      " 순위가 업데이트됩니다."
  )

  st.sidebar.header("⚙️ 4대 인프라 가중치 조절")
  w_price = st.sidebar.slider("부동산 가성비 (%)", 0, 100, 30)
  w_size = st.sidebar.slider("평형 적합도 (%)", 0, 100, 20)
  w_med = st.sidebar.slider("의료 인프라 (%)", 0, 100, 30)
  w_trans = st.sidebar.slider("교통/환경 (%)", 0, 100, 20)

  total = w_price + w_size + w_med + w_trans
  total = 1 if total == 0 else total

  try:
    df = pd.read_csv("busan_retiree_recommendation_advanced.csv")
    df["실시간_종합점수"] = (
        df["가성비점수"] * (w_price / total)
        + df["평형적합도점수"] * (w_size / total)
        + df["의료점수"] * (w_med / total)
        + df["교통환경점수"] * (w_trans / total)
    ).round(1)

    df_sorted = df.sort_values(by="실시간_종합점수", ascending=False).reset_index(
        drop=True
    )
    df_sorted["순위"] = df_sorted.index + 1

    col1, col2 = st.columns([6, 4])
    with col1:
      st.subheader("📊 실시간 종합 순위 Top 10")
      st.dataframe(
          df_sorted[[
              "순위",
              "구이름",
              "실시간_종합점수",
              "가성비점수",
              "의료점수",
              "교통점수",
              "환경점수",
          ]].head(10),
          use_container_width=True,
      )
    with col2:
      st.subheader("📈 구·군별 종합 점수 시각화")
      st.bar_chart(df_sorted.set_index("구이름")["실시간_종합점수"])
  except Exception as e:
    st.error("데이터 파일(busan_retiree_recommendation_advanced.csv) 로딩 실패.")

# -------------------------------------------------------------------
# 2. 구별 아파트 단지 AI 맞춤 추천 (상세 필터링 & 지도 시각화 연동)
# -------------------------------------------------------------------
else:
  st.title("🎯 부산시 구별 아파트 단지 AI 맞춤 추천 & 지도 시각화")

  # 구별 데이터베이스 (건축연도, 위도 lat, 경도 lon 포함)
  gu_database = {
      "부산진구": {
          "단지명": [
              "개금동 반도보라",
              "전포동 대동파크",
              "부암동 화승삼성",
              "가야동 가야벽산",
              "양정동 현대아파트",
              "당감동 백양순환",
          ],
          "법정동": ["개금동", "전포동", "부암동", "가야동", "양정동", "당감동"],
          "매매가_억": [2.3, 2.8, 3.1, 2.1, 2.7, 1.8],
          "전용면적_m2": [84.9, 75.2, 84.8, 84.6, 79.8, 59.9],
          "건축연도": [1999, 1997, 1997, 1996, 1998, 1995],
          "의료접근성_점수": [90, 85, 95, 75, 90, 70],
          "지하철역_거리_m": [450, 300, 600, 500, 400, 850],
          "lat": [35.1525, 35.1568, 35.1630, 35.1552, 35.1721, 35.1685],
          "lon": [129.0210, 129.0652, 129.0520, 129.0345, 129.0712, 129.0390],
      },
      "해운대구": {
          "단지명": [
              "좌동 해운대대림",
              "우동 삼호가든",
              "재송동 더샵센텀파크",
              "좌동 화목데파트",
              "반여동 아시아선수촌",
              "우동 해운대자이",
          ],
          "법정동": ["좌동", "우동", "재송동", "좌동", "반여동", "우동"],
          "매매가_억": [3.2, 5.8, 6.5, 2.9, 3.8, 7.2],
          "전용면적_m2": [84.9, 84.7, 84.8, 59.8, 84.9, 84.9],
          "건축연도": [1997, 1985, 2005, 1996, 2002, 2013],
          "의료접근성_점수": [95, 90, 88, 92, 75, 92],
          "지하철역_거리_m": [350, 250, 700, 200, 900, 300],
          "lat": [35.1701, 35.1662, 35.1745, 35.1689, 35.2012, 35.1635],
          "lon": [129.1760, 129.1360, 129.1285, 129.1725, 129.1290, 129.1412],
      },
      "사하구": {
          "단지명": [
              "하단동 가락타운",
              "신평동 삼익아파트",
              "괴정동 한신아파트",
              "다대동 몰운대아파트",
              "당리동 동원베네스트",
          ],
          "법정동": ["하단동", "신평동", "괴정동", "다대동", "당리동"],
          "매매가_억": [2.4, 1.7, 2.2, 1.5, 2.9],
          "전용면적_m2": [84.8, 72.5, 84.9, 59.9, 84.7],
          "건축연도": [1992, 1990, 1995, 1996, 2006],
          "의료접근성_점수": [80, 70, 85, 65, 78],
          "지하철역_거리_m": [400, 300, 350, 950, 450],
          "lat": [35.1050, 35.0920, 35.1012, 35.0510, 35.1085],
          "lon": [128.9610, 128.9680, 128.9912, 128.9670, 128.9750],
      },
  }

  selected_gu = st.selectbox(
      "📍 분석 대상 구·군 선택", list(gu_database.keys())
  )

  if selected_gu in gu_database:
    # -----------------------------------------------
    # 고도화 1: 상세 필터링 로직 (예산, 평형, 건축연한)
    # -----------------------------------------------
    st.sidebar.subheader("🔍 상세 필터링 조건")
    user_budget = st.sidebar.number_input(
        "최대 보유 예산 (억 원)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
    )
    min_area, max_area = st.sidebar.slider(
        "희망 전용면적 Range (㎡)", 40, 120, (50, 100)
    )
    min_year = st.sidebar.slider("최소 준공연도 (건축연한)", 1980, 2025, 1990)

    st.sidebar.subheader("⚙️ AI 추천 선호 가중치")
    w_c_price = st.sidebar.slider("가성비 중요도", 1, 5, 4)
    w_c_med = st.sidebar.slider("의료 접근성 중요도", 1, 5, 4)
    w_c_trans = st.sidebar.slider("교통 역세권 중요도", 1, 5, 3)

    df_c = pd.DataFrame(gu_database[selected_gu])
    df_c["공급평형"] = (df_c["전용면적_m2"] / 3.3058 * 1.3).round(1)
    df_c["평당가격_만원"] = (
        (df_c["매매가_억"] * 10000) / df_c["공급평형"]
    ).round(0)

    # 조건별 필터링
    df_filtered = df_c[
        (df_c["매매가_억"] <= user_budget)
        & (df_c["전용면적_m2"] >= min_area)
        & (df_c["전용면적_m2"] <= max_area)
        & (df_c["건축연도"] >= min_year)
    ].reset_index(drop=True)

    if len(df_filtered) > 0:
      max_price = max(6.0, df_c["매매가_억"].max() + 0.5)
      min_price = max(0.5, df_c["매매가_억"].min() - 0.5)
      df_filtered["가성비_vector"] = (
          (max_price - df_filtered["매매가_억"]) / (max_price - min_price) * 100
      )
      df_filtered["의료_vector"] = df_filtered["의료접근성_점수"]
      df_filtered["교통_vector"] = (
          (1500 - df_filtered["지하철역_거리_m"]) / (1500 - 100) * 100
      )

      X_filtered = df_filtered[
          ["가성비_vector", "의료_vector", "교통_vector"]
      ].values
      user_vec = np.array(
          [w_c_price * 20, w_c_med * 20, w_c_trans * 20]
      ).reshape(1, -1)

      sims = cosine_similarity(user_vec, X_filtered)[0]
      df_filtered["AI_유사도점수"] = (sims * 100).round(1)
      df_result = df_filtered.sort_values(
          by="AI_유사도점수", ascending=False
      )

      st.subheader(
          f"💡 [{selected_gu}] 조건 맞춤 AI 추천 단지 리스트 (총"
          f" {len(df_result)}개)"
      )
      st.dataframe(
          df_result[[
              "단지명",
              "법정동",
              "매매가_억",
              "평당가격_만원",
              "전용면적_m2",
              "건축연도",
              "의료접근성_점수",
              "지하철역_거리_m",
              "AI_유사도점수",
          ]],
          use_container_width=True,
      )

      # -----------------------------------------------
      # 고도화 2: Pydeck 활용 지도 시각화 및 마커 핀
      # -----------------------------------------------
      st.subheader("🗺️ 추천 단지 지도 위치 및 인터랙티브 핀 마커")

      # Pydeck 레이어 설정 (빨간색 마커)
      layer = pdk.Layer(
          "ScatterplotLayer",
          df_result,
          get_position=["lon", "lat"],
          get_color=[255, 75, 75, 200],
          get_radius=150,
          pickable=True,
      )

      # 중심 좌표 설정
      view_state = pdk.ViewState(
          latitude=df_result["lat"].mean(),
          longitude=df_result["lon"].mean(),
          zoom=12,
          pitch=30,
      )

      # 지도 렌더링 (마커 툴팁 포함)
      r = pdk.Deck(
          layers=[layer],
          initial_view_state=view_state,
          tooltip={
              "text": (
                  "🏢 단지명: {단지명}\n💰 매매가: {매매가_억}억\n🏥 의료점수:"
                  " {의료접근성_점수}점\n🚇 지하철: {지하철역_거리_m}m"
              )
          },
      )
      st.pydeck_chart(r)

    else:
      st.warning(
          "설정하신 필터링 조건(예산, 평형, 준공연도)에 해당하는 아파트 단지가"
          " 없습니다. 사이드바 필터를 변경해 보세요."
      )