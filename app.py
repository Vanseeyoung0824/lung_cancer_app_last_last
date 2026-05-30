import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ────────────────────────────────────────────────
def set_korean_font():
    local_font = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
    if os.path.exists(local_font):
        fm.fontManager.addfont(local_font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=local_font).get_name()
        plt.rcParams["axes.unicode_minus"] = False
        return
    for path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                 "/usr/share/fonts/nanum/NanumGothic.ttf"]:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    korean_fonts = ["NanumGothic", "Malgun Gothic", "AppleGothic", "UnDotum"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in korean_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return

set_korean_font()

# ── 페이지 설정 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="폐암 환자 군집 분석 시스템",
    page_icon="🫁",
    layout="wide",
)

# ── 전체 CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

/* 전체 배경 */
.stApp {
    background-color: #f0f4f8;
    font-family: 'Noto Sans KR', sans-serif;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1f44 0%, #0d2d6b 60%, #102f7a 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #e8edf5 !important;
}
[data-testid="stSidebar"] .stNumberInput label {
    color: #a0b4cc !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #000000 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #4a9eff !important;
    box-shadow: 0 0 0 2px rgba(74,158,255,0.2) !important;
}

/* 메인 컨텐츠 패딩 */
.main .block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* 헤더 배너 */
.med-header {
    background: linear-gradient(135deg, #0a1f44 0%, #1a3a7a 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(10,31,68,0.18);
    border-left: 5px solid #4a9eff;
}
.med-header-left h1 {
    font-family: 'DM Serif Display', serif;
    color: #ffffff;
    font-size: 1.9rem;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}
.med-header-left p {
    color: #7fa8d4;
    font-size: 0.85rem;
    margin: 0;
    letter-spacing: 0.04em;
}
.med-badge {
    background: rgba(74,158,255,0.15);
    border: 1px solid rgba(74,158,255,0.35);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    text-align: center;
}
.med-badge span {
    display: block;
    color: #4a9eff;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
}
.med-badge strong {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 600;
}

/* 카드 공통 */
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(10,31,68,0.07);
    border: 1px solid #dde6f0;
    height: 100%;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a7a9a;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid #e8f0f8;
    padding-bottom: 0.5rem;
}

/* 결과 박스 */
.result-panel {
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    border-left: 6px solid;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.result-panel.safe {
    background: linear-gradient(135deg, #f0fdf4, #e8f8f0);
    border-color: #16a34a;
}
.result-panel.danger {
    background: linear-gradient(135deg, #fff1f2, #fee2e2);
    border-color: #dc2626;
}
.result-panel.caution {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-color: #d97706;
}
.result-panel .rp-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.3rem;
}
.result-panel .rp-cluster {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
    margin-bottom: 0.2rem;
}
.result-panel .rp-desc {
    font-size: 0.82rem;
    opacity: 0.75;
    line-height: 1.5;
}

/* 통계 수치 카드 */
.stat-val {
    font-size: 2rem;
    font-weight: 700;
    color: #0a1f44;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.stat-label {
    font-size: 0.75rem;
    color: #7a9ab8;
    font-weight: 500;
}

/* 분석 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #0a1f44, #1a3a7a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(10,31,68,0.3) !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a3a7a, #2952a0) !important;
    box-shadow: 0 6px 20px rgba(10,31,68,0.4) !important;
    transform: translateY(-1px) !important;
}

/* 사이드바 구분선 */
.sidebar-section {
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 1.2rem;
    margin-bottom: 1.2rem;
}
.sidebar-title {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a8ab0 !important;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

/* 테이블 */
.stDataFrame {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* 숫자 입력 버튼 숨기기 */
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: white !important;
}

div[data-testid="stNumberInput"] > div {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ── 데이터 & 모델 ─────────────────────────────────────────────────
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    c0 = pd.DataFrame({
        "나이": np.random.normal(35, 5, 30).clip(20, 55),
        "흡연량": np.random.normal(3, 2, 30).clip(0, 8),
        "음주량": np.random.normal(1, 1, 30).clip(0, 4),
    })
    c1 = pd.DataFrame({
        "나이": np.random.normal(60, 7, 30).clip(40, 80),
        "흡연량": np.random.normal(20, 5, 30).clip(10, 35),
        "음주량": np.random.normal(5, 2, 30).clip(2, 10),
    })
    c2 = pd.DataFrame({
        "나이": np.random.normal(48, 6, 30).clip(30, 65),
        "흡연량": np.random.normal(10, 3, 30).clip(5, 20),
        "음주량": np.random.normal(3, 1, 30).clip(1, 6),
    })
    return pd.concat([c0, c1, c2], ignore_index=True)

@st.cache_resource
def train_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["나이", "흡연량", "음주량"]])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    cluster_means = pd.DataFrame({
        "cluster": range(3),
        "mean_smoking": [df[labels == i]["흡연량"].mean() for i in range(3)],
    }).sort_values("mean_smoking")
    rank_map = {
        cluster_means.iloc[0]["cluster"]: 0,
        cluster_means.iloc[1]["cluster"]: 2,
        cluster_means.iloc[2]["cluster"]: 1,
    }
    remapped = np.array([rank_map[l] for l in labels])
    return kmeans, scaler, remapped, rank_map

def predict_cluster(kmeans, scaler, rank_map, age, smoke, drink):
    raw = kmeans.predict(scaler.transform([[age, smoke, drink]]))[0]
    return rank_map[raw]

df = generate_sample_data()
kmeans, scaler, labels, rank_map = train_model(df)

CLUSTER_INFO = {
    0: {"name": "매우 건강군",  "eng": "Very Healthy",  "panel": "safe",    "emoji": "●", "color": "#16a34a",
        "desc": "흡연·음주량이 낮고 연령대가 젊습니다. 현재 건강 습관을 유지하세요.",
        "action": "정기 건강검진 유지"},
    1: {"name": "위험군",       "eng": "High Risk",     "panel": "danger",  "emoji": "▲", "color": "#dc2626",
        "desc": "높은 흡연량과 중장년 연령대로 폐암 위험도가 가장 높습니다.",
        "action": "즉각적인 금연·절주 및 정밀 검진 권장"},
    2: {"name": "건강군",       "eng": "Healthy",       "panel": "caution", "emoji": "◆", "color": "#d97706",
        "desc": "보통 수준의 흡연·음주량을 보입니다. 생활습관 개선이 권장됩니다.",
        "action": "정기 검진 및 생활습관 개선 권장"},
}
COLORS = {0: "#16a34a", 1: "#dc2626", 2: "#d97706"}


# ── 산점도 ────────────────────────────────────────────────────────
def draw_scatter(df, labels, patient_smoke, patient_drink):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    names = {0: "매우 건강군 (Cluster 0)", 1: "위험군 (Cluster 1)", 2: "건강군 (Cluster 2)"}
    for cid, color in COLORS.items():
        mask = labels == cid
        ax.scatter(df.loc[mask, "흡연량"], df.loc[mask, "음주량"],
                   c=color, alpha=0.65, s=55, zorder=2, label=names[cid],
                   edgecolors="white", linewidths=0.5)

    ax.scatter(patient_smoke, patient_drink,
               marker="*", s=320, c="#0a1f44", zorder=5,
               label="현재 환자", edgecolors="white", linewidths=0.8)

    ax.set_xlabel("흡연량 (갑/년)", fontsize=9, color="#4a6a8a")
    ax.set_ylabel("음주량", fontsize=9, color="#4a6a8a")
    ax.set_title("환자 군집 분포도", fontsize=11, fontweight="bold", color="#0a1f44", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dde6f0")
    ax.spines["bottom"].set_color("#dde6f0")
    ax.tick_params(colors="#7a9ab8", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5, color="#dde6f0")

    legend = ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95,
                       edgecolor="#dde6f0", fancybox=False)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    return fig


# ── 사이드바 ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.2rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1.2rem;">
        <div style="font-size:0.6rem; letter-spacing:0.18em; color:#4a7aaa; font-weight:700; text-transform:uppercase; margin-bottom:0.4rem;">
            PULMONARY ONCOLOGY
        </div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff; font-family:'DM Serif Display',serif; line-height:1.3;">
            폐암 군집<br>분석 시스템
        </div>
        <div style="font-size:0.7rem; color:#5a8ab0; margin-top:0.4rem;">K-Means Clustering v2.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">📋 환자 정보 입력</div>', unsafe_allow_html=True)
    age   = st.number_input("나이 (세)", min_value=0.0, max_value=120.0, value=50.0, step=1.0, format="%.1f")
    smoke = st.number_input("흡연량 (갑/년)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, format="%.1f")
    drink = st.number_input("음주량 (잔/주)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, format="%.1f")

    analyze = st.button("🔬 군집 분석 실행", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.08);">
        <div class="sidebar-title" style="color:#4a7aaa!important;">군집 분류 기준</div>
        <div style="font-size:0.78rem; line-height:2;">
            <span style="color:#16a34a; font-weight:700;">● 0</span>
            <span style="color:#a0b4cc;"> 매우 건강군</span><br>
            <span style="color:#dc2626; font-weight:700;">▲ 1</span>
            <span style="color:#a0b4cc;"> 위험군</span><br>
            <span style="color:#d97706; font-weight:700;">◆ 2</span>
            <span style="color:#a0b4cc;"> 건강군</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:auto; padding-top:2rem;">
        <div style="font-size:0.62rem; color:#3a5a7a; line-height:1.6;">
            본 시스템은 연구·교육 목적으로<br>
            제작되었습니다. 임상 진단을<br>
            대체하지 않습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 메인 영역 ─────────────────────────────────────────────────────
st.markdown("""
<div class="med-header">
    <div class="med-header-left">
        <h1>🫁 폐암 환자 군집 분석 시스템</h1>
        <p>LUNG CANCER PATIENT CLUSTER ANALYSIS SYSTEM &nbsp;|&nbsp; AI-POWERED DIAGNOSTIC SUPPORT</p>
    </div>
    <div style="display:flex; gap:1rem;">
        <div class="med-badge">
            <span>모델</span>
            <strong>K-Means</strong>
        </div>
        <div class="med-badge">
            <span>군집 수</span>
            <strong>3 Groups</strong>
        </div>
        <div class="med-badge">
            <span>학습 데이터</span>
            <strong>90 Cases</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 데이터 요약 카드 (상단) ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
group_counts = {i: int((labels == i).sum()) for i in range(3)}

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">전체 학습 데이터</div>
        <div class="stat-val">90</div>
        <div class="stat-label">Total Cases</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title" style="color:#16a34a;">매우 건강군 (0)</div>
        <div class="stat-val" style="color:#16a34a;">{group_counts[0]}</div>
        <div class="stat-label">Very Healthy Cases</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title" style="color:#dc2626;">위험군 (1)</div>
        <div class="stat-val" style="color:#dc2626;">{group_counts[1]}</div>
        <div class="stat-label">High Risk Cases</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="card">
        <div class="card-title" style="color:#d97706;">건강군 (2)</div>
        <div class="stat-val" style="color:#d97706;">{group_counts[2]}</div>
        <div class="stat-label">Healthy Cases</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 분석 결과 영역 ────────────────────────────────────────────────
if analyze:
    cluster = predict_cluster(kmeans, scaler, rank_map, age, smoke, drink)
    info = CLUSTER_INFO[cluster]

    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        st.markdown(f"""
        <div class="result-panel {info['panel']}">
            <div class="rp-label">분석 결과 · CLUSTER RESULT</div>
            <div class="rp-cluster" style="color:{info['color']};">
                {info['emoji']} Cluster {cluster} &nbsp;—&nbsp; {info['name']}
            </div>
            <div style="font-size:0.72rem; color:{info['color']}; font-weight:600; letter-spacing:0.06em; margin-bottom:0.8rem;">
                {info['eng'].upper()}
            </div>
            <div class="rp-desc">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # 권고사항
        st.markdown(f"""
        <div class="card" style="margin-bottom:1rem;">
            <div class="card-title">권고사항 · RECOMMENDATION</div>
            <div style="font-size:0.88rem; color:#0a1f44; font-weight:600; padding-top:0.3rem;">
                {info['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 입력값 요약
        st.markdown(f"""
        <div class="card">
            <div class="card-title">입력 데이터 요약 · INPUT SUMMARY</div>
            <table style="width:100%; font-size:0.83rem; border-collapse:collapse; margin-top:0.5rem;">
                <tr style="border-bottom:1px solid #e8f0f8;">
                    <td style="padding:0.4rem 0; color:#5a7a9a; font-weight:500;">나이</td>
                    <td style="padding:0.4rem 0; color:#0a1f44; font-weight:700; text-align:right;">{age:.1f} 세</td>
                </tr>
                <tr style="border-bottom:1px solid #e8f0f8;">
                    <td style="padding:0.4rem 0; color:#5a7a9a; font-weight:500;">흡연량</td>
                    <td style="padding:0.4rem 0; color:#0a1f44; font-weight:700; text-align:right;">{smoke:.1f} 갑/년</td>
                </tr>
                <tr>
                    <td style="padding:0.4rem 0; color:#5a7a9a; font-weight:500;">음주량</td>
                    <td style="padding:0.4rem 0; color:#0a1f44; font-weight:700; text-align:right;">{drink:.1f} 잔/주</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        fig = draw_scatter(df, labels, smoke, drink)
        st.pyplot(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 군집별 상세 테이블
    with st.expander("📊 군집별 통계 상세 보기", expanded=False):
        detail = []
        for cid in range(3):
            mask = labels == cid
            sub = df[mask]
            detail.append({
                "군집": f"Cluster {cid} — {CLUSTER_INFO[cid]['name']}",
                "환자 수": int(mask.sum()),
                "평균 나이": f"{sub['나이'].mean():.1f} 세",
                "평균 흡연량": f"{sub['흡연량'].mean():.1f} 갑/년",
                "평균 음주량": f"{sub['음주량'].mean():.1f} 잔/주",
                "권고사항": CLUSTER_INFO[cid]["action"],
            })
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

else:
    # 분석 전 안내
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 2rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🫁</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#0a1f44; margin-bottom:0.8rem;">
            환자 정보를 입력하고 분석을 시작하세요
        </div>
        <div style="font-size:0.85rem; color:#7a9ab8; line-height:1.7;">
            좌측 패널에서 나이, 흡연량, 음주량을 입력한 뒤<br>
            <strong style="color:#0a1f44;">군집 분석 실행</strong> 버튼을 클릭하면<br>
            AI가 해당 환자의 위험 군집을 분류합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 빈 상태에서도 차트 미리보기
    st.markdown("<br>", unsafe_allow_html=True)
    fig = draw_scatter(df, labels, -999, -999)
    st.pyplot(fig, use_container_width=True)
