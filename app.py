import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd # 데이터 확인용

# --- Scipy 체크 ---
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (자동 시점 모드)")
st.markdown("⚠️ **도형이 자동으로 화면 중앙에 오도록 카메라 고정을 풀었습니다.**")

# --- 사이드바 ---
st.sidebar.header("설정")
category = st.sidebar.radio("도형 카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"])

# 그래프 초기화
fig = go.Figure()
points_df = None # 데이터 확인용 변수

# ========================================================
# 1. 각기둥 / 각뿔 / 각뿔대
# ========================================================
if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    
    # 윗면 반지름 설정
    if sub_type == "각기둥": rt = rb
    elif sub_type == "각뿔": rt = 0
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)

    # 좌표 계산
    theta = np.linspace(0, 2*np.pi, n+1)
    x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
    x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)
    
    # 1. 면(Mesh) 데이터 구성
    x = np.concatenate([x_top, x_bot, [0], [0]])
    y = np.concatenate([y_top, y_bot, [0], [0]])
    z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
    
    i, j, k = [], [], []
    for idx in range(n): # 옆면
        i.extend([idx, idx]); j.extend([n+1+idx, n+1+idx+1]); k.extend([n+1+idx+1, idx+1])
    if rt > 0: # 뚜껑
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
    if rb > 0: # 바닥
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

    # 2. 그래프 추가 (면 + 점)
    # 점을 먼저 그립니다 (빨간색, 큰 점)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'), name='점(Vertex)'))
    # 면을 그립니다 (반투명)
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='cyan', opacity=0.5, name='면(Mesh)'))
    
    # 데이터 디버깅용 저장
    points_df = pd.DataFrame({"X": x, "Y": y, "Z": z})

# ========================================================
# 2. 원기둥 / 원뿔 / 원뿔대
# ========================================================
elif category == "원기둥/원뿔/원뿔대":
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    n = 40 # 원은 40각형
    
    if sub_type == "원기둥": rt = rb
    elif sub_type == "원뿔": rt = 0
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)

    theta = np.linspace(0, 2*np.pi, n+1)
    x = np.concatenate([rt*np.cos(theta), rb*np.cos(theta), [0], [0]])
    y = np.concatenate([rt*np.sin(theta), rb*np.sin(theta), [0], [0]])
    z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
    
    i, j, k = [], [], []
    for idx in range(n):
        i.extend([idx, idx]); j.extend([n+1+idx, n+1+idx+1]); k.extend([n+1+idx+1, idx+1])
    if rt > 0:
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
    if rb > 0:
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, color='red'), name='점'))
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='gold', opacity=0.6, name='면'))
    points_df = pd.DataFrame({"X": x, "Y": y, "Z": z})

# ========================================================
# 3. 정다면체
# ========================================================
elif category == "정다면체":
    if not has_scipy:
        st.error("Scipy 라이브러리가 없습니다. Mesh는 안 보이지만 점은 찍어보겠습니다.")
    
    sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
    size = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
    points = []
    phi = (1 + np.sqrt(5)) / 2

    if sub_type == "정사면체": points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    elif sub_type == "정육면체":
        for x in [-1,1]:
            for y in [-1,1]:
                for z in [-1,1]: points.append([x,y,z])
    elif sub_type == "정팔면체": points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif sub_type == "정십이면체":
        for x in [-1,1]:
            for y in [-1,1]:
                for z in [-1,1]: points.append([x,y,z])
        for i in [-1,1]:
            for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif sub_type == "정이십면체":
        for i in [-1,1]:
            for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

    points = np.array(points) * size
    
    # 1. 점(Scatter) 무조건 찍기
    fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers+text', marker=dict(size=6, color='red')))
    
    # 2. 면(Mesh) 시도
    if has_scipy and len(points) > 3:
        hull = ConvexHull(points)
        fig.add_trace(go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2], 
                                i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2], 
                                color='magenta', opacity=0.5))
        
    points_df = pd.DataFrame(points, columns=["X", "Y", "Z"])

# ========================================================
# 4. 구
# ========================================================
elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 30))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.8))

# ========================================================
# [레이아웃] 자동 시점 (aspectmode='data')
# ========================================================
# 수동으로 범위를 지정하지 않고 Plotly가 데이터에 맞춰서 알아서 줌인/줌아웃 하게 합니다.
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data' # << 여기가 핵심입니다. 데이터 있는 곳을 비춥니다.
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- 디버깅용: 데이터가 실제로 존재하는지 확인 ---
if points_df is not None:
    with st.expander("🔍 좌표 데이터 확인 (여기에 숫자가 없으면 수학 공식 오류입니다)"):
        st.dataframe(points_df)
