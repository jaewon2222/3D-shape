import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- Scipy 체크 ---
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (면 연결 수정)")
st.markdown("점과 점 사이를 잇는 **순서(Index)**를 수학적으로 완벽하게 맞췄습니다.")

# --- 사이드바 ---
st.sidebar.header("설정")
category = st.sidebar.radio("도형 카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"])

fig = go.Figure()

# --- 조명 설정 (강하게) ---
lighting_effects = dict(ambient=0.6, diffuse=0.5, roughness=0.1, specular=0.4)

# ========================================================
# 1. 각기둥 / 각뿔 / 각뿔대 (인덱스 로직 수정)
# ========================================================
if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)

    # 윗면 반지름 결정
    if sub_type == "각기둥": rt = rb
    elif sub_type == "각뿔": rt = 0
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)

    # [핵심 수정] endpoint=False로 중복 점 제거 (깔끔한 연결을 위해)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # 좌표 계산
    x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
    x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)

    # 전체 점 배열: [Top 점들(0~n-1), Bot 점들(n~2n-1), Top중심(2n), Bot중심(2n+1)]
    x = np.concatenate([x_top, x_bot, [0], [0]])
    y = np.concatenate([y_top, y_bot, [0], [0]])
    z = np.concatenate([np.full(n, h), np.zeros(n), [h], [0]])

    # 인덱스 계산 (Modulo 연산 사용으로 끊김 없이 연결)
    i, j, k = [], [], []
    
    # 주요 인덱스
    top_start = 0
    bot_start = n
    top_center = 2 * n
    bot_center = 2 * n + 1

    for idx in range(n):
        next_idx = (idx + 1) % n  # 마지막 점과 첫 점을 연결

        # 1. 옆면 (사각형을 삼각형 2개로 쪼개기)
        # 삼각형 1: Top_current -> Bot_current -> Bot_next
        i.extend([top_start + idx])
        j.extend([bot_start + idx])
        k.extend([bot_start + next_idx])

        # 삼각형 2: Top_current -> Bot_next -> Top_next
        i.extend([top_start + idx])
        j.extend([bot_start + next_idx])
        k.extend([top_start + next_idx])

        # 2. 뚜껑 (윗면 반지름 > 0 일 때)
        if rt > 0:
            i.extend([top_start + idx])
            j.extend([top_start + next_idx])
            k.extend([top_center])

        # 3. 바닥 (밑면 반지름 > 0 일 때)
        if rb > 0:
            i.extend([bot_start + idx])
            j.extend([bot_center])
            k.extend([bot_start + next_idx])

    # 점과 면 동시에 그리기
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#00BFFF', opacity=1.0, flatshading=True, lighting=lighting_effects, name='면'))
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, color='black'), name='꼭짓점'))


# ========================================================
# 2. 원기둥 / 원뿔 / 원뿔대 (같은 로직 적용)
# ========================================================
elif category == "원기둥/원뿔/원뿔대":
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    n = 60 # 해상도
    
    if sub_type == "원기둥": rt = rb
    elif sub_type == "원뿔": rt = 0
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)

    theta = np.linspace(0, 2*np.pi, n, endpoint=False) # endpoint=False 중요
    x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
    x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)

    x = np.concatenate([x_top, x_bot, [0], [0]])
    y = np.concatenate([y_top, y_bot, [0], [0]])
    z = np.concatenate([np.full(n, h), np.zeros(n), [h], [0]])

    i, j, k = [], [], []
    top_center = 2 * n
    bot_center = 2 * n + 1

    for idx in range(n):
        next_idx = (idx + 1) % n
        # 옆면
        i.extend([idx, idx])
        j.extend([n + idx, n + next_idx])
        k.extend([n + next_idx, next_idx])
        
        # 뚜껑/바닥
        if rt > 0: 
            i.extend([idx]); j.extend([next_idx]); k.extend([top_center])
        if rb > 0:
            i.extend([n + idx]); j.extend([bot_center]); k.extend([n + next_idx])

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FFD700', opacity=1.0, flatshading=True, lighting=lighting_effects))

# ========================================================
# 3. 정다면체
# ========================================================
elif category == "정다면체":
    if not has_scipy:
        st.error("Scipy가 없습니다.")
    else:
        sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        size = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
        phi = (1 + np.sqrt(5)) / 2
        points = []

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
        hull = ConvexHull(points) # ConvexHull이 자동으로 면(Triangle)을 계산해줌
        
        fig.add_trace(go.Mesh3d(
            x=points[:,0], y=points[:,1], z=points[:,2], 
            i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2], 
            color='#FF00FF', opacity=1.0, flatshading=True, lighting=lighting_effects
        ))

# ========================================================
# 4. 구
# ========================================================
elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', lighting=lighting_effects))

# ========================================================
# [레이아웃] 자동 시점 (aspectmode='data')
# ========================================================
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=True), yaxis=dict(visible=True), zaxis=dict(visible=True),
        aspectmode='data' # 데이터가 있는 곳으로 카메라 자동 이동
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
