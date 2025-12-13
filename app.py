import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Scipy가 없으면 정다면체에서 에러나지 않게 처리
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (함수 미사용 버전)")

# 사이드바 설정
st.sidebar.header("설정")
category = st.sidebar.radio("도형 카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"])

# 그래프 그릴 준비
fig = go.Figure()

# 조명 설정 (밝게)
light_config = dict(ambient=0.6, diffuse=0.5, roughness=0.1, specular=0.1)

# ========================================================
# 1. 각기둥 / 각뿔 / 각뿔대 (함수 없이 각각 작성)
# ========================================================
if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)

    # --- [1-1] 각기둥 코드 ---
    if sub_type == "각기둥":
        rt = rb # 윗면 = 아랫면
        theta = np.linspace(0, 2*np.pi, n+1)
        x_bot = rb * np.cos(theta)
        y_bot = rb * np.sin(theta)
        x_top = rt * np.cos(theta)
        y_top = rt * np.sin(theta)
        
        # 좌표 합치기
        x = np.concatenate([x_top, x_bot, [0], [0]])
        y = np.concatenate([y_top, y_bot, [0], [0]])
        z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
        
        # 면 만들기
        i, j, k = [], [], []
        for idx in range(n):
            i.extend([idx, idx])
            j.extend([n+1+idx, n+1+idx+1])
            k.extend([n+1+idx+1, idx+1])
        # 뚜껑/바닥
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#00CCFF', flatshading=True, lighting=light_config, name="각기둥"))

    # --- [1-2] 각뿔 코드 ---
    elif sub_type == "각뿔":
        rt = 0 # 윗면 0
        theta = np.linspace(0, 2*np.pi, n+1)
        x_bot = rb * np.cos(theta)
        y_bot = rb * np.sin(theta)
        x_top = rt * np.cos(theta)
        y_top = rt * np.sin(theta)
        
        x = np.concatenate([x_top, x_bot, [0], [0]])
        y = np.concatenate([y_top, y_bot, [0], [0]])
        z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
        
        i, j, k = [], [], []
        for idx in range(n): # 옆면
            i.extend([idx, idx])
            j.extend([n+1+idx, n+1+idx+1])
            k.extend([n+1+idx+1, idx+1])
        # 바닥만 있음 (뚜껑 없음)
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FF6666', flatshading=True, lighting=light_config, name="각뿔"))

    # --- [1-3] 각뿔대 코드 ---
    elif sub_type == "각뿔대":
        rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)
        theta = np.linspace(0, 2*np.pi, n+1)
        x_bot = rb * np.cos(theta)
        y_bot = rb * np.sin(theta)
        x_top = rt * np.cos(theta)
        y_top = rt * np.sin(theta)
        
        x = np.concatenate([x_top, x_bot, [0], [0]])
        y = np.concatenate([y_top, y_bot, [0], [0]])
        z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
        
        i, j, k = [], [], []
        for idx in range(n):
            i.extend([idx, idx])
            j.extend([n+1+idx, n+1+idx+1])
            k.extend([n+1+idx+1, idx+1])
        # 뚜껑/바닥 모두 있음
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#66FF66', flatshading=True, lighting=light_config, name="각뿔대"))


# ========================================================
# 2. 원기둥 / 원뿔 / 원뿔대 (함수 없이 각각 작성 - n을 60으로 고정)
# ========================================================
elif category == "원기둥/원뿔/원뿔대":
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    n = 60 # 원은 60각형으로 처리

    # --- [2-1] 원기둥 코드 ---
    if sub_type == "원기둥":
        rt = rb
        theta = np.linspace(0, 2*np.pi, n+1)
        x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
        x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)
        
        x = np.concatenate([x_top, x_bot, [0], [0]])
        y = np.concatenate([y_top, y_bot, [0], [0]])
        z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
        
        i, j, k = [], [], []
        for idx in range(n):
            i.extend([idx, idx]); j.extend([n+1+idx, n+1+idx+1]); k.extend([n+1+idx+1, idx+1])
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FFD700', flatshading=True, lighting=light_config, name="원기둥"))

    # --- [2-2] 원뿔 코드 ---
    elif sub_type == "원뿔":
        rt = 0
        theta = np.linspace(0, 2*np.pi, n+1)
        x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
        x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)
        
        x = np.concatenate([x_top, x_bot, [0], [0]])
        y = np.concatenate([y_top, y_bot, [0], [0]])
        z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
        
        i, j, k = [], [], []
        for idx in range(n):
            i.extend([idx, idx]); j.extend([n+1+idx, n+1+idx+1]); k.extend([n+1+idx+1, idx+1])
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FF4500', flatshading=True, lighting=light_config, name="원뿔"))

    # --- [2-3] 원뿔대 코드 ---
    elif sub_type == "원뿔대":
        rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)
        theta = np.linspace(0, 2*np.pi, n+1)
        x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
        x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)
        
        x = np.concatenate([x_top, x_bot, [0], [0]])
        y = np.concatenate([y_top, y_bot, [0], [0]])
        z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
        
        i, j, k = [], [], []
        for idx in range(n):
            i.extend([idx, idx]); j.extend([n+1+idx, n+1+idx+1]); k.extend([n+1+idx+1, idx+1])
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#32CD32', flatshading=True, lighting=light_config, name="원뿔대"))


# ========================================================
# 3. 정다면체 (좌표 데이터 입력)
# ========================================================
elif category == "정다면체":
    if not has_scipy:
        st.error("Scipy 라이브러리가 필요합니다.")
    else:
        sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        size = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
        phi = (1 + np.sqrt(5)) / 2
        points = []

        if sub_type == "정사면체":
            points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
        elif sub_type == "정육면체":
            for x in [-1,1]:
                for y in [-1,1]:
                    for z in [-1,1]: points.append([x,y,z])
        elif sub_type == "정팔면체":
            points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
        elif sub_type == "정십이면체":
            for x in [-1,1]:
                for y in [-1,1]:
                    for z in [-1,1]: points.append([x,y,z])
            for i in [-1,1]:
                for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
        elif sub_type == "정이십면체":
            for i in [-1,1]:
                for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

        # ConvexHull로 면 만들기
        points = np.array(points) * size
        hull = ConvexHull(points)
        x, y, z = points[:,0], points[:,1], points[:,2]
        i, j, k = hull.simplices[:,0], hull.simplices[:,1], hull.simplices[:,2]
        
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FF00FF', flatshading=True, lighting=light_config))


# ========================================================
# 4. 구 (수학 공식 사용)
# ========================================================
elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', lighting=light_config))


# ========================================================
# [중요] 카메라 설정 (고정 범위)
# ========================================================
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], title='X'),
        yaxis=dict(range=[-10, 10], title='Y'),
        zaxis=dict(range=[-10, 10], title='Z'), # 위아래 넉넉하게
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
