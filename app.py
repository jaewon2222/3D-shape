import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 시뮬레이터", layout="wide")
st.title("🧊 3D 입체도형 시뮬레이터")

# ==========================================
# 1. 핵심 계산 함수들 (Core Logic)
# ==========================================

def create_cap(r, height, n_sides, is_top=True):
    """도형의 뚜껑/바닥 생성"""
    if r <= 0: return None
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    x = np.append(r * np.cos(theta), 0)
    y = np.append(r * np.sin(theta), 0)
    z_val = height if is_top else 0
    z = np.append(np.full_like(theta, z_val), z_val)
    
    center_idx = len(x) - 1
    i = np.arange(n_sides)
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=np.full(n_sides, center_idx),
        j=i, k=(i + 1) % (n_sides + 1),
        color='skyblue', opacity=0.8, flatshading=True, name='Cap'
    )

def get_clean_wireframe(points):
    """정다면체용: 대각선 없는 깔끔한 모서리 선 추출"""
    dist_mat = distance_matrix(points, points)
    rounded_dists = np.round(dist_mat, 4)
    unique_dists = np.unique(rounded_dists)
    edge_length = unique_dists[1] if len(unique_dists) > 1 else 0
    tol = 1e-4
    pairs = np.argwhere(np.abs(dist_mat - edge_length) < tol)
    
    xl, yl, zl = [], [], []
    for i, j in pairs:
        if i < j:
            p1, p2 = points[i], points[j]
            xl.extend([p1[0], p2[0], None])
            yl.extend([p1[1], p2[1], None])
            zl.extend([p1[2], p2[2], None])
            
    return go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines', 
        line=dict(color='black', width=4), name='Edge'
    )

def make_prism_like(n_sides, r_bottom, r_top, height):
    """기둥, 뿔, 뿔대 통합 생성 함수"""
    traces = []
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    
    x_b, y_b = r_bottom * np.cos(theta), r_bottom * np.sin(theta)
    x_t, y_t = r_top * np.cos(theta), r_top * np.sin(theta)
    z_b, z_t = np.zeros_like(theta), np.full_like(theta, height)
    
    # 옆면 (Side)
    i = np.arange(n_sides)
    mesh = go.Mesh3d(
        x=np.concatenate([x_b[:-1], x_t[:-1]]),
        y=np.concatenate([y_b[:-1], y_t[:-1]]),
        z=np.concatenate([z_b[:-1], z_t[:-1]]),
        i=np.concatenate([i, i + n_sides]),
        j=np.concatenate([(i + 1) % n_sides, (i + 1) % n_sides]),
        k=np.concatenate([i + n_sides, (i + 1) % n_sides + n_sides]),
        color='skyblue', opacity=0.8, flatshading=True, name='Side'
    )
    traces.append(mesh)
    
    # 뚜껑/바닥
    if r_bottom > 0: traces.append(create_cap(r_bottom, 0, n_sides, False))
    if r_top > 0: traces.append(create_cap(r_top, height, n_sides, True))
    
    # 와이어프레임 (다각형일 때만)
    if n_sides < 30:
        xl, yl, zl = [], [], []
        # 가로선
        for x, y, z in [(x_b, y_b, z_b), (x_t, y_t, z_t)]:
            xl.extend(x); xl.append(None)
            yl.extend(y); yl.append(None)
            zl.extend(z); zl.append(None)
        # 세로선
        for k in range(n_sides):
            xl.extend([x_b[k], x_t[k], None])
            yl.extend([y_b[k], y_t[k], None])
            zl.extend([z_b[k], z_t[k], None])
        traces.append(go.Scatter3d(x=xl, y=yl, z=zl, mode='lines', line=dict(color='black', width=3), name='Edge'))
        
    return traces

def make_platonic_solid(solid_type, size):
    """정다면체 생성"""
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    
    if "정4" in solid_type: vertices = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    elif "정6" in solid_type:
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]: vertices.append([x, y, z])
    elif "정8" in solid_type: vertices = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif "정12" in solid_type:
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]: vertices.append([x, y, z])
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.extend([[0, i/phi, j*phi], [i/phi, j*phi, 0], [j*phi, 0, i/phi]])
    elif "정20" in solid_type:
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.extend([[0, i, j*phi], [i, j*phi, 0], [j*phi, 0, i]])

    points = np.array(vertices) * size
    hull = ConvexHull(points)
    x, y, z = points.T
    mesh = go.Mesh3d(x=x, y=y, z=z, i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                     color='orange', opacity=0.9, flatshading=True, name='Face')
    lines = get_clean_wireframe(points)
    return [mesh, lines]

def make_sphere(radius):
    """구 생성"""
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 30), np.linspace(0, 2 * np.pi, 60))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return [go.Surface(x=x, y=y, z=z, colorscale='Viridis', showscale=False, opacity=0.9)]

# ==========================================
# 2. 사이드바 UI 로직 (메뉴 분리)
# ==========================================

st.sidebar.header("도형 선택")

# 최상위 카테고리를 명확하게 분리
category = st.sidebar.selectbox(
    "어떤 도형을 만드나요?",
    ("기둥 (Prism/Cylinder)", "뿔 (Pyramid/Cone)", "뿔대 (Frustum)", "정다면체", "구")
)

fig = go.Figure()
traces = []
title_text = ""

# --- A. 기둥 / 뿔 / 뿔대 로직 ---
if category in ["기둥 (Prism/Cylinder)", "뿔 (Pyramid/Cone)", "뿔대 (Frustum)"]:
    
    # 1. 밑면 형태 선택
    base_shape = st.sidebar.radio("밑면의 모양", ["다각형", "원형"])
    
    # 2. 변의 개수 (다각형일 때만)
    if base_shape == "다각형":
        sides = st.sidebar.slider("밑면 변의 개수 (n)", 3, 12, 4)
        n = sides
    else:
        n = 60 # 원형은 변을 많이
        sides = 0 # 텍스트용

    # 3. 공통 슬라이더
    r_bottom = st.sidebar.slider("밑면 반지름", 1.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)

    # 4. 카테고리별 특수 로직 (반지름 결정)
    if "기둥" in category:
        r_top = r_bottom # 기둥은 위아래가 같음
        # 이름 생성
        if base_shape == "원형": title_text = "원기둥"
        else: title_text = f"{sides}각기둥"
        
    elif "뿔" in category and "뿔대" not in category: # 순수 뿔
        r_top = 0 # 뿔은 윗면이 0
        if base_shape == "원형": title_text = "원뿔"
        else: title_text = f"{sides}각뿔"
        
    else: # 뿔대
        r_top = st.sidebar.slider("윗면 반지름 (밑면과 다르게)", 0.1, 10.0, 3.0)
        if base_shape == "원형": title_text = "원뿔대"
        else: title_text = f"{sides}각뿔대"

    # 그리기
    traces = make_prism_like(n, r_bottom, r_top, h)


# --- B. 정다면체 로직 ---
elif category == "정다면체":
    solid_type = st.sidebar.selectbox(
        "종류",
        ["정4면체", "정6면체", "정8면체", "정12면체", "정20면체"]
    )
    size = st.sidebar.slider("크기", 1.0, 10.0, 5.0)
    traces = make_platonic_solid(solid_type, size)
    title_text = solid_type


# --- C. 구 로직 ---
elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 10.0, 5.0)
    traces = make_sphere(r)
    title_text = "구 (Sphere)"

# ==========================================
# 3. 시각화 (Visualization)
# ==========================================

st.subheader(f"📌 {title_text}")

for trace in traces:
    fig.add_trace(trace)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
