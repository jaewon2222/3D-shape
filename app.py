import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 시뮬레이터", layout="wide")
st.title("🧊 3D 입체도형 시뮬레이터")

# ==========================================
# 0. 공통 조명/재질 설정
# ==========================================
st.sidebar.header("도형 선택")

category = st.sidebar.selectbox(
    "카테고리",
    ("다각형 입체도형 (각기둥/각뿔...)", "회전체 (원기둥/원뿔...)", "정다면체", "구")
)

with st.sidebar.expander("💡 조명 & 재질 설정 (Lighting)", expanded=False):
    light_ambient = st.slider("기본 밝기", 0.0, 1.0, 0.3)
    light_diffuse = st.slider("빛 퍼짐", 0.0, 1.0, 0.9)
    light_specular = st.slider("광택", 0.0, 2.0, 0.5)
    light_roughness = st.slider("거칠기", 0.0, 1.0, 0.5)
    
    col1, col2, col3 = st.columns(3)
    lx = col1.number_input("X", value=100)
    ly = col2.number_input("Y", value=100)
    lz = col3.number_input("Z", value=1000)

lighting_config = dict(
    ambient=light_ambient, diffuse=light_diffuse, 
    specular=light_specular, roughness=light_roughness, fresnel=0.5
)
light_position = dict(x=lx, y=ly, z=lz)


# ==========================================
# 1. 핵심 계산 함수들
# ==========================================

def create_cap(r, height, n_sides, is_top=True):
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
        i=np.full(n_sides, center_idx), j=i, k=(i + 1) % (n_sides + 1),
        color='skyblue', opacity=1.0, flatshading=True, name='Cap',
        lighting=lighting_config, lightposition=light_position
    )

def get_clean_wireframe(points):
    """정다면체용 모서리"""
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
        line=dict(color='black', width=4), name='Edge', hoverinfo='skip'
    )

def make_prism_like(n_sides, r_bottom, r_top, height, is_smooth=False):
    traces = []
    # 끝점을 맞추기 위해 닫힌 루프 생성
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    
    x_b, y_b = r_bottom * np.cos(theta), r_bottom * np.sin(theta)
    x_t, y_t = r_top * np.cos(theta), r_top * np.sin(theta)
    z_b, z_t = np.zeros_like(theta), np.full_like(theta, height)
    
    # 1. 면 그리기 (Mesh)
    i = np.arange(n_sides)
    mesh = go.Mesh3d(
        x=np.concatenate([x_b[:-1], x_t[:-1]]),
        y=np.concatenate([y_b[:-1], y_t[:-1]]),
        z=np.concatenate([z_b[:-1], z_t[:-1]]),
        i=np.concatenate([i, i + n_sides]),
        j=np.concatenate([(i + 1) % n_sides, (i + 1) % n_sides]),
        k=np.concatenate([i + n_sides, (i + 1) % n_sides + n_sides]),
        color='skyblue', opacity=1.0, 
        flatshading=not is_smooth,
        name='Side', lighting=lighting_config, lightposition=light_position
    )
    traces.append(mesh)
    
    # 2. 뚜껑/바닥 그리기
    if r_bottom > 0: traces.append(create_cap(r_bottom, 0, n_sides, False))
    if r_top > 0: traces.append(create_cap(r_top, height, n_sides, True))
    
    # 3. 테두리(Wireframe) 그리기
    xl, yl, zl = [], [], []

    # (A) 상단/하단 원형 테두리 (모든 도형 공통)
    # 바닥 원
    xl.extend(x_b); xl.append(x_b[0]); xl.append(None) # 시작점과 끝점 연결
    yl.extend(y_b); yl.append(y_b[0]); yl.append(None)
    zl.extend(z_b); zl.append(z_b[0]); zl.append(None)
    
    # 윗면 원 (반지름이 0보다 클 때만)
    if r_top > 0:
        xl.extend(x_t); xl.append(x_t[0]); xl.append(None)
        yl.extend(y_t); yl.append(y_t[0]); yl.append(None)
        zl.extend(z_t); zl.append(z_t[0]); zl.append(None)

    # (B) 세로선 (다각형일 때만 그리기)
    # 회전체(is_smooth=True)는 세로선을 그리면 까맣게 되므로 생략
    if not is_smooth:
        for k in range(n_sides):
            xl.extend([x_b[k], x_t[k], None])
            yl.extend([y_b[k], y_t[k], None])
            zl.extend([z_b[k], z_t[k], None])

    traces.append(go.Scatter3d(
        x=xl, y=yl, z=zl, mode='lines', 
        line=dict(color='black', width=3), 
        name='Edge', hoverinfo='skip'
    ))
        
    return traces

def make_platonic_solid(solid_type, size):
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
    mesh = go.Mesh3d(
        x=x, y=y, z=z, i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
        color='orange', opacity=1.0, flatshading=True, name='Face',
        lighting=lighting_config, lightposition=light_position
    )
    lines = get_clean_wireframe(points)
    return [mesh, lines]

def make_sphere(radius):
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2 * np.pi, 100))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # 구는 와이어프레임을 그리면 복잡해지므로 면만 표시하거나
    # 원한다면 경도/위도 선을 추가할 수 있음 (여기선 깔끔하게 면만 유지)
    return [go.Surface(
        x=x, y=y, z=z, colorscale='Viridis', showscale=False, opacity=1.0,
        lighting=lighting_config, lightposition=light_position
    )]


# ==========================================
# 2. 메인 UI 로직
# ==========================================

fig = go.Figure()
traces = []
title_text = ""

if "다각형" in category:
    shape_type = st.sidebar.radio("형태", ["기둥", "뿔", "뿔대"], horizontal=True)
    sides = st.sidebar.slider("밑면의 변 (n)", 3, 12, 4)
    r_bottom = st.sidebar.slider("밑면 반지름", 1.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)

    if shape_type == "기둥": r_top = r_bottom; title_text = f"{sides}각기둥"
    elif shape_type == "뿔": r_top = 0; title_text = f"{sides}각뿔"
    else: r_top = st.sidebar.slider("윗면 반지름", 0.1, 10.0, 3.0); title_text = f"{sides}각뿔대"
        
    traces = make_prism_like(sides, r_bottom, r_top, h, is_smooth=False)

elif "회전체" in category:
    shape_type = st.sidebar.radio("형태", ["기둥", "뿔", "뿔대"], horizontal=True)
    r_bottom = st.sidebar.slider("밑면 반지름", 1.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    
    if shape_type == "기둥": r_top = r_bottom; title_text = "원기둥"
    elif shape_type == "뿔": r_top = 0; title_text = "원뿔"
    else: r_top = st.sidebar.slider("윗면 반지름", 0.1, 10.0, 3.0); title_text = "원뿔대"
        
    # 원형은 테두리(윗원, 아랫원)는 그리되, 세로선은 생략하여 깔끔하게 표현
    traces = make_prism_like(80, r_bottom, r_top, h, is_smooth=True)

elif category == "정다면체":
    solid_type = st.sidebar.selectbox("종류", ["정4면체", "정6면체", "정8면체", "정12면체", "정20면체"])
    size = st.sidebar.slider("크기", 1.0, 10.0, 5.0)
    traces = make_platonic_solid(solid_type, size)
    title_text = solid_type

elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 10.0, 5.0)
    traces = make_sphere(r)
    title_text = "구"

# ==========================================
# 3. 시각화
# ==========================================
st.subheader(f"📌 {title_text}")
for trace in traces: fig.add_trace(trace)
fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
    margin=dict(l=0, r=0, b=0, t=0), height=600
)
st.plotly_chart(fig, use_container_width=True)
