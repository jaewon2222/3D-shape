import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 시뮬레이터", layout="wide")
st.title("🧊 3D 입체도형 시뮬레이터 (Clean Wireframe)")

# --- 사이드바 설정 ---
st.sidebar.header("도형 설정")
main_category = st.sidebar.selectbox(
    "카테고리 선택",
    ("기둥/뿔/대 (Prism/Cone)", "정다면체 (Platonic Solids)", "구 (Sphere)")
)

# --- [유틸리티] 1. 다각형 뚜껑 만들기 ---
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
        i=np.full(n_sides, center_idx),
        j=i, k=(i + 1) % (n_sides + 1),
        color='skyblue', opacity=0.8, flatshading=True, name='Cap'
    )

# --- [유틸리티] 2. 정다면체용 모서리 추출기 (핵심 수정!) ---
def get_clean_wireframe(points):
    """
    모든 점 사이의 거리를 계산하여, '가장 짧은 거리(모서리 길이)'를 가진
    점들끼리만 선으로 연결합니다. (대각선 제거)
    """
    # 1. 모든 점들 간의 거리 행렬 계산
    dist_mat = distance_matrix(points, points)
    
    # 2. 자기 자신(거리 0)을 제외한 최소 거리(모서리 길이) 찾기
    # 0보다 큰 값 중 최소값 찾기 (약간의 오차 허용을 위해 round 처리)
    rounded_dists = np.round(dist_mat, 4)
    unique_dists = np.unique(rounded_dists)
    edge_length = unique_dists[1] if len(unique_dists) > 1 else 0
    
    # 3. 모서리 길이가 일치하는 쌍만 추출
    # 오차 범위(tolerance)를 두어 비교
    tol = 1e-4
    pairs = np.argwhere(np.abs(dist_mat - edge_length) < tol)
    
    # 4. 선 그리기용 좌표 리스트 생성
    xl, yl, zl = [], [], []
    # 중복 그리기 방지를 위해 i < j 인 경우만 처리
    for i, j in pairs:
        if i < j:
            p1, p2 = points[i], points[j]
            xl.extend([p1[0], p2[0], None])
            yl.extend([p1[1], p2[1], None])
            zl.extend([p1[2], p2[2], None])
            
    return go.Scatter3d(
        x=xl, y=yl, z=zl,
        mode='lines',
        line=dict(color='black', width=4), # 선 두께를 조금 더 키움
        name='Edge'
    )

# --- [함수 1] 기둥/뿔/대 생성 ---
def make_prism_like(n_sides, r_bottom, r_top, height):
    traces = []
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    
    x_b, y_b = r_bottom * np.cos(theta), r_bottom * np.sin(theta)
    x_t, y_t = r_top * np.cos(theta), r_top * np.sin(theta)
    z_b, z_t = np.zeros_like(theta), np.full_like(theta, height)
    
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
    
    if r_bottom > 0: traces.append(create_cap(r_bottom, 0, n_sides, False))
    if r_top > 0: traces.append(create_cap(r_top, height, n_sides, True))
    
    if n_sides < 30:
        xl, yl, zl = [], [], []
        for x, y, z in [(x_b, y_b, z_b), (x_t, y_t, z_t)]:
            xl.extend(x); xl.append(None)
            yl.extend(y); yl.append(None)
            zl.extend(z); zl.append(None)
        for k in range(n_sides):
            xl.extend([x_b[k], x_t[k], None])
            yl.extend([y_b[k], y_t[k], None])
            zl.extend([z_b[k], z_t[k], None])
        traces.append(go.Scatter3d(x=xl, y=yl, z=zl, mode='lines', line=dict(color='black', width=3), name='Edge'))
        
    return traces

# --- [함수 2] 정다면체 생성 (수정됨) ---
def make_platonic_solid(solid_type, size):
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    
    if solid_type == "정4면체 (Tetrahedron)":
        vertices = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    elif solid_type == "정6면체 (Cube)":
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append([x, y, z])
    elif solid_type == "정8면체 (Octahedron)":
        vertices = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif solid_type == "정12면체 (Dodecahedron)":
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append([x, y, z])
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.append([0, i/phi, j*phi])
                vertices.append([i/phi, j*phi, 0])
                vertices.append([j*phi, 0, i/phi])
    elif solid_type == "정20면체 (Icosahedron)":
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.append([0, i, j*phi])
                vertices.append([i, j*phi, 0])
                vertices.append([j*phi, 0, i])

    points = np.array(vertices) * size
    hull = ConvexHull(points) # 면 생성을 위해 사용 (Triangulation)
    
    # 1. 면 그리기 (Mesh3d)
    # flatshading=True 덕분에 같은 평면의 삼각형들은 경계선 없이 매끈하게 보임
    x, y, z = points.T
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=hull.simplices[:, 0],
        j=hull.simplices[:, 1],
        k=hull.simplices[:, 2],
        color='orange',
        opacity=0.9,
        flatshading=True, 
        name='Face'
    )
    
    # 2. 모서리 선 그리기 (새로 만든 함수 사용)
    # ConvexHull의 simplices를 쓰지 않고, 거리 기반으로 진짜 모서리만 찾음
    lines = get_clean_wireframe(points)
    
    return [mesh, lines]

# --- [함수 3] 구 생성 ---
def make_sphere(radius):
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return [go.Surface(x=x, y=y, z=z, colorscale='Viridis', showscale=False, opacity=0.9)]

# ====== 메인 실행 ======
fig = go.Figure()
traces = []

if main_category == "기둥/뿔/대 (Prism/Cone)":
    sub_type = st.sidebar.radio("세부 유형", ["다각형 (각기둥/뿔)", "원형 (원기둥/뿔)"])
    if "다각형" in sub_type:
        sides = st.sidebar.slider("밑면 변의 개수", 3, 12, 4)
        n = sides
    else:
        n = 60
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0=뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    traces = make_prism_like(n, r_b, r_t, h)
    
    if "다각형" in sub_type:
        name = "각기둥" if r_b == r_t else ("각뿔" if r_t == 0 else "각뿔대")
        st.subheader(f"{sides}{name}")
    else:
        name = "원기둥" if r_b == r_t else ("원뿔" if r_t == 0 else "원뿔대")
        st.subheader(name)

elif main_category == "정다면체 (Platonic Solids)":
    solid_type = st.sidebar.selectbox(
        "정다면체 종류 선택",
        ["정4면체 (Tetrahedron)", "정6면체 (Cube)", "정8면체 (Octahedron)", 
         "정12면체 (Dodecahedron)", "정20면체 (Icosahedron)"]
    )
    size = st.sidebar.slider("크기 (반지름)", 1.0, 10.0, 5.0)
    traces = make_platonic_solid(solid_type, size)
    st.subheader(solid_type)

elif main_category == "구 (Sphere)":
    r = st.sidebar.slider("반지름", 1.0, 10.0, 5.0)
    traces = make_sphere(r)
    st.subheader("구 (Sphere)")

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
