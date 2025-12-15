import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="완벽한 겨냥도 v2", layout="wide")
st.title("📐 3D 입체도형 관측소 (최종 수정판)")
st.markdown("""
**[개선 사항]**
1. **대각선 제거:** 사각형 면을 삼각형으로 쪼갤 때 생기는 불필요한 대각선을 지웠습니다.
2. **오차 보정:** 두 면만 보일 때 선이 깜빡이거나 사라지는 현상을 수정했습니다.
""")

# --- 1. 사이드바 설정 ---
st.sidebar.header("1. 도형 선택")
category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (다각형 근사)", "정다면체"])

st.sidebar.header("2. 도형 회전")
rot_x = st.sidebar.slider("X축 회전", 0, 360, 20)
rot_y = st.sidebar.slider("Y축 회전", 0, 360, 30)
rot_z = st.sidebar.slider("Z축 회전", 0, 360, 0)

# --- 2. 회전 함수 ---
def rotate_points(points, rx, ry, rz):
    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)
    mat_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    mat_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    mat_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    return points @ mat_x.T @ mat_y.T @ mat_z.T

# --- 3. 도형 데이터 생성 ---
points = []
if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = 4.0; rb = 2.0
    if sub_type == "각기둥": rt = rb
    elif sub_type == "각뿔": rt = 0.001
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, 1.9, 1.0)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    for t in theta: points.append([rt*np.cos(t), rt*np.sin(t), h/2])
    for t in theta: points.append([rb*np.cos(t), rb*np.sin(t), -h/2])

elif category == "원기둥/원뿔 (다각형 근사)":
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    n = 30 
    h = 4.0; rb = 2.0
    if sub_type == "원기둥": rt = rb
    elif sub_type == "원뿔": rt = 0.001
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, 1.9, 1.0)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    for t in theta: points.append([rt*np.cos(t), rt*np.sin(t), h/2])
    for t in theta: points.append([rb*np.cos(t), rb*np.sin(t), -h/2])

elif category == "정다면체":
    sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
    phi = (1 + np.sqrt(5)) / 2
    if sub_type == "정사면체": points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    elif sub_type == "정육면체": points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
    elif sub_type == "정팔면체": points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif sub_type == "정십이면체":
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        for i in [-1,1]:
             for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif sub_type == "정이십면체":
        for i in [-1,1]:
            for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])
points = np.array(points)

# --- 4. 고급 렌더링 로직 ---
rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
hull = ConvexHull(rotated_points)
normals = hull.equations[:, :3]

# (1) 면의 가시성 판단 (Epsilon 적용으로 깜빡임 방지)
# 1e-5보다 크면 보이는 것으로 간주
visible_faces_mask = [normal[2] > 1e-5 for normal in normals]

visible_edges = set()
hidden_edges = set()
visible_mesh_indices = []

# (2) Coplanar(같은 평면) 감지 로직
# ConvexHull은 사각형을 삼각형 2개로 쪼갭니다. 이 "가짜 모서리"를 찾아내서 지워야 깔끔합니다.
def is_coplanar(n1, n2):
    # 두 법선 벡터의 내적이 1에 가까우면(각도 0) 같은 평면입니다.
    # 정규화된 벡터라고 가정할 때 dot product가 1에 가까우면 평행
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 == 0 or norm2 == 0: return False
    dot = np.dot(n1, n2) / (norm1 * norm2)
    return dot > 0.999 # 거의 평행하면 True

# 각 면(Simplex) 순회
for i, simplex in enumerate(hull.simplices):
    # 보이는 면이라면 메쉬 그리기에 추가
    if visible_faces_mask[i]:
        visible_mesh_indices.append(simplex)

    # 이웃 정보 (neighbors)
    # hull.neighbors[i] 에는 i번째 면의 3개 모서리와 맞닿은 이웃 면들의 인덱스가 들어있음
    # 순서는 simplex의 점 순서와 대응됨: 
    # neighbor[i, 0]은 point 1-2 사이 변의 건너편 이웃
    # neighbor[i, 1]은 point 2-0 사이 변의 건너편 이웃 ... (scipy 버전에 따라 다를 수 있어 직접 매칭 권장)
    
    # 더 안전한 방법: 직접 엣지 루프 돌면서 이웃 찾기
    for k in range(3):
        p1, p2 = simplex[k], simplex[(k+1)%3]
        edge = tuple(sorted((p1, p2)))
        
        # 이 엣지의 건너편 이웃 면 인덱스 찾기
        neighbor_idx = hull.neighbors[i, k]
        
        # 1. Coplanar 체크 (가짜 선 제거)
        # 나와 내 이웃이 같은 평면(사각형의 쪼개진 틈)이라면 -> 선을 그리지 않음
        if is_coplanar(normals[i], normals[neighbor_idx]):
            continue 

        # 2. 실선/점선 분류
        # 내 면(i)과 이웃 면(neighbor_idx) 중 "하나라도 보이면" 실선
        is_me_visible = visible_faces_mask[i]
        is_neighbor_visible = visible_faces_mask[neighbor_idx]
        
        if is_me_visible or is_neighbor_visible:
            # 실선
            if edge in hidden_edges: hidden_edges.remove(edge)
            visible_edges.add(edge)
        else:
            # 둘 다 안 보여야 점선
            if edge not in visible_edges:
                hidden_edges.add(edge)

# --- 5. 시각화 ---
fig = go.Figure()

# (1) 숨은 선
x_dash, y_dash, z_dash = [], [], []
for p1, p2 in hidden_edges:
    pts = rotated_points[[p1, p2]]
    x_dash.extend([pts[0][0], pts[1][0], None])
    y_dash.extend([pts[0][1], pts[1][1], None])
    z_dash.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_dash, y=y_dash, z=z_dash, mode='lines',
    line=dict(color='gray', width=3, dash='dash'),
    name='숨은 선', hoverinfo='none'
))

# (2) 보이는 선
x_solid, y_solid, z_solid = [], [], []
for p1, p2 in visible_edges:
    pts = rotated_points[[p1, p2]]
    x_solid.extend([pts[0][0], pts[1][0], None])
    y_solid.extend([pts[0][1], pts[1][1], None])
    z_solid.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_solid, y=y_solid, z=z_solid, mode='lines',
    line=dict(color='black', width=5),
    name='보이는 선', hoverinfo='none'
))

# (3) 면 채우기 (보이는 면만)
if visible_mesh_indices:
    visible_mesh_indices = np.array(visible_mesh_indices)
    fig.add_trace(go.Mesh3d(
        x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
        i=visible_mesh_indices[:,0], j=visible_mesh_indices[:,1], k=visible_mesh_indices[:,2],
        color='#dceefc', opacity=0.5,
        lighting=dict(ambient=0.8), hoverinfo='none', name='면'
    ))

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data',
        camera=dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False
)

st.plotly_chart(fig, use_container_width=True)
