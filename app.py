import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="완벽한 겨냥도", layout="wide")
st.title("📐 3D 입체도형 관측소 (수학적 겨냥도)")
st.markdown("""
**[사용법]** 왼쪽의 **'도형 회전' 슬라이더**를 움직여보세요.
* **앞에 있는 면:** 색칠됨 + 실선 테두리
* **뒤에 있는 면:** 색칠 안 됨(투명) + 점선 테두리
""")

# --- 1. 사이드바 설정 ---
st.sidebar.header("1. 도형 선택")
category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (다각형 근사)", "정다면체"])

st.sidebar.header("2. 도형 회전 (필수)")
rot_x = st.sidebar.slider("X축 회전 (위아래)", 0, 360, 20)
rot_y = st.sidebar.slider("Y축 회전 (좌우)", 0, 360, 30)
rot_z = st.sidebar.slider("Z축 회전", 0, 360, 0)

# --- 2. 회전 함수 ---
def rotate_points(points, rx, ry, rz):
    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)
    mat_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    mat_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    mat_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    return points @ mat_x.T @ mat_y.T @ mat_z.T

# --- 3. 점 데이터 생성 ---
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
    n = 30 # 원 근사
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

# --- 4. 핵심 로직: 보이는 면만 추출 ---
rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
hull = ConvexHull(rotated_points)
normals = hull.equations[:, :3]

# 법선 벡터의 z값이 양수면 '앞면', 음수면 '뒷면'
visible_faces_mask = [normal[2] > 0 for normal in normals]

visible_edges = set()
hidden_edges = set()
visible_mesh_i, visible_mesh_j, visible_mesh_k = [], [], []

for simplex_idx, simplex in enumerate(hull.simplices):
    is_visible = visible_faces_mask[simplex_idx]
    
    # [중요 변경점] 보이는 면(Visible Face)만 메쉬 그리기 목록에 추가
    if is_visible:
        visible_mesh_i.append(simplex[0])
        visible_mesh_j.append(simplex[1])
        visible_mesh_k.append(simplex[2])
    
    # 엣지(선) 분류
    n_pts = len(simplex)
    for i in range(n_pts):
        p1, p2 = simplex[i], simplex[(i+1)%n_pts]
        edge = tuple(sorted((p1, p2)))
        
        if is_visible:
            if edge in hidden_edges: hidden_edges.remove(edge)
            visible_edges.add(edge)
        else:
            if edge not in visible_edges:
                hidden_edges.add(edge)

# --- 5. 시각화 ---
fig = go.Figure()

# (1) 숨은 선 (점선)
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

# (2) 보이는 선 (실선)
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

# (3) 보이는 면만 채우기 (뒷면은 렌더링 X)
fig.add_trace(go.Mesh3d(
    x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
    i=visible_mesh_i, j=visible_mesh_j, k=visible_mesh_k, # 필터링된 인덱스만 사용
    color='#dceefc', opacity=0.5, # 반투명
    lighting=dict(ambient=0.8),
    hoverinfo='none', name='면'
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
