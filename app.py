import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="궁극의 입체도형", layout="wide")
st.title("📐 3D 입체도형 관측소 (투영 모드 선택)")
st.markdown("""
**[사용법]** * **교과서 모드 (직교):** 왜곡 없이 정직한 모양을 보여줍니다. (길이, 각도 확인용)
* **현실 모드 (원근):** 가까운 건 크게, 먼 건 작게 보입니다. (입체감 확인용)
""")

# --- 1. 사이드바 설정 ---
st.sidebar.header("1. 보기 설정")
# [핵심] 투영 방식 선택 스위치 추가
projection_mode = st.sidebar.radio(
    "투영 방식 (카메라 렌즈)", 
    ["교과서 모드 (직교 투영)", "현실 모드 (원근 투영)"],
    index=0
)

st.sidebar.header("2. 도형 선택")
category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (다각형 근사)", "정다면체"])

st.sidebar.header("3. 도형 회전")
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

# --- 4. 렌더링 로직 (모드에 따라 분기) ---
rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
hull = ConvexHull(rotated_points)

# 법선 벡터 계산 및 정규화
normals = []
for eq in hull.equations:
    n = eq[:3]
    normals.append(n / np.linalg.norm(n))
normals = np.array(normals)

# [가시성 판단 로직 분기]
# 1. 교과서 모드 (Orthographic): 평행한 빛 (Z축 방향)
# 2. 현실 모드 (Perspective): 카메라 시점 (0,0,10)에서 발사
camera_pos = np.array([0, 0, 10.0])
visible_faces_mask = []

for i, simplex in enumerate(hull.simplices):
    if "교과서 모드" in projection_mode:
        # 직교 투영: 단순히 법선의 Z값이 양수면 보임
        is_visible = normals[i][2] > 0
    else:
        # 원근 투영: 시선 벡터와 법선의 내적 이용
        face_points = rotated_points[simplex]
        face_center = np.mean(face_points, axis=0)
        view_vector = face_center - camera_pos
        is_visible = np.dot(view_vector, normals[i]) < 0
        
    visible_faces_mask.append(is_visible)

# 엣지 추출 및 대각선 제거 로직
edge_to_faces = {}
for face_idx, simplex in enumerate(hull.simplices):
    n_pts = len(simplex)
    for k in range(n_pts):
        p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))
        edge = (p1, p2)
        if edge not in edge_to_faces: edge_to_faces[edge] = []
        edge_to_faces[edge].append(face_idx)

def is_coplanar(n1, n2):
    return np.dot(n1, n2) > 0.999

visible_edges = set()
hidden_edges = set()

for edge, faces in edge_to_faces.items():
    if len(faces) == 2:
        f1, f2 = faces
        n1, n2 = normals[f1], normals[f2]
        
        if is_coplanar(n1, n2): continue 
        
        if visible_faces_mask[f1] or visible_faces_mask[f2]:
            visible_edges.add(edge)
        else:
            hidden_edges.add(edge)
    else:
        is_visible = any(visible_faces_mask[f] for f in faces)
        if is_visible: visible_edges.add(edge)
        else: hidden_edges.add(edge)

visible_mesh_indices = []
for i, is_vis in enumerate(visible_faces_mask):
    if is_vis: visible_mesh_indices.append(hull.simplices[i])

# --- 5. 시각화 ---
fig = go.Figure()

# 숨은 선
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

# 보이는 선
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

# 면 채우기
if visible_mesh_indices:
    visible_mesh_indices = np.array(visible_mesh_indices)
    fig.add_trace(go.Mesh3d(
        x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
        i=visible_mesh_indices[:,0], j=visible_mesh_indices[:,1], k=visible_mesh_indices[:,2],
        color='#dceefc', opacity=0.5,
        lighting=dict(ambient=0.8), hoverinfo='none', name='면'
    ))

# [핵심] 카메라 설정 업데이트
if "교과서 모드" in projection_mode:
    proj_type = "orthographic"
    cam_dist = 2.0
else:
    proj_type = "perspective"
    cam_dist = 2.0

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data', # 비율 고정 (매우 중요)
        camera=dict(
            projection=dict(type=proj_type), 
            eye=dict(x=0, y=0, z=cam_dist),
            up=dict(x=0, y=1, z=0)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False
)

st.plotly_chart(fig, use_container_width=True)
