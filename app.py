import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="완벽한 겨냥도 v3", layout="wide")
st.title("📐 3D 입체도형 관측소 (대각선 완벽 제거판)")
st.markdown("""
**[최종 수정]** '엣지 트래킹' 방식을 도입하여 평면 위의 불필요한 대각선을 강제로 삭제했습니다.
이제 사각기둥의 옆면이 깨끗한 직사각형으로 보일 것입니다.
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

# --- 4. 핵심 렌더링 로직 (완전 개편) ---
rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
hull = ConvexHull(rotated_points)
normals = hull.equations[:, :3]

# (1) 각 면이 보이는지 판단 (앞면/뒷면)
visible_faces_mask = [normal[2] > 1e-4 for normal in normals]

# (2) 모든 엣지를 수집하고 공유하는 면들을 기록
# edge_to_faces = { (p1_idx, p2_idx) : [face_idx1, face_idx2, ...] }
edge_to_faces = {}

for face_idx, simplex in enumerate(hull.simplices):
    n_pts = len(simplex)
    for k in range(n_pts):
        p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts])) # 점 인덱스 정렬해서 키로 사용
        edge = (p1, p2)
        if edge not in edge_to_faces:
            edge_to_faces[edge] = []
        edge_to_faces[edge].append(face_idx)

# (3) 평면 판별 함수
def is_coplanar(n1, n2):
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 == 0 or norm2 == 0: return False
    dot = np.dot(n1, n2) / (norm1 * norm2)
    return dot > 0.999 # 거의 평행하면 True

visible_edges = set()
hidden_edges = set()

# (4) 엣지 분류 로직 (여기가 핵심!)
for edge, faces in edge_to_faces.items():
    # 엣지는 보통 2개의 면을 공유합니다.
    if len(faces) == 2:
        f1, f2 = faces
        n1, n2 = normals[f1], normals[f2]
        
        # [핵심] 두 면이 평평하게 이어져 있으면(Coplanar), 이 엣지는 '가짜'입니다.
        if is_coplanar(n1, n2):
            continue # 그리지 않고 건너뜀!
            
        # 평평하지 않다면 '진짜 모서리'입니다. 이제 실선/점선 구분
        v1 = visible_faces_mask[f1]
        v2 = visible_faces_mask[f2]
        
        if v1 or v2: 
            # 둘 중 하나라도 보이면 실선
            visible_edges.add(edge)
        else:
            # 둘 다 안 보이면 점선
            hidden_edges.add(edge)
            
    else:
        # 면을 1개만 공유하거나 3개 이상 공유하는 특이 케이스 (보통 외곽선)
        # 해당 면이 보이면 실선, 아니면 점선
        is_visible = False
        for f in faces:
            if visible_faces_mask[f]:
                is_visible = True
                break
        if is_visible:
            visible_edges.add(edge)
        else:
            hidden_edges.add(edge)

# (5) 채울 면 수집 (보이는 면만)
visible_mesh_indices = []
for i, is_vis in enumerate(visible_faces_mask):
    if is_vis:
        visible_mesh_indices.append(hull.simplices[i])

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

# (3) 면 채우기
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
