import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="궁극의 도형 관측소", layout="wide")
st.title("📐 3D 입체도형 관측소 (문제집 스타일)")
st.markdown("""
**[최종 업데이트]**
1. **음영 제거:** 조명 효과를 끄고 면을 흰색(투명) 처리하여 **수학 문제집 그림**처럼 깨끗하게 만들었습니다.
2. **원기둥/원뿔:** 매끈한 곡면 처리로 불필요한 선을 모두 지웠습니다.
""")

# --- 1. 사이드바 설정 ---
st.sidebar.header("1. 보기 설정")
projection_mode = st.sidebar.radio(
    "투영 방식", 
    ["교과서 모드 (직교 투영)", "현실 모드 (원근 투영)"],
    index=0
)

# [추가] 면 스타일 선택 기능
face_style = st.sidebar.radio(
    "면(Face) 스타일",
    ["깔끔한 흰색 (음영 없음)", "투명 (선만 보기)", "기존 (파란색 입체감)"],
    index=0
)

st.sidebar.header("2. 도형 선택")
category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (매끈함)", "정다면체"])

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
is_smooth_surface = False 

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

elif category == "원기둥/원뿔 (매끈함)":
    is_smooth_surface = True
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    n = 60 
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

# --- 4. 렌더링 로직 ---
rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
hull = ConvexHull(rotated_points)

# 법선 벡터 
normals = []
for eq in hull.equations:
    n_vec = eq[:3]
    normals.append(n_vec / np.linalg.norm(n_vec))
normals = np.array(normals)

# 가시성 판단
camera_pos = np.array([0, 0, 10.0])
visible_faces_mask = []

for i, simplex in enumerate(hull.simplices):
    if "교과서 모드" in projection_mode:
        is_visible = normals[i][2] > 0
    else:
        face_center = np.mean(rotated_points[simplex], axis=0)
        view_vector = face_center - camera_pos
        is_visible = np.dot(view_vector, normals[i]) < 0
    visible_faces_mask.append(is_visible)

# 엣지 분류 및 매끈한 처리
edge_to_faces = {}
for face_idx, simplex in enumerate(hull.simplices):
    n_pts = len(simplex)
    for k in range(n_pts):
        p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))
        edge = (p1, p2)
        if edge not in edge_to_faces: edge_to_faces[edge] = []
        edge_to_faces[edge].append(face_idx)

def is_coplanar(n1, n2): return np.dot(n1, n2) > 0.999
def is_side_face(normal): return abs(normal[2]) < 0.9

visible_edges = set()
hidden_edges = set()

for edge, faces in edge_to_faces.items():
    if len(faces) == 2:
        f1, f2 = faces
        n1, n2 = normals[f1], normals[f2]
        v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]
        
        if is_coplanar(n1, n2): continue 
        
        if is_smooth_surface:
            if is_side_face(n1) and is_side_face(n2):
                if v1 and v2: continue # 곡면 내부 선 제거
        
        if v1 or v2: visible_edges.add(edge)
        else: hidden_edges.add(edge)
    else:
        if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)
        else: hidden_edges.add(edge)

visible_mesh_indices = []
for i, is_vis in enumerate(visible_faces_mask):
    if is_vis: visible_mesh_indices.append(hull.simplices[i])

# --- 5. 시각화 (스타일 적용) ---
fig = go.Figure()

# [1] 숨은 선 그리기
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

# [2] 면 그리기 (스타일에 따라 분기)
if visible_mesh_indices:
    
    # 기본값: 투명 (선만 보기) - 아무것도 안 그림
    mesh_color = 'white'
    mesh_opacity = 0.0 
    mesh_lighting = dict(ambient=1.0)
    
    if "기존" in face_style:
        mesh_color = '#dceefc'
        mesh_opacity = 0.5
        mesh_lighting = dict(ambient=0.6, diffuse=0.9, roughness=0.1, specular=0.3)
    elif "깔끔한 흰색" in face_style:
        # [핵심] 그림자를 없애기 위해 ambient=1.0으로 설정
        mesh_color = 'white' 
        mesh_opacity = 0.1 # 아주 희미하게 면이 있음을 표시 (숨은 선이 뒤에 보이도록)
        mesh_lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0)
    
    # 투명 모드가 아니면 메쉬 추가
    if "투명" not in face_style:
        visible_mesh_indices = np.array(visible_mesh_indices)
        fig.add_trace(go.Mesh3d(
            x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
            i=visible_mesh_indices[:,0], j=visible_mesh_indices[:,1], k=visible_mesh_indices[:,2],
            color=mesh_color, opacity=mesh_opacity,
            lighting=mesh_lighting, # 조명 설정 적용
            hoverinfo='none', name='면'
        ))

# [3] 보이는 선 그리기 (맨 위에 그림)
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

# 카메라 및 배경 설정
if "교과서 모드" in projection_mode:
    proj_type = "orthographic"
    cam_dist = 2.0
else:
    proj_type = "perspective"
    cam_dist = 2.5

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False, showbackground=False), # 배경 격자 제거
        yaxis=dict(visible=False, showbackground=False),
        zaxis=dict(visible=False, showbackground=False),
        bgcolor='white', # 배경색 완전 흰색
        aspectmode='data',
        camera=dict(
            projection=dict(type=proj_type), 
            eye=dict(x=0, y=0, z=cam_dist),
            up=dict(x=0, y=1, z=0)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False,
    paper_bgcolor='white' # 전체 배경 흰색
)

st.plotly_chart(fig, use_container_width=True)
