import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="수학 문제집 생성기", layout="wide")
st.title("💎 수학 문제집 도형 생성기 (고화질 스무스 버전)")
st.caption("해상도를 높이고 부드러운 쉐이딩을 적용하여 깨짐 현상을 없앴습니다.")

# --- 1. 사이드바 설정 ---
with st.sidebar:
    st.header("1. 보기 설정")
    projection_mode = st.radio(
        "투영 방식", 
        ["교과서 모드 (직교 투영)", "현실 모드 (원근 투영)"],
        index=0
    )

    st.header("2. 도형 선택")
    category = st.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/구 (매끈함)", "정다면체"])

    st.header("3. 도형 회전")
    col1, col2, col3 = st.columns(3)
    with col1: rot_x = st.slider("X", 0, 360, 20)
    with col2: rot_y = st.slider("Y", 0, 360, 30)
    with col3: rot_z = st.slider("Z", 0, 360, 0)

# --- 2. 수학 함수 ---
def rotate_points(points, rx, ry, rz):
    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)
    mat_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    mat_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    mat_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    return points @ mat_x.T @ mat_y.T @ mat_z.T

# --- 3. 도형 데이터 생성 (고해상도 적용) ---
points = []
is_curved_surface = False 

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

elif category == "원기둥/원뿔/구 (매끈함)":
    is_curved_surface = True
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대", "구"])
    
    if sub_type == "구":
        r = st.sidebar.slider("반지름", 1.0, 3.0, 2.0)
        # [수정] 해상도 대폭 증가 (깨짐 방지)
        u_steps = 60 
        v_steps = 30 
        u = np.linspace(0, 2 * np.pi, u_steps)
        v = np.linspace(0, np.pi, v_steps)
        for theta in u:
            for phi in v:
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                points.append([x, y, z])
    else:
        # [수정] 원기둥 해상도 증가
        n = 100 
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

# --- 4. 렌더링 및 가시성 계산 ---
try:
    rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
    hull = ConvexHull(rotated_points)
    
    normals = []
    for eq in hull.equations:
        n_vec = eq[:3] 
        normals.append(n_vec) 
    normals = np.array(normals)

    if "교과서 모드" in projection_mode:
        view_mode = "ortho"
        camera_pos = np.array([0, 0, 1000.0])
    else:
        view_mode = "perspective"
        camera_pos = np.array([0, 0, 10.0])

    visible_faces_mask = []
    for i, simplex in enumerate(hull.simplices):
        if view_mode == "ortho":
            is_visible = normals[i][2] > 0
        else:
            face_center = np.mean(rotated_points[simplex], axis=0)
            view_vector = camera_pos - face_center
            is_visible = np.dot(view_vector, normals[i]) > 0
        visible_faces_mask.append(is_visible)

    edge_to_faces = {}
    for face_idx, simplex in enumerate(hull.simplices):
        n_pts = len(simplex)
        for k in range(n_pts):
            p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))
            edge = (p1, p2)
            if edge not in edge_to_faces: edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # --- 5. 선 그리기 로직 ---
    visible_edges = set()
    hidden_edges = set()

    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            n1, n2 = normals[f1], normals[f2]
            v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]
            
            dot_val = np.dot(n1, n2)
            # 해상도가 높아지면 면 사이 각도가 매우 작아지므로 smooth 기준을 높임
            is_smooth_edge = dot_val > 0.9  
            is_flat_internal = dot_val > 0.999 

            if is_curved_surface and is_smooth_edge:
                if v1 != v2: visible_edges.add(edge)
            else:
                if is_flat_internal: continue
                if v1 or v2: visible_edges.add(edge)
                else: hidden_edges.add(edge)
        else:
            if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)
            else: hidden_edges.add(edge)

    # --- 6. 시각화 (스무스 렌더링) ---
    fig = go.Figure()

    def get_coords(edge_set):
        x_list, y_list, z_list = [], [], []
        for p1, p2 in edge_set:
            pts = rotated_points[[p1, p2]]
            x_list.extend([pts[0][0], pts[1][0], None])
            y_list.extend([pts[0][1], pts[1][1], None])
            z_list.extend([pts[0][2], pts[1][2], None])
        return x_list, y_list, z_list

    # 1. 숨은 선 (더 얇고 연하게)
    xh, yh, zh = get_coords(hidden_edges)
    fig.add_trace(go.Scatter3d(
        x=xh, y=yh, z=zh, mode='lines',
        line=dict(color='rgb(150, 150, 150)', width=2, dash='dash'),
        name='숨은 선', hoverinfo='none'
    ))

    # 2. 면 채우기 (그라데이션 & 스무스 쉐이딩)
    all_mesh_indices = hull.simplices 
    
    # 그라데이션을 위해 intensity 설정 (Z값 기준)
    # 색상을 일정하게 하고 싶으면 intensity를 제거하고 color='...'만 쓰면 되지만,
    # intensity를 쓰면 면의 경계가 덜 보여서 훨씬 매끄러워 보임.
    z_values = rotated_points[:, 2]
    
    fig.add_trace(go.Mesh3d(
        x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
        i=all_mesh_indices[:,0], j=all_mesh_indices[:,1], k=all_mesh_indices[:,2],
        intensity=z_values, # Z값에 따라 미세한 색상 변화 -> 경계선 숨김 효과
        colorscale=[[0, '#A5D8DD'], [1, '#A5D8DD']], # 단일 색상 그라데이션 (매끄러움 유지용)
        showscale=False,
        opacity=0.35,       # 투명도
        flatshading=False,  # [중요] True면 각져보임. False여야 부드러움.
        lighting=dict(
            ambient=0.6,    
            diffuse=0.5,    
            specular=0.8,   # 반짝임
            roughness=0.1,  
            fresnel=1.0     # 외곽선 발광
        ),
        lightposition=dict(x=100, y=100, z=1000), 
        hoverinfo='none', name='면'
    ))

    # 3. 보이는 선 (깔끔하게)
    xv, yv, zv = get_coords(visible_edges)
    fig.add_trace(go.Scatter3d(
        x=xv, y=yv, z=zv, mode='lines',
        line=dict(color='black', width=4),
        name='보이는 선', hoverinfo='none'
    ))

    proj_type = "orthographic" if "교과서 모드" in projection_mode else "perspective"
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white',
            aspectmode='data',
            camera=dict(
                projection=dict(type=proj_type), 
                eye=dict(x=0, y=0, z=1.8),
                up=dict(x=0, y=1, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False,
        paper_bgcolor='white',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
