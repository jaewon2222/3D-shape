import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="수학 문제집 생성기", layout="wide")
st.title("💎 수학 문제집 도형 생성기 (최종_매끈함_수정.py)")
st.caption("깨진 유리 느낌을 없애고, 매끈한 플라스틱/유리 질감으로 변경했습니다.")

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

# --- 3. 도형 데이터 생성 (고해상도) ---
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
        u_steps = 60; v_steps = 30 
        u = np.linspace(0, 2 * np.pi, u_steps)
        v = np.linspace(0, np.pi, v_steps)
        for theta in u:
            for phi in v:
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                points.append([x, y, z])
    else:
        # 원기둥 해상도 충분히 확보
        n = 80 
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
        normals.append(eq[:3])
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

    # --- 5. 선 그리기 로직 (스무스 엣지 제거 강화) ---
    visible_edges = set()
    hidden_edges = set()

    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            n1, n2 = normals[f1], normals[f2]
            v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]
            
            dot_val = np.dot(n1, n2)
            
            # [중요] 곡면 판정 기준을 더 엄격하게 (0.9 -> 0.96)
            # 해상도가 높아서 인접 면 각도가 매우 작으므로, 이 값을 높여야
            # 원기둥 옆면의 세로줄들이 안 보이고 매끈해짐.
            is_smooth_edge = dot_val > 0.96
            is_flat_internal = dot_val > 0.999 

            if is_curved_surface and is_smooth_edge:
                # 외곽선(실루엣)만 그림
                if v1 != v2: visible_edges.add(edge)
            else:
                if is_flat_internal: continue
                if v1 or v2: visible_edges.add(edge)
                else: hidden_edges.add(edge)
        else:
            if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)
            else: hidden_edges.add(edge)

    # --- 6. 시각화 (깨짐 방지 설정) ---
    fig = go.Figure()

    def get_coords(edge_set):
        x_list, y_list, z_list = [],
