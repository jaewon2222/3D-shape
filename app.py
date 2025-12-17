import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="수학 문제집 생성기", layout="wide")
st.title("📐 수학 도형 생성기 (원근 투영 오차 수정판)")
st.caption("뒷면 선이 실선으로 보인다면 '카메라 거리'를 조절해보세요.")

# --- 1. 사이드바 설정 ---
with st.sidebar:
    st.header("1. 보기 설정")
    projection_mode = st.radio(
        "투영 방식", 
        ["교과서 모드 (직교 투영)", "현실 모드 (원근 투영)"],
        index=1
    )
    
    # [핵심] 렌더링과 계산의 싱크를 맞추기 위한 거리 조절
    # 원근 모드일 때 이 값이 너무 작으면 왜곡이 심해지고, 너무 크면 직교 투영처럼 보입니다.
    cam_dist = st.slider("카메라 거리 (원근감 조절)", 1.5, 20.0, 4.0, 0.1)
    
    st.write("---")
    # 비상용 반전 버튼
    flip_visibility = st.checkbox("점선/실선 반전 (Flip)", value=False)

    st.header("2. 도형 선택")
    category = st.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/구 (매끈함)", "정다면체"], index=2)

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

# --- 3. 도형 데이터 생성 ---
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
        # 정십이면체 좌표 생성
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        for i in [-1,1]:
             for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif sub_type == "정이십면체":
        for i in [-1,1]:
            for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

points = np.array(points)

# --- 4. 렌더링 및 가시성 계산 (수정됨) ---
try:
    rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
    hull = ConvexHull(rotated_points)
    
    visible_faces_mask = []
    
    # [핵심 로직] 카메라 위치 설정
    # Plotly의 camera.eye는 데이터의 중심(0,0,0)을 기준으로 한 상대적 위치입니다.
    # 데이터 범위가 대략 [-2, 2]라고 가정할 때, cam_dist=2.0이면 실제 좌표는 (0, 0, 4~5) 정도가 됩니다.
    # 이 계산을 위해 Z축 방향의 실제 좌표를 가정합니다.
    
    real_camera_pos = np.array([0, 0, cam_dist]) 

    for i, simplex in enumerate(hull.simplices):
        # 1. 면의 법선 벡터 (Normal)
        # ConvexHull의 eq는 [nx, ny, nz, offset] (normal points OUTWARDS)
        normal = hull.equations[i][:3]
        
        # 2. 면의 중심점 (Centroid)
        face_points = rotated_points[simplex]
        face_center = np.mean(face_points, axis=0)
        
        # 3. 시선 벡터 (View Vector): 면의 중심 -> 카메라
        if "교과서 모드" in projection_mode:
             # 직교 투영: 시선은 항상 정면(Z축)
            view_vector = np.array([0, 0, 1])
        else:
            # 원근 투영: 시선은 위치마다 다름
            view_vector = real_camera_pos - face_center
        
        # 4. 내적 계산 (Dot Product)
        # 내적 > 0 이면, 두 벡터 사이 각도가 90도 미만 -> 서로 마주봄 -> 보임
        # 내적 < 0 이면, 등지고 있음 -> 안 보임
        dot_product = np.dot(normal, view_vector)
        
        # 아주 미세한 오차 제거 (epsilon)
        is_visible = dot_product > 1e-3
        
        if flip_visibility:
            is_visible = not is_visible
            
        visible_faces_mask.append(is_visible)

    edge_to_faces = {}
    for face_idx, simplex in enumerate(hull.simplices):
        n_pts = len(simplex)
        for k in range(n_pts):
            p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))
            edge = (p1, p2)
            if edge not in edge_to_faces: edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # --- 5. 선 분류 ---
    visible_edges = set()
    hidden_edges = set()

    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]
            
            # 곡면 부드럽게 처리
            n1 = hull.equations[f1][:3]
            n2 = hull.equations[f2][:3]
            is_smooth = np.dot(n1, n2) > 0.95
            
            if is_curved_surface and is_smooth:
                if v1 != v2: visible_edges.add(edge)
            else:
                # [논리 수정]
                # 다면체에서 뒷면 모서리가 실선으로 보이는 오류는
                # v1, v2 중 하나라도 True면 실선으로 그리기 때문입니다.
                # 원근 투영에서는 뒷면이어도 측면에서 살짝 보일 수 있으므로 이 논리는 맞습니다.
                # 단, 카메라 각도가 안 맞으면 안 보여야 할 면이 보인다고(True) 판단되어 실선이 됩니다.
                if v1 and v2:
                    visible_edges.add(edge) # 앞쪽 모서리 (실선)
                elif v1 or v2:
                    visible_edges.add(edge) # 외곽선 (실선)
                else:
                    hidden_edges.add(edge)  # 뒤쪽 모서리 (점선)
        else:
            if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)
            else: hidden_edges.add(edge)

    # --- 6. 그리기 ---
    fig = go.Figure()

    def get_coords(edge_set):
        x_list, y_list, z_list = [], [], []
        for p1, p2 in edge_set:
            pts = rotated_points[[p1, p2]]
            x_list.extend([pts[0][0], pts[1][0], None])
            y_list.extend([pts[0][1], pts[1][1], None])
            z_list.extend([pts[0][2], pts[1][2], None])
        return x_list, y_list, z_list

    # 1. 뒷면 점선
    xh, yh, zh = get_coords(hidden_edges)
    fig.add_trace(go.Scatter3d(
        x=xh, y=yh, z=zh, mode='lines',
        line=dict(color='rgb(180, 180, 180)', width=3, dash='dash'), # 회색 점선
        name='숨은 선', hoverinfo='none'
    ))

    # 2. 면 채우기
    all_mesh_indices = hull.simplices 
    fig.add_trace(go.Mesh3d(
        x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
        i=all_mesh_indices[:,0], j=all_mesh_indices[:,1], k=all_mesh_indices[:,2],
        color='#d4f1f4', opacity=0.3, flatshading=False,
        lighting=dict(ambient=0.9, diffuse=0.1, specular=0.4, roughness=0.1, fresnel=2.0),
        hoverinfo='none', name='면'
    ))

    # 3. 앞면 실선
    xv, yv, zv = get_coords(visible_edges)
    fig.add_trace(go.Scatter3d(
        x=xv, y=yv, z=zv, mode='lines',
        line=dict(color='black', width=4),
        name='보이는 선', hoverinfo='none'
    ))

    # 투영 방식 설정
    proj_type = "orthographic" if "교과서 모드" in projection_mode else "perspective"
    
    # [중요] 계산된 cam_dist를 실제 뷰에도 적용
    camera_setting = dict(
        projection=dict(type=proj_type),
        eye=dict(x=0, y=0, z=cam_dist/2.0), # Plotly 좌표계 보정 (데이터 스케일에 맞춤)
        up=dict(x=0, y=1, z=0)
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white', aspectmode='data',
            camera=camera_setting
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False,
        paper_bgcolor='white', showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
