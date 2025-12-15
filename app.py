import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="수학 문제집 생성기", layout="wide")
st.title("💎 수학 문제집 도형 생성기 (은선 처리 완벽 수정판)")
st.caption("뒷면 모서리가 실선으로 잘못 나오는 계산 오류를 수정했습니다.")

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
    
    # 면의 법선 벡터 (ConvexHull은 기본적으로 바깥쪽을 향함)
    normals = []
    for eq in hull.equations:
        normals.append(eq[:3])
    normals = np.array(normals)

    # [핵심 수정] 투영 모드에 따른 시선 벡터 계산
    # 이전 코드의 버그: 직교 투영인데도 시선 벡터를 위치에 따라 다르게 계산하여 왜곡 발생
    visible_faces_mask = []
    
    camera_z_ortho = 1000.0
    camera_pos_persp = np.array([0, 0, 10.0])

    for i, simplex in enumerate(hull.simplices):
        normal = normals[i]
        
        if "교과서 모드" in projection_mode:
            # 직교 투영: 시선은 항상 Z축과 평행 (모든 면에 대해 동일한 뷰 벡터)
            view_vector = np.array([0, 0, 1]) 
        else:
            # 원근 투영: 시선은 카메라와 면의 중심을 잇는 선
            face_center = np.mean(rotated_points[simplex], axis=0)
            view_vector = camera_pos_persp - face_center
        
        # 내적 계산 (양수면 보이는 면, 음수면 뒷면)
        dot_product = np.dot(view_vector, normal)
        
        # [수정] 0에 아주 가까운 경우(90도 측면) 깜빡거림 방지를 위해 약간의 여유(epsilon)를 둠
        is_visible = dot_product > 1e-5 
        visible_faces_mask.append(is_visible)

    edge_to_faces = {}
    for face_idx, simplex in enumerate(hull.simplices):
        n_pts = len(simplex)
        for k in range(n_pts):
            p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))
            edge = (p1, p2)
            if edge not in edge_to_faces: edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # --- 5. 선 그리기 로직 (엄격한 판정) ---
    visible_edges = set()
    hidden_edges = set()

    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]
            n1, n2 = normals[f1], normals[f2]
            
            # 곡면 판정
            dot_val = np.dot(n1, n2)
            is_smooth_edge = dot_val > 0.96 # 매끄러운 곡면의 내부 선
            is_flat_internal = dot_val > 0.999 # 평면 내부의 대각선 등

            if is_curved_surface and is_smooth_edge:
                # 곡면(원기둥 등)에서는 '실루엣(경계)'만 그린다
                # 하나는 보이고 하나는 안 보일 때 -> 실루엣 (실선)
                if v1 != v2: 
                    visible_edges.add(edge)
                # 둘 다 보이면 -> 매끄러운 앞면이므로 선을 그리지 않음 (통과)
                # 둘 다 안 보이면 -> 매끄러운 뒷면이므로 선을 그리지 않음 (통과)
            else:
                # 각진 도형 (육면체 등)
                if is_flat_internal: continue # 평면 내부 선 제거
                
                if v1 and v2:
                    # 두 면이 다 보임 -> 확실한 앞면 (실선)
                    visible_edges.add(edge)
                elif v1 or v2:
                    # 하나만 보임 -> 외곽선 (실선)
                    visible_edges.add(edge)
                else:
                    # 둘 다 안 보임 -> 확실한 뒷면 (점선)
                    hidden_edges.add(edge)
        else:
            # 면이 하나뿐인 경계 (거의 없음)
            if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)
            else: hidden_edges.add(edge)

    # --- 6. 시각화 ---
    fig = go.Figure()

    def get_coords(edge_set):
        x_list, y_list, z_list = [], [], []
        for p1, p2 in edge_set:
            pts = rotated_points[[p1, p2]]
            x_list.extend([pts[0][0], pts[1][0], None])
            y_list.extend([pts[0][1], pts[1][1], None])
            z_list.extend([pts[0][2], pts[1][2], None])
        return x_list, y_list, z_list

    # 1. 숨은 선 (뒤에 있으므로 먼저 그림)
    xh, yh, zh = get_coords(hidden_edges)
    fig.add_trace(go.Scatter3d(
        x=xh, y=yh, z=zh, mode='lines',
        line=dict(color='rgb(150, 150, 150)', width=3, dash='dash'),
        name='숨은 선', hoverinfo='none'
    ))

    # 2. 면 채우기 (깨끗한 유리)
    all_mesh_indices = hull.simplices 
    
    fig.add_trace(go.Mesh3d(
        x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
        i=all_mesh_indices[:,0], j=all_mesh_indices[:,1], k=all_mesh_indices[:,2],
        color='#d4f1f4',    
        opacity=0.3,       
        flatshading=False,  
        lighting=dict(
            ambient=0.9,
            diffuse=0.1,    
            specular=0.4,   
            roughness=0.1,  
            fresnel=2.0     
        ),
        hoverinfo='none', name='면'
    ))

    # 3. 보이는 선 (맨 위에 그림)
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
