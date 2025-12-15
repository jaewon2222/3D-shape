import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="수학 문제집 생성기", layout="wide")
st.title("📐 수학 문제집 도형 생성기 (실루엣 알고리즘)")
st.caption("교과서에 나오는 것처럼 '보이는 곡면'은 외곽선만, '각진 모서리'는 선명하게 그립니다.")

# --- 1. 사이드바 설정 ---
with st.sidebar:
    st.header("1. 보기 설정")
    projection_mode = st.radio(
        "투영 방식", 
        ["교과서 모드 (직교 투영)", "현실 모드 (원근 투영)"],
        index=0
    )

    st.header("2. 도형 선택")
    category = st.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (매끈함)", "정다면체"])

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
    elif sub_type == "각뿔": rt = 0.001 # 0이면 ConvexHull 오류 가능성 있어 아주 작은 값 사용
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, 1.9, 1.0)
    
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    # 윗면, 아랫면 점 생성
    for t in theta: points.append([rt*np.cos(t), rt*np.sin(t), h/2])
    for t in theta: points.append([rb*np.cos(t), rb*np.sin(t), -h/2])

elif category == "원기둥/원뿔 (매끈함)":
    is_curved_surface = True
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    n = 60 # 곡면을 부드럽게 표현하기 위한 점의 개수
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

    # 각 면의 법선 벡터 계산
    normals = []
    for eq in hull.equations:
        n_vec = eq[:3]
        normals.append(n_vec / np.linalg.norm(n_vec))
    normals = np.array(normals)

    # 카메라 설정
    # 교과서 모드(Orthographic)는 뷰 벡터가 항상 Z축과 평행
    # 현실 모드(Perspective)는 카메라 위치에서 각 면의 중심으로 향하는 벡터 계산
    camera_pos = np.array([0, 0, 100.0]) # 멀리서 바라보는 것 처럼 설정
    visible_faces_mask = []

    for i, simplex in enumerate(hull.simplices):
        if "교과서 모드" in projection_mode:
            # 직교 투영: 법선의 Z값이 양수면 보임 (화면 밖으로 튀어나오는 방향)
            is_visible = normals[i][2] > 0
        else:
            # 원근 투영 효과
            face_center = np.mean(rotated_points[simplex], axis=0)
            view_vector = camera_pos - face_center
            is_visible = np.dot(view_vector, normals[i]) > 0
        visible_faces_mask.append(is_visible)

    # 엣지 정보 수집
    edge_to_faces = {}
    for face_idx, simplex in enumerate(hull.simplices):
        n_pts = len(simplex)
        for k in range(n_pts):
            p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))
            edge = (p1, p2)
            if edge not in edge_to_faces: edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # --- 5. 선 그리기 로직 (수정된 핵심 알고리즘) ---
    visible_edges = set()
    hidden_edges = set()

    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            n1, n2 = normals[f1], normals[f2]
            v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]
            
            # 두 면 사이의 각도 계산 (내적)
            dot_val = np.dot(n1, n2)
            # 1.0에 가까울수록 평평하게 이어진 면 (곡면의 일부 혹은 평면 위의 분할선)
            is_smooth_edge = dot_val > 0.8 
            is_flat_internal = dot_val > 0.999 # 완전히 평평한 면 위의 선 (삼각형 분할선)

            if is_curved_surface and is_smooth_edge:
                # [원기둥/원뿔 해결책]
                # 곡면의 부드러운 모서리는 '실루엣'일 때만 그린다.
                # v1 != v2 : 하나는 보이고 하나는 안 보일 때 (경계선)
                if v1 != v2:
                    visible_edges.add(edge)
                # 곡면 내부의 선(둘 다 보이거나 둘 다 안 보임)은 절대 그리지 않음 -> 바코드 제거됨
            
            else:
                # [각기둥/각뿔 및 뚜껑 모서리 해결책]
                # 1. 평면 내부의 쓸데없는 대각선은 제거
                if is_flat_internal:
                    continue
                
                # 2. 각진 모서리 처리
                if v1 or v2:
                    visible_edges.add(edge) # 둘 중 하나라도 보이면 실선 (외곽선 포함)
                else:
                    hidden_edges.add(edge)  # 둘 다 안 보이면 점선

        else:
            # 면이 하나뿐인 경계선 (거의 없지만 예외처리)
            if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)
            else: hidden_edges.add(edge)

    # --- 6. 시각화 ---
    fig = go.Figure()

    # 좌표 추출 함수
    def get_coords(edge_set):
        x_list, y_list, z_list = [], [], []
        for p1, p2 in edge_set:
            pts = rotated_points[[p1, p2]]
            x_list.extend([pts[0][0], pts[1][0], None])
            y_list.extend([pts[0][1], pts[1][1], None])
            z_list.extend([pts[0][2], pts[1][2], None])
        return x_list, y_list, z_list

    # 숨은 선 (점선)
    xh, yh, zh = get_coords(hidden_edges)
    fig.add_trace(go.Scatter3d(
        x=xh, y=yh, z=zh, mode='lines',
        line=dict(color='silver', width=3, dash='dash'),
        name='숨은 선', hoverinfo='none'
    ))

    # 보이는 선 (실선)
    xv, yv, zv = get_coords(visible_edges)
    fig.add_trace(go.Scatter3d(
        x=xv, y=yv, z=zv, mode='lines',
        line=dict(color='black', width=5),
        name='보이는 선', hoverinfo='none'
    ))

    # 면 채우기 (흰색, 그림자 없이)
    visible_mesh_indices = [hull.simplices[i] for i, vis in enumerate(visible_faces_mask) if vis]
    if visible_mesh_indices:
        visible_mesh_indices = np.array(visible_mesh_indices)
        fig.add_trace(go.Mesh3d(
            x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
            i=visible_mesh_indices[:,0], j=visible_mesh_indices[:,1], k=visible_mesh_indices[:,2],
            color='white', opacity=0.15,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), # 완전한 무광 흰색
            hoverinfo='none', name='면'
        ))

    # 뷰 설정
    proj_type = "orthographic" if "교과서 모드" in projection_mode else "perspective"
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white',
            aspectmode='data',
            camera=dict(
                projection=dict(type=proj_type), 
                eye=dict(x=0, y=0, z=2.0), # 줌 레벨 조정
                up=dict(x=0, y=1, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False,
        paper_bgcolor='white',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"도형을 생성할 수 없습니다. 설정을 변경해보세요. (Error: {e})")
    st.info("팁: 각형(n)이 너무 작거나 반지름이 0이면 도형이 만들어지지 않을 수 있습니다.")
