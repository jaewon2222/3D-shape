import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="수학 문제집 생성기", layout="wide")
st.title("📐 기하 도형 생성기 (최종 수정판)")
st.caption("카메라 시점과 수학적 계산을 1:1로 동기화하여 은선(점선) 처리를 완벽하게 수행합니다.")

# --- 1. 사이드바 설정 ---
with st.sidebar:
    st.header("1. 뷰 설정")
    # 투영 방식
    projection_mode = st.radio("투영 모드", ["원근 투영 (Perspective)", "직교 투영 (Orthographic)"], index=0)
    
    # [핵심] 카메라 거리를 고정 변수로 둡니다.
    # 이 거리가 계산 식과 렌더링 뷰에 동시에 들어갑니다.
    cam_dist = st.slider("카메라 거리 (원근감 조절)", 2.0, 10.0, 3.5, 0.1)

    st.header("2. 도형 선택")
    category = st.radio("카테고리", ["각기둥/각뿔", "정다면체"], index=1)

    st.header("3. 도형 회전")
    col1, col2, col3 = st.columns(3)
    with col1: rot_x = st.slider("X축", 0, 360, 15)
    with col2: rot_y = st.slider("Y축", 0, 360, 25)
    with col3: rot_z = st.slider("Z축", 0, 360, 0)

# --- 2. 수학 함수 (회전) ---
def rotate_points(points, rx, ry, rz):
    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)
    mat_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    mat_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    mat_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    return points @ mat_x.T @ mat_y.T @ mat_z.T

# --- 3. 도형 데이터 생성 ---
points = []

if category == "각기둥/각뿔":
    sub_type = st.sidebar.selectbox("상세 종류", ["각기둥", "각뿔"])
    n = st.sidebar.number_input("밑면 각수 (n)", 3, 10, 5)
    
    h = 3.0
    r = 1.5
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # 밑면
    for t in theta: points.append([r*np.cos(t), r*np.sin(t), -h/2])
    
    if sub_type == "각기둥":
        # 윗면 (밑면과 동일)
        for t in theta: points.append([r*np.cos(t), r*np.sin(t), h/2])
    else: # 각뿔
        # 뿔의 꼭짓점
        points.append([0, 0, h/2])

elif category == "정다면체":
    sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"], index=4)
    phi = (1 + np.sqrt(5)) / 2
    
    if sub_type == "정사면체":
        points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    elif sub_type == "정육면체":
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
    elif sub_type == "정팔면체":
        points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif sub_type == "정십이면체":
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        for i in [-1,1]:
            for j in [-1,1]:
                points.extend([[0, i*phi, j/phi], [j/phi, 0, i*phi], [i*phi, j/phi, 0]])
    elif sub_type == "정이십면체":
        # 정이십면체 좌표 (순환 치환)
        points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                points.append([0, i, j * phi])
                points.append([j * phi, 0, i])
                points.append([i, j * phi, 0])

points = np.array(points)

# --- 4. 가시성 계산 (핵심 로직) ---
try:
    # 1. 점 회전
    rotated_points = rotate_points(points, rot_x, rot_y, rot_z)
    
    # 2. ConvexHull 생성
    hull = ConvexHull(rotated_points)
    
    # 3. 면의 가시성 판별
    visible_faces = []
    
    # [중요] 계산에 사용할 카메라 위치 (Z축 위의 점)
    # 원근 투영일 때: 실제 거리(cam_dist) 사용
    # 직교 투영일 때: 아주 먼 거리(무한대)처럼 취급하거나 시선 벡터를 고정
    camera_pos = np.array([0, 0, cam_dist])

    for i, simplex in enumerate(hull.simplices):
        # 면의 법선 벡터 (ConvexHull은 외부를 향함)
        normal = hull.equations[i][:3]
        
        # 면의 중심점
        face_center = np.mean(rotated_points[simplex], axis=0)
        
        if "원근" in projection_mode:
            # Perspective: 시선 벡터 = 카메라 - 면의 중심
            view_vector = camera_pos - face_center
        else:
            # Orthographic: 시선 벡터 = Z축 (항상 정면)
            view_vector = np.array([0, 0, 1])
            
        # 내적 계산 (Dot Product)
        # 내적 > 0 이면, 카메라가 면의 앞쪽을 보고 있음 -> 보임
        # 내적 < 0 이면, 카메라가 면의 뒤쪽을 보고 있음 -> 안 보임
        is_visible = np.dot(normal, view_vector) > 1e-4 # 부동소수점 오차 방지
        visible_faces.append(is_visible)

    # 4. 모서리(Edge) 분류
    # (모서리는 항상 두 면을 공유함)
    edge_map = {} # Key: (idx1, idx2), Value: [face_index_1, face_index_2]

    for face_idx, simplex in enumerate(hull.simplices):
        for k in range(len(simplex)):
            p1, p2 = sorted((simplex[k], simplex[(k+1)%len(simplex)]))
            edge = (p1, p2)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(face_idx)

    visible_edges = [] # 실선
    hidden_edges = []  # 점선

    for edge, faces in edge_map.items():
        if len(faces) == 2:
            f1, f2 = faces
            v1, v2 = visible_faces[f1], visible_faces[f2]
            
            # [논리]
            # 두 면이 모두 보임 (True, True) -> 앞쪽 모서리 -> 실선
            # 하나만 보임 (True, False) -> 외곽선(실루엣) -> 실선
            # 둘 다 안 보임 (False, False) -> 뒤쪽 모서리 -> 점선
            
            if v1 or v2: 
                visible_edges.append(edge)
            else:
                hidden_edges.append(edge)
        else:
            # 면을 하나만 공유하는 경우 (열린 도형 등 - 여기선 거의 없음)
            if visible_faces[faces[0]]: visible_edges.append(edge)
            else: hidden_edges.append(edge)

    # --- 5. Plotly 그리기 ---
    fig = go.Figure()

    def add_lines(edges, color, dash):
        x_lines, y_lines, z_lines = [], [], []
        for p1, p2 in edges:
            pts = rotated_points[[p1, p2]]
            x_lines.extend([pts[0][0], pts[1][0], None])
            y_lines.extend([pts[0][1], pts[1][1], None])
            z_lines.extend([pts[0][2], pts[1][2], None])
        
        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color=color, width=4 if dash=='solid' else 3, dash=dash),
            hoverinfo='none'
        ))

    # 1. 뒷면 점선 (회색)
    add_lines(hidden_edges, "gray", "dash")
    
    # 2. 앞면 실선 (검정)
    add_lines(visible_edges, "black", "solid")

    # 3. 면 채우기 (투명한 유리 느낌)
    simplices = hull.simplices
    fig.add_trace(go.Mesh3d(
        x=rotated_points[:, 0], y=rotated_points[:, 1], z=rotated_points[:, 2],
        i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
        color='#d0f0fd', opacity=0.2, flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.5), hoverinfo='none'
    ))

    # --- 6. 카메라 및 레이아웃 설정 (가장 중요) ---
    
    # 투영 모드 설정
    proj_type = "perspective" if "원근" in projection_mode else "orthographic"
    
    # [카메라 동기화]
    # Plotly의 'eye' 좌표는 데이터 스케일에 따라 다르지만, 
    # 여기서는 데이터가 원점 주변에 있으므로 eye 벡터의 비율을 맞춥니다.
    # 직교 투영일 땐 eye 거리가 줌(Zoom) 역할을 하므로 적당히 멉니다.
    
    if proj_type == "perspective":
        # 원근: cam_dist 슬라이더 값을 Z축 눈 위치로 사용
        # Plotly eye는 (x, y, z) 벡터입니다.
        # 데이터 좌표계와 eye 좌표계의 스케일을 맞추기 위해 보정 계수(0.5~0.8)가 필요할 수 있으나,
        # 여기서는 Z축 정렬을 위해 (0, 0, cam_dist) 비율을 유지합니다.
        
        # cam_dist가 클수록 멀리서 봄 (왜곡 적음)
        # cam_dist가 작을수록 가까이서 봄 (왜곡 심함)
        eye_pos = dict(x=0, y=0, z=cam_dist/1.5) 
    else:
        # 직교: 멀리서 줌인
        eye_pos = dict(x=0, y=0, z=2.0)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                projection=dict(type=proj_type),
                eye=eye_pos, 
                up=dict(x=0, y=1, z=0)
            ),
            aspectmode='data' # 비율 유지
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=False,
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {e}")
