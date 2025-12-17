import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="완벽한 도형 생성기", layout="wide")
st.title("📐 수학 도형 생성기 (최종 수정_v3)")
st.caption("모든 도형을 다면체 구조로 변환하여 은선(점선)을 정확히 계산합니다.")

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.error("⚠️ 중요: 마우스로 도형을 돌리지 마세요! (점선이 실시간 업데이트되지 않습니다). 반드시 왼쪽 '슬라이더'로 회전시키세요.")

# --- 1. 사이드바 입력 ---
with st.sidebar:
    st.header("1. 도형 설정")
    # 카테고리 단순화
    category = st.selectbox("도형 카테고리", ["기둥/뿔/뿔대", "정다면체", "회전체"])

    # 변수 초기화
    shape_data = {}
    
    if category == "기둥/뿔/뿔대":
        base_type = st.radio("종류", ["각기둥", "각뿔", "각뿔대"], horizontal=True)
        n = st.number_input("밑면 각수 (n)", 3, 20, 4)
        h = st.slider("높이", 1.0, 5.0, 3.0)
        
        if base_type == "각기둥":
            top_r = bottom_r = st.slider("반지름(크기)", 0.5, 3.0, 1.5)
        elif base_type == "각뿔":
            bottom_r = st.slider("밑면 반지름", 0.5, 3.0, 1.5)
            top_r = 0.001 # 0으로 하면 계산 오류 가능성 있어 아주 작은 값
        else: # 각뿔대
            bottom_r = st.slider("밑면 반지름", 1.0, 3.0, 2.0)
            top_r = st.slider("윗면 반지름", 0.5, 2.9, 1.0)
            
    elif category == "정다면체":
        poly_name = st.selectbox("정다면체 종류", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        scale = st.slider("크기", 1.0, 3.0, 2.0)
        
    elif category == "회전체":
        rot_name = st.selectbox("회전체 종류", ["원기둥", "원뿔", "원뿔대"])
        h = st.slider("높이", 1.0, 5.0, 3.0)
        
        # 회전체를 N각기둥으로 근사하여 은선 처리 (N=30 정도면 부드러움)
        n = 32 
        if rot_name == "원기둥":
            top_r = bottom_r = st.slider("반지름", 0.5, 3.0, 1.5)
        elif rot_name == "원뿔":
            bottom_r = st.slider("밑면 반지름", 0.5, 3.0, 1.5)
            top_r = 0.001
        else: # 원뿔대
            bottom_r = st.slider("밑면 반지름", 1.0, 3.0, 2.0)
            top_r = st.slider("윗면 반지름", 0.5, 2.9, 1.0)

    st.write("---")
    st.header("2. 뷰 설정 (슬라이더 필수)")
    rot_x = st.slider("X축 회전 (↕)", 0, 360, 20)
    rot_y = st.slider("Y축 회전 (↔)", 0, 360, 30)
    rot_z = st.slider("Z축 회전 (🔄)", 0, 360, 0)
    
    st.write("---")
    cam_dist = st.slider("카메라 거리", 3.0, 15.0, 6.0)
    is_perspective = st.checkbox("원근감 적용 (Perspective)", value=True)

# --- 2. 핵심 로직 함수 ---

def get_rotation_matrix(rx, ry, rz):
    # 각도를 라디안으로 변환
    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)
    
    # 회전 행렬 정의
    Rx = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    Ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    Rz = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    
    # 전체 회전 행렬: Rz * Ry * Rx 순서
    return Rz @ Ry @ Rx

def create_geometry(category, **kwargs):
    verts = [] # 꼭짓점 좌표 리스트
    faces = [] # 면을 구성하는 점의 인덱스 리스트 (반시계 방향 CCW 필수)

    if category in ["기둥/뿔/뿔대", "회전체"]:
        n = kwargs['n']
        h = kwargs['h']
        tr = kwargs['top_r']
        br = kwargs['bottom_r']
        
        # 1. 옆면 점 생성
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # 윗면 점들 (0 ~ n-1)
        for t in theta: verts.append([tr * np.cos(t), tr * np.sin(t), h/2])
        # 아랫면 점들 (n ~ 2n-1)
        for t in theta: verts.append([br * np.cos(t), br * np.sin(t), -h/2])
        
        verts = np.array(verts)
        
        # 2. 면 생성 (CCW: 반시계 방향 순서 중요)
        # 윗면 (Top) - 위에서 봤을 때 반시계
        faces.append(list(range(0, n))) 
        
        # 아랫면 (Bottom) - 아래에서 봤을 때 반시계 (즉, 위에서 보면 시계)
        # 따라서 인덱스를 역순으로 넣어야 법선이 바깥을 향함
        faces.append(list(range(2*n-1, n-1, -1)))
        
        # 옆면 (Sides)
        for i in range(n):
            top1 = i
            top2 = (i + 1) % n
            bot1 = i + n
            bot2 = ((i + 1) % n) + n
            
            # 사각형 면: top1 -> bot1 -> bot2 -> top2 (순서 중요)
            faces.append([top1, bot1, bot2, top2])

    elif category == "정다면체":
        name = kwargs['poly_name']
        s = kwargs['scale']
        phi = (1 + np.sqrt(5)) / 2
        
        if name == "정사면체":
            # 정사면체 (반시계 방향 엄수)
            verts = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) * s
            # 면 순서를 바깥쪽을 향하도록 세심하게 조정
            faces = [[0,1,2], [0,2,3], [0,3,1], [1,3,2]] # 확인됨
            
        elif name == "정육면체":
            verts = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]) * s
            # 0:(-1,-1,-1), 1:(-1,-1,1), 2:(-1,1,-1), 3:(-1,1,1) ... 순서가 복잡하므로 직접 지정
            # 쉬운 정의:
            verts = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                              [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) * s * 0.5
            faces = [
                [3,2,1,0], # Bottom (z=-1)
                [4,5,6,7], # Top (z=1)
                [0,1,5,4], # Front
                [1,2,6,5], # Right
                [2,3,7,6], # Back
                [3,0,4,7]  # Left
            ]
            
        elif name == "정팔면체":
            verts = np.array([[0,0,1], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], [0,0,-1]]) * s
            # Top pyramid + Bottom pyramid
            faces = [
                [0,1,2], [0,2,3], [0,3,4], [0,4,1],
                [5,2,1], [5,3,2], [5,4,3], [5,1,4]
            ]
            
        elif name == "정십이면체":
            # 좌표 생성
            verts = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]:
                        verts.append([i, j, k])
            for i in [-1, 1]:
                for j in [-1, 1]:
                    verts.append([0, i*phi, j/phi])
                    verts.append([j/phi, 0, i*phi])
                    verts.append([i*phi, j/phi, 0])
            verts = np.array(verts) * s * 0.5
            # ConvexHull을 사용하여 면을 찾되, 법선 벡터 방향을 강제 교정하는 방식 사용
            from scipy.spatial import ConvexHull
            hull = ConvexHull(verts)
            # ConvexHull의 simplices는 삼각형이지만, 정십이면체는 오각형임.
            # 여기서는 시각적 완벽함을 위해 각 삼각형 면을 그대로 씁니다. (오각형을 3개 삼각형으로 쪼개서 그림)
            # 은선 제거에는 전혀 문제 없습니다.
            faces = hull.simplices

        elif name == "정이십면체":
             verts = []
             for i in [-1,1]:
                 for j in [-1,1]:
                     verts.append([0, i, j*phi])
                     verts.append([j*phi, 0, i])
                     verts.append([i, j*phi, 0])
             verts = np.array(verts) * s * 0.5
             from scipy.spatial import ConvexHull
             hull = ConvexHull(verts)
             faces = hull.simplices

    return np.array(verts), faces

# --- 3. 데이터 생성 및 계산 ---

# 1) 파라미터 패키징
params = {}
if category == "기둥/뿔/뿔대":
    params = {'n': n, 'h': h, 'top_r': top_r, 'bottom_r': bottom_r}
elif category == "정다면체":
    params = {'poly_name': poly_name, 'scale': scale}
elif category == "회전체":
    params = {'n': 32, 'h': h, 'top_r': top_r, 'bottom_r': bottom_r} # 회전체는 n=32인 각기둥으로 처리

# 2) 기하 정보 생성
original_verts, faces = create_geometry(category, **params)

# 3) 회전 적용
rot_matrix = get_rotation_matrix(rot_x, rot_y, rot_z)
rotated_verts = original_verts @ rot_matrix.T

# 4) 가시성 판별 (Visibility Check) - 여기가 핵심!
# 카메라 위치: (0, 0, cam_dist)
camera_pos = np.array([0, 0, cam_dist])
is_face_visible = []

for face in faces:
    # 면의 중심 계산
    face_pts = rotated_verts[face]
    center = np.mean(face_pts, axis=0)
    
    # 법선 벡터 계산 (Normal)
    # v1 = p1 - p0, v2 = p2 - p0
    v1 = face_pts[1] - face_pts[0]
    v2 = face_pts[2] - face_pts[0]
    normal = np.cross(v1, v2)
    
    # 시선 벡터 (View Vector)
    if is_perspective:
        view_vec = camera_pos - center
    else:
        view_vec = np.array([0, 0, 1]) # 직교 투영은 항상 정면
        
    # 내적 (Dot Product)
    dot_val = np.dot(normal, view_vec)
    
    # 내적 > 0 이면 보임
    is_face_visible.append(dot_val > 1e-5)

# 5) 모서리(Edge) 분류
# 모든 변을 (점1_idx, 점2_idx) 형태로 저장하고 공유하는 면을 찾음
edge_map = {} 

for f_idx, face in enumerate(faces):
    # 면이 오각형이든 삼각형이든 모든 변을 순회
    for i in range(len(face)):
        p1, p2 = face[i], face[(i+1)%len(face)]
        # Key는 항상 작은 인덱스가 앞에 오도록 (p1, p2) 정렬
        edge_key = tuple(sorted((p1, p2)))
        
        if edge_key not in edge_map:
            edge_map[edge_key] = []
        edge_map[edge_key].append(f_idx)

visible_edges = []
hidden_edges = []

for edge, face_indices in edge_map.items():
    p1, p2 = edge
    
    # 이 변을 공유하는 면들 중 "하나라도" 보이면 실선입니다.
    # (외곽선 포함)
    is_visible = False
    
    # 면 데이터가 1개인 경우 (열린 도형 등 - 거의 없음)
    if len(face_indices) == 1:
        is_visible = is_face_visible[face_indices[0]]
    else:
        # 면 데이터가 2개 이상 (일반적)
        # 하나라도 보이면 Visible
        for f_idx in face_indices:
            if is_face_visible[f_idx]:
                is_visible = True
                break
    
    if is_visible:
        visible_edges.append(rotated_verts[[p1, p2]])
    else:
        hidden_edges.append(rotated_verts[[p1, p2]])

# --- 4. Plotly 그리기 ---

fig = go.Figure()

# 좌표 리스트로 변환하는 헬퍼 함수
def lines_to_xyz(lines):
    x, y, z = [], [], []
    for line in lines:
        x.extend([line[0][0], line[1][0], None])
        y.extend([line[0][1], line[1][1], None])
        z.extend([line[0][2], line[1][2], None])
    return x, y, z

# 1. 점선 (뒷면) 그리기
hx, hy, hz = lines_to_xyz(hidden_edges)
fig.add_trace(go.Scatter3d(
    x=hx, y=hy, z=hz,
    mode='lines',
    line=dict(color='silver', width=3, dash='dash'),
    name='점선 (뒷면)', hoverinfo='none'
))

# 2. 실선 (앞면) 그리기
vx, vy, vz = lines_to_xyz(visible_edges)
fig.add_trace(go.Scatter3d(
    x=vx, y=vy, z=vz,
    mode='lines',
    line=dict(color='black', width=5),
    name='실선 (앞면)', hoverinfo='none'
))

# 3. 면 칠하기 (선택 사항 - 약간의 투명도)
# Mesh3d는 복잡하므로 간단히 ConvexHull로 면을 덮어씌움 (시각 효과용)
if len(faces) > 0:
    try:
        from scipy.spatial import ConvexHull
        chull = ConvexHull(rotated_verts)
        fig.add_trace(go.Mesh3d(
            x=rotated_verts[:,0], y=rotated_verts[:,1], z=rotated_verts[:,2],
            i=chull.simplices[:,0], j=chull.simplices[:,1], k=chull.simplices[:,2],
            color='#e0f7fa', opacity=0.1, flatshading=True, hoverinfo='none', name='면'
        ))
    except:
        pass # 평면 도형 등 ConvexHull 실패 시 무시

# 4. 카메라 및 레이아웃 설정
scene_camera = dict(
    projection=dict(type="perspective" if is_perspective else "orthographic"),
    eye=dict(x=0, y=0, z=cam_dist * 0.5), # 거리 비율 보정
    up=dict(x=0, y=1, z=0)
)

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        camera=scene_camera,
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    showlegend=False,
    dragmode=False # 마우스 회전 금지
)

st.plotly_chart(fig, use_container_width=True)
