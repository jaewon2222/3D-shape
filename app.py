import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="도형 생성기 (최종)", layout="wide")
st.title("📐 수학 도형 생성기 (Normal Vector 강제 보정판)")
st.caption("면의 방향을 도형 중심 기준으로 강제로 재정렬하여 오류를 원천 차단했습니다.")

# 스타일 설정
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- 1. 사이드바 설정 ---
with st.sidebar:
    st.header("1. 도형 설정")
    category = st.selectbox("카테고리", ["기둥/뿔/뿔대", "정다면체", "회전체"])
    
    params = {}
    
    if category == "기둥/뿔/뿔대":
        type_ = st.radio("종류", ["각기둥", "각뿔", "각뿔대"], horizontal=True)
        params['n'] = st.number_input("밑면 각수 (n)", 3, 20, 4)
        params['h'] = st.slider("높이", 1.0, 5.0, 3.0)
        
        if type_ == "각기둥":
            r = st.slider("반지름", 0.5, 4.0, 1.5)
            params['top_r'] = params['bottom_r'] = r
        elif type_ == "각뿔":
            params['bottom_r'] = st.slider("밑면 반지름", 0.5, 4.0, 1.5)
            params['top_r'] = 0.0001
        else: # 각뿔대
            # 밑면을 아주 크게 키워도 오류가 나는지 확인해주세요
            params['bottom_r'] = st.slider("밑면 반지름 (Bottom)", 0.5, 6.0, 4.0)
            params['top_r'] = st.slider("윗면 반지름 (Top)", 0.5, 6.0, 1.0)
            
    elif category == "정다면체":
        params['poly_type'] = st.selectbox("종류", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        params['scale'] = st.slider("크기", 1.0, 3.0, 2.0)

    elif category == "회전체":
        rot_type = st.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
        params['n'] = 60 
        params['h'] = st.slider("높이", 1.0, 5.0, 3.0)
        
        if rot_type == "원기둥":
            r = st.slider("반지름", 0.5, 3.0, 1.5)
            params['top_r'] = params['bottom_r'] = r
        elif rot_type == "원뿔":
            params['bottom_r'] = st.slider("밑면 반지름", 0.5, 3.0, 1.5)
            params['top_r'] = 0.0001
        else: # 원뿔대
            params['bottom_r'] = st.slider("밑면 반지름", 0.5, 5.0, 2.5)
            params['top_r'] = st.slider("윗면 반지름", 0.5, 5.0, 1.0)

    st.write("---")
    st.header("2. 뷰 설정")
    # 마우스 회전 이슈를 방지하기 위해 슬라이더 사용 권장
    rot_x = st.slider("X축 회전 (↕)", 0, 360, 20)
    rot_y = st.slider("Y축 회전 (↔)", 0, 360, 30)
    rot_z = st.slider("Z축 회전 (🔄)", 0, 360, 0)
    cam_dist = st.slider("카메라 거리", 3.0, 20.0, 8.0)
    is_perspective = st.checkbox("원근 투영 (Perspective)", value=True)

# --- 2. 도형 데이터 생성 ---

def create_geometry(cat, **p):
    verts = []
    faces = []
    
    # [A] 기둥/뿔/뿔대 & 회전체
    if cat in ["기둥/뿔/뿔대", "회전체"]:
        n = p['n']
        h = p['h']
        tr = p['top_r']
        br = p['bottom_r']
        
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        # 윗면 (z = h/2)
        for t in theta: verts.append([tr * np.cos(t), tr * np.sin(t), h/2])
        # 아랫면 (z = -h/2)
        for t in theta: verts.append([br * np.cos(t), br * np.sin(t), -h/2])
        
        verts = np.array(verts)
        
        # 2-1. 윗면
        faces.append(list(range(n)))
        # 2-2. 아랫면
        faces.append(list(range(2*n-1, n-1, -1)))
        # 2-3. 옆면
        for i in range(n):
            faces.append([i, i+n, ((i+1)%n)+n, (i+1)%n])
            
        return verts, faces

    # [B] 정다면체
    elif cat == "정다면체":
        name = p['poly_type']
        s = p['scale']
        phi = (1 + np.sqrt(5)) / 2
        points = []
        
        if name == "정사면체":
            points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
        elif name == "정육면체":
            points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        elif name == "정팔면체":
            points = [[0,0,1], [0,0,-1], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0]]
        elif name == "정십이면체":
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]: points.append([i, j, k])
            for i in [-1, 1]:
                for j in [-1, 1]:
                    points.append([0, i*phi, j/phi])
                    points.append([j/phi, 0, i*phi])
                    points.append([i*phi, j/phi, 0])
        elif name == "정이십면체":
             for i in [-1,1]:
                 for j in [-1,1]:
                     points.append([0, i, j*phi])
                     points.append([j*phi, 0, i])
                     points.append([i, j*phi, 0])
        
        verts = np.array(points) * s * 0.5
        hull = ConvexHull(verts)
        return verts, hull.simplices

    return np.array([]), []

# --- 3. 메인 연산 및 수정된 가시성 판별 로직 ---

verts, faces = create_geometry(category, **params)

# 회전 행렬
def get_rotation_matrix(x, y, z):
    rad = np.radians([x, y, z])
    c, s = np.cos(rad), np.sin(rad)
    Rx = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
    Ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    Rz = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

rot_mat = get_rotation_matrix(rot_x, rot_y, rot_z)
rotated_verts = verts @ rot_mat.T 

# 카메라 위치
camera_pos = np.array([0, 0, cam_dist])

# [핵심 수정] 가시성 판별 로직 (Visible Surface Determination)
visible_faces_idx = set()
object_center = np.mean(rotated_verts, axis=0) # 도형의 무게중심

for i, face in enumerate(faces):
    face_pts = rotated_verts[face]
    face_center = np.mean(face_pts, axis=0)
    
    # 1. 법선 벡터 계산 (기본 외적)
    v1 = face_pts[1] - face_pts[0]
    v2 = face_pts[-1] - face_pts[0]
    normal = np.cross(v1, v2)
    
    # [강제 보정] 법선 벡터가 도형의 중심에서 바깥쪽을 향하는지 확인
    # "면의 중심" - "도형의 중심" 벡터와 법선 벡터의 내적이 양수여야 함 (같은 방향)
    center_to_face_vec = face_center - object_center
    
    if np.dot(normal, center_to_face_vec) < 0:
        normal = -normal # 반대면 뒤집는다 (무조건 바깥을 보게 함)

    # 2. 카메라 시선과 비교
    if is_perspective:
        view_vec = camera_pos - face_center
    else:
        view_vec = np.array([0, 0, 1])
        
    # 내적 > 0 이면 보임
    if np.dot(normal, view_vec) > 0:
        visible_faces_idx.add(i)

# --- 4. 선 그리기 ---
edge_map = {} 

for f_idx, face in enumerate(faces):
    n_pts = len(face)
    for i in range(n_pts):
        p1, p2 = face[i], face[(i+1)%n_pts]
        key = tuple(sorted((p1, p2)))
        if key not in edge_map:
            edge_map[key] = []
        edge_map[key].append(f_idx)

vis_edges = []
hid_edges = []

for (p1, p2), f_indices in edge_map.items():
    is_visible = False
    
    # 공유하는 면 중 하나라도 보이면 실선
    for f_idx in f_indices:
        if f_idx in visible_faces_idx:
            is_visible = True
            break
    
    pts = rotated_verts[[p1, p2]]
    line_seg = [pts[0], pts[1], [None, None, None]]
    
    if is_visible:
        vis_edges.append(line_seg)
    else:
        hid_edges.append(line_seg)

# --- 5. 시각화 ---
def flatten(seg_list):
    x, y, z = [], [], []
    for s in seg_list:
        x.extend([s[0][0], s[1][0], None])
        y.extend([s[0][1], s[1][1], None])
        z.extend([s[0][2], s[1][2], None])
    return x, y, z

fig = go.Figure()

# 점선 (뒤)
hx, hy, hz = flatten(hid_edges)
fig.add_trace(go.Scatter3d(
    x=hx, y=hy, z=hz, mode='lines',
    line=dict(color='gray', width=3, dash='dash'),
    hoverinfo='none'
))

# 실선 (앞)
vx, vy, vz = flatten(vis_edges)
fig.add_trace(go.Scatter3d(
    x=vx, y=vy, z=vz, mode='lines',
    line=dict(color='black', width=5),
    hoverinfo='none'
))

# 면 색칠 (선택)
try:
    hull = ConvexHull(rotated_verts)
    fig.add_trace(go.Mesh3d(
        x=rotated_verts[:,0], y=rotated_verts[:,1], z=rotated_verts[:,2],
        i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2],
        color='#d0f0fd', opacity=0.1, flatshading=True, hoverinfo='none'
    ))
except:
    pass

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        camera=dict(
            projection=dict(type="perspective" if is_perspective else "orthographic"),
            eye=dict(x=0, y=0, z=cam_dist*0.5),
            up=dict(x=0, y=1, z=0)
        ),
        aspectmode='data',
        dragmode=False
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
