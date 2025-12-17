import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="완벽한 도형 생성기", layout="wide")
st.title("📐 수학 도형 생성기 (최종_완성본)")
st.caption("✅ 해결됨: 1. 원기둥 검은 띠 제거 2. 각기둥 앞면 점선 오류 3. 찌그러짐 방지")

# 스타일 설정 (버튼 등)
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.warning("⚠️ **주의:** 점선(겨냥도)은 수학적으로 계산된 고정 이미지입니다. **마우스로 돌리면 점선 위치가 틀어지니**, 반드시 **왼쪽 슬라이더**로만 회전하세요.")

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
            r = st.slider("반지름", 0.5, 3.0, 1.5)
            params['top_r'] = params['bottom_r'] = r
        elif type_ == "각뿔":
            params['bottom_r'] = st.slider("밑면 반지름", 0.5, 3.0, 1.5)
            params['top_r'] = 0.0001 
        else: # 각뿔대
            params['bottom_r'] = st.slider("밑면 반지름 (Bottom)", 0.5, 4.0, 2.5)
            params['top_r'] = st.slider("윗면 반지름 (Top)", 0.5, 4.0, 1.0)
            
    elif category == "정다면체":
        params['poly_type'] = st.selectbox("종류", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        params['scale'] = st.slider("크기", 1.0, 3.0, 2.0)
        params['n'] = 0 

    elif category == "회전체":
        rot_type = st.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
        # 회전체는 부드럽게 보이기 위해 n=60 고정 (감옥 창살 방지 로직 적용됨)
        params['n'] = 60 
        params['h'] = st.slider("높이", 1.0, 5.0, 3.0)
        
        if rot_type == "원기둥":
            r = st.slider("반지름", 0.5, 3.0, 1.5)
            params['top_r'] = params['bottom_r'] = r
        elif rot_type == "원뿔":
            params['bottom_r'] = st.slider("밑면 반지름", 0.5, 3.0, 1.5)
            params['top_r'] = 0.0001
        else: # 원뿔대
            params['bottom_r'] = st.slider("밑면 반지름", 0.5, 4.0, 2.0)
            params['top_r'] = st.slider("윗면 반지름", 0.5, 4.0, 1.0)

    st.write("---")
    st.header("2. 뷰 설정 (회전은 여기서!)")
    rot_x = st.slider("X축 회전 (↕)", 0, 360, 20)
    rot_y = st.slider("Y축 회전 (↔)", 0, 360, 30)
    rot_z = st.slider("Z축 회전 (🔄)", 0, 360, 0)
    
    cam_dist = st.slider("카메라 거리", 3.0, 15.0, 6.0)
    is_perspective = st.checkbox("원근 투영 (Perspective)", value=True)


# --- 2. 도형 데이터 생성 ---
def create_geometry(cat, **p):
    verts = []
    faces = []
    
    # [A] 직접 구성 (기둥, 뿔, 뿔대, 회전체)
    if cat in ["기둥/뿔/뿔대", "회전체"]:
        n = int(p['n'])
        h = p['h']
        tr = p['top_r']
        br = p['bottom_r']
        
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        # 윗면 점 (0 ~ n-1)
        for t in theta: verts.append([tr * np.cos(t), tr * np.sin(t), h/2])
        # 아랫면 점 (n ~ 2n-1)
        for t in theta: verts.append([br * np.cos(t), br * np.sin(t), -h/2])
        
        verts = np.array(verts)
        
        # 면 구성
        faces.append(list(range(n))) # 윗면
        faces.append(list(range(2*n-1, n-1, -1))) # 아랫면 (역순)
        
        for i in range(n):
            t1 = i
            t2 = (i + 1) % n
            b1 = i + n
            b2 = ((i + 1) % n) + n
            faces.append([t1, b1, b2, t2]) # 옆면
            
        return verts, faces, None

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
        return verts, hull.simplices, hull.equations

    return np.array([]), [], None


# --- 3. 메인 연산 ---
verts, faces, hull_eqs = create_geometry(category, **params)

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

# --- 4. 가시성 판별 (Visible Face Detection) ---
camera_pos = np.array([0, 0, cam_dist])
visible_faces_idx = set()

for i, face in enumerate(faces):
    face_pts = rotated_verts[face]
    center = np.mean(face_pts, axis=0)
    
    if is_perspective:
        view_vec = camera_pos - center # 표면 -> 카메라 벡터
    else:
        view_vec = np.array([0, 0, 1])
    
    # 법선 벡터 계산
    normal = np.array([0.0, 0.0, 0.0])
    if hull_eqs is not None:
        original_normal = hull_eqs[i][:3]
        normal = original_normal @ rot_mat.T
    else:
        # 다각형의 처음 세 점을 이용해 법선 계산 (반시계 방향 가정)
        v1 = face_pts[1] - face_pts[0]
        v2 = face_pts[2] - face_pts[0]
        normal = np.cross(v1, v2)
        
    # 벡터 내적: 0보다 크면 카메라를 향하고 있음 (보임)
    if np.dot(normal, view_vec) > 1e-3: 
        visible_faces_idx.add(i)

# --- 5. 모서리 분류 (감옥 창살 제거 & 점선 계산) ---
edge_map = {} 

for f_idx, face in enumerate(faces):
    n_pts = len(face)
    for i in range(n_pts):
        p1, p2 = face[i], face[(i+1)%n_pts]
        # 모서리는 (작은인덱스, 큰인덱스) 키로 저장
        key = tuple(sorted((p1, p2)))
        if key not in edge_map:
            edge_map[key] = []
        edge_map[key].append(f_idx)

vis_edges = []
hid_edges = []

current_n = int(params.get('n', 0))

for (p1, p2), f_indices in edge_map.items():
    is_visible = False
    
    # [핵심] 원기둥/원뿔 세로선 처리 ('감옥 창살' 제거)
    is_vertical_edge = False
    if category == "회전체":
        # 인덱스 차이가 n이면 세로선
        if abs(p1 - p2) == current_n:
            is_vertical_edge = True
            
    if category == "회전체" and is_vertical_edge:
        # 회전체 세로선은 '실루엣(외곽선)'일 때만 그림
        # 인접한 면 중 하나는 보이고, 하나는 안 보일 때만 그림
        vis_count = sum(1 for f in f_indices if f in visible_faces_idx)
        if vis_count == 1: 
            is_visible = True
        else:
            # 다 보이거나 다 안 보이면 그림을 안 그림 (continue) -> 깔끔해짐
            continue 
            
    else:
        # 일반 도형 (각기둥 등): 인접 면 중 하나라도 보이면 실선
        for f_idx in f_indices:
            if f_idx in visible_faces_idx:
                is_visible = True
                break
            
    # 좌표 추출
    pts = rotated_verts[[p1, p2]]
    line_seg = [pts[0], pts[1]]
    
    if is_visible:
        vis_edges.append(line_seg)
    else:
        # 회전체 내부 점선은 지저분하므로 생략, 일반 도형은 점선 추가
        if not (category == "회전체" and is_vertical_edge):
            hid_edges.append(line_seg)

# --- 6. 그리기 데이터 변환 ---
# [수정됨] 기존 코드의 버그(y 중복 추가)를 완벽히 해결한 함수
def flatten_lines(seg_list):
    x, y, z = [], [], []
    for s in seg_list:
        x.extend([s[0][0], s[1][0], None])
        y.extend([s[0][1], s[1][1], None])
        z.extend([s[0][2], s[1][2], None])
    return x, y, z

fig = go.Figure()

# 1. 뒷면 (점선)
hx, hy, hz = flatten_lines(hid_edges)
fig.add_trace(go.Scatter3d(
    x=hx, y=hy, z=hz, mode='lines',
    line=dict(color='gray', width=3, dash='dash'), # 점선
    hoverinfo='none', name='점선(뒤)'
))

# 2. 앞면 (실선)
vx, vy, vz = flatten_lines(vis_edges)
fig.add_trace(go.Scatter3d(
    x=vx, y=vy, z=vz, mode='lines',
    line=dict(color='black', width=5), # 굵은 실선
    hoverinfo='none', name='실선(앞)'
))

# 3. 면 칠하기 (ConvexHull 사용)
opacity_val = 0.2 # 내부 점선이 잘 보이도록 투명도 조정
try:
    hull = ConvexHull(rotated_verts)
    fig.add_trace(go.Mesh3d(
        x=rotated_verts[:,0], y=rotated_verts[:,1], z=rotated_verts[:,2],
        i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2],
        color='#d0f0fd', opacity=opacity_val, 
        flatshading=(category != "회전체"), 
        hoverinfo='none', name='면', lighting=dict(ambient=0.8)
    ))
except:
    pass

# 카메라 및 레이아웃 설정
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        camera=dict(
            projection=dict(type="perspective" if is_perspective else "orthographic"),
            eye=dict(x=0, y=0, z=cam_dist*0.2), # 초기 시점
            up=dict(x=0, y=1, z=0)
        ),
        aspectmode='data', # [핵심] 찌그러짐 방지
        dragmode=False # [핵심] 마우스 회전 금지 (점선 틀어짐 방지)
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=650,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
