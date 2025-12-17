import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="완벽한 도형 생성기", layout="wide")
st.title("📐 수학 도형 생성기 (최종_디버깅)")
st.caption("✅ 정다면체 대각선 삭제 + 실선/점선 판정 완벽 수정")

# 스타일 설정
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.warning("⚠️ **마우스 회전 금지**: 정확한 점선/실선 계산을 위해 **왼쪽 슬라이더**만 사용해주세요.")

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
    st.header("2. 뷰 설정")
    rot_x = st.slider("X축 회전 (↕)", 0, 360, 20)
    rot_y = st.slider("Y축 회전 (↔)", 0, 360, 30)
    rot_z = st.slider("Z축 회전 (🔄)", 0, 360, 0)
    cam_dist = st.slider("카메라 거리", 3.0, 15.0, 6.0)
    is_perspective = st.checkbox("원근 투영 (Perspective)", value=True)


# --- 2. 도형 데이터 생성 ---
def create_geometry(cat, **p):
    verts = []
    
    # [A] 직접 구성 (기둥, 뿔, 뿔대, 회전체)
    if cat in ["기둥/뿔/뿔대", "회전체"]:
        n = int(p['n'])
        h = p['h']
        tr = p['top_r']
        br = p['bottom_r']
        
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        for t in theta: verts.append([tr * np.cos(t), tr * np.sin(t), h/2])
        for t in theta: verts.append([br * np.cos(t), br * np.sin(t), -h/2])
        
        verts = np.array(verts)
        # ConvexHull을 사용하여 면을 자동으로 구성 (가장 확실한 방법)
        hull = ConvexHull(verts)
        return verts, hull.simplices

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

# --- 3. 메인 연산 및 회전 ---
verts, simplices = create_geometry(category, **params)

def get_rotation_matrix(x, y, z):
    rad = np.radians([x, y, z])
    c, s = np.cos(rad), np.sin(rad)
    Rx = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
    Ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    Rz = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

rot_mat = get_rotation_matrix(rot_x, rot_y, rot_z)
rotated_verts = verts @ rot_mat.T 

# --- 4. 면의 법선 벡터 및 가시성 계산 ---
# 각 삼각형 면(simplex)의 법선 벡터를 구합니다.
face_normals = []
face_visible = []
camera_pos = np.array([0, 0, cam_dist])

for face in simplices:
    # 회전된 좌표 기준으로 법선 계산
    pts = rotated_verts[face]
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    norm = np.cross(v1, v2)
    norm = norm / (np.linalg.norm(norm) + 1e-9) # 정규화
    face_normals.append(norm)
    
    # 가시성 판별
    center = np.mean(pts, axis=0)
    if is_perspective:
        view_vec = camera_pos - center
        view_vec = view_vec / (np.linalg.norm(view_vec) + 1e-9)
    else:
        view_vec = np.array([0, 0, 1])
        
    # 카메라를 향하면(내적 > 0) 보이는 면
    face_visible.append(np.dot(norm, view_vec) > 1e-4)

# --- 5. 모서리 분류 (대각선 삭제 및 실선/점선 구분 로직 개선) ---
edge_map = {} # (p1, p2) -> [face_idx1, face_idx2, ...]

# 모든 삼각형 면의 변을 수집
for f_idx, face in enumerate(simplices):
    n_pts = len(face)
    for i in range(n_pts):
        p1, p2 = face[i], face[(i+1)%n_pts]
        key = tuple(sorted((p1, p2)))
        if key not in edge_map:
            edge_map[key] = []
        edge_map[key].append(f_idx)

vis_edges = []
hid_edges = []
current_n = int(params.get('n', 0))

for (p1, p2), f_indices in edge_map.items():
    # 1. 인접한 면이 2개 미만이면(열린 도형 등) 일단 그림 (예외처리)
    if len(f_indices) < 2:
        continue
        
    f1_idx = f_indices[0]
    f2_idx = f_indices[1]
    
    # 2. [대각선 삭제] 두 면의 법선 벡터가 거의 같으면(평행하면) 그 사이 선은 '내부 대각선'이므로 삭제
    # 내적값이 1에 가까우면 같은 방향을 보는 면임
    normal_dot = np.dot(face_normals[f1_idx], face_normals[f2_idx])
    if normal_dot > 0.999: 
        # 단, 회전체(원기둥 옆면)는 부드럽게 보여야 하므로 제외하지 않음? 
        # 아니요, 회전체도 옆면 띠를 없애려면 이 로직이 필요합니다.
        # 하지만 원기둥의 '모서리'는 그려야 하므로, 카테고리에 따라 다름.
        
        # 정다면체나 각기둥의 평평한 면 위의 대각선은 확실히 삭제
        if category != "회전체":
            continue
        # 회전체일 경우, 세로선(감옥창살) 제거 로직을 따름
    
    # 3. [회전체 창살 제거]
    is_vertical_edge = False
    if category == "회전체":
        # 인덱스 차이가 n이면 세로선
        if abs(p1 - p2) == current_n:
            is_vertical_edge = True
            
        if is_vertical_edge:
            # 외곽선(실루엣)만 그림: 하나는 보이고 하나는 안 보일 때
            vis1 = face_visible[f1_idx]
            vis2 = face_visible[f2_idx]
            if vis1 != vis2: # XOR
                vis_edges.append([rotated_verts[p1], rotated_verts[p2]])
            continue # 나머지 세로선은 생략
    
    # 4. [실선/점선 판정]
    # 인접한 두 면 중 하나라도 보이면 -> 실선 (겉에 드러난 모서리)
    # 인접한 두 면이 모두 안 보이면 -> 점선 (뒤쪽에 숨은 모서리)
    is_vis_f1 = face_visible[f1_idx]
    is_vis_f2 = face_visible[f2_idx]
    
    line_seg = [rotated_verts[p1], rotated_verts[p2]]
    
    if is_vis_f1 or is_vis_f2:
        vis_edges.append(line_seg)
    else:
        # 회전체가 아닐 때만 점선 그림 (회전체 내부는 지저분하므로)
        if category != "회전체":
            hid_edges.append(line_seg)


# --- 6. 그리기 ---
def flatten(seg_list):
    x, y, z = [], [], []
    for s in seg_list:
        x.extend([s[0][0], s[1][0], None])
        y.extend([s[0][1], s[1][1], None])
        z.extend([s[0][2], s[1][2], None])
    return x, y, z

fig = go.Figure()

# 1. 점선 (Hidden)
hx, hy, hz = flatten(hid_edges)
fig.add_trace(go.Scatter3d(
    x=hx, y=hy, z=hz, mode='lines',
    line=dict(color='gray', width=3, dash='dash'),
    hoverinfo='none', name='점선'
))

# 2. 실선 (Visible)
vx, vy, vz = flatten(vis_edges)
fig.add_trace(go.Scatter3d(
    x=vx, y=vy, z=vz, mode='lines',
    line=dict(color='black', width=5),
    hoverinfo='none', name='실선'
))

# 3. 면 (투명도 조절)
# 회전체는 약간 더 불투명하게, 다면체는 내부 점선 보이게 투명하게
opacity = 0.5 if category == "회전체" else 0.15

fig.add_trace(go.Mesh3d(
    x=rotated_verts[:,0], y=rotated_verts[:,1], z=rotated_verts[:,2],
    i=simplices[:,0], j=simplices[:,1], k=simplices[:,2],
    color='#d0f0fd', opacity=opacity,
    flatshading=True, lighting=dict(ambient=0.8, diffuse=0.1),
    hoverinfo='none', name='면'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        camera=dict(
            projection=dict(type="perspective" if is_perspective else "orthographic"),
            eye=dict(x=0, y=0, z=cam_dist*0.2),
            up=dict(x=0, y=1, z=0)
        ),
        aspectmode='data',
        dragmode=False # 마우스 회전 금지
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=600,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
