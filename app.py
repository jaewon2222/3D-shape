import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="진짜 겨냥도 생성기", layout="wide")
st.title("📐 3D 입체도형 관측소 (Real-time 겨냥도)")
st.markdown("""
**[안내]** 마우스 드래그 대신 **왼쪽 사이드바의 '도형 회전' 슬라이더**를 움직이세요.
그래야 컴퓨터가 **어느 선이 뒤에 있는지 수학적으로 계산**하여 점선으로 바꿔줍니다.
""")

# --- 1. 사이드바 설정 ---
st.sidebar.header("1. 도형 선택")
category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (다각형 근사)", "정다면체"])

st.sidebar.header("2. 도형 회전 (필수)")
rot_x = st.sidebar.slider("X축 회전 (앞뒤)", 0, 360, 20)
rot_y = st.sidebar.slider("Y축 회전 (좌우)", 0, 360, 30)
rot_z = st.sidebar.slider("Z축 회전 (풍차)", 0, 360, 0)

# --- 2. 수학 함수: 회전 행렬 ---
def rotate_points(points, rx, ry, rz):
    # 라디안 변환
    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)
    
    # 회전 행렬 정의
    mat_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    mat_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
    mat_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
    
    # 회전 적용 (순서: X -> Y -> Z)
    rotated = points @ mat_x.T
    rotated = rotated @ mat_y.T
    rotated = rotated @ mat_z.T
    return rotated

# --- 3. 도형 데이터 생성 ---
points = []

if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = 4.0; rb = 2.0
    if sub_type == "각기둥": rt = rb
    elif sub_type == "각뿔": rt = 0.001 # 계산 오류 방지를 위해 0 대신 아주 작은 값
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, 1.9, 1.0)
    
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    for t in theta: points.append([rt*np.cos(t), rt*np.sin(t), h/2]) # 윗면
    for t in theta: points.append([rb*np.cos(t), rb*np.sin(t), -h/2]) # 아랫면

elif category == "원기둥/원뿔 (다각형 근사)":
    # 수학적 계산(ConvexHull)을 위해 원을 N각형(30각형)으로 근사합니다.
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    n = 30 # 충분히 원처럼 보임
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
    elif sub_type == "정육면체":
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
    elif sub_type == "정팔면체": points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif sub_type == "정십이면체":
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        for i in [-1,1]:
             for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif sub_type == "정이십면체":
        for i in [-1,1]:
            for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

points = np.array(points)

# --- 4. 핵심 알고리즘: 보이는 선/숨은 선 계산 ---
# 1) 점들을 회전시킵니다.
rotated_points = rotate_points(points, rot_x, rot_y, rot_z)

# 2) ConvexHull로 면(Face) 정보를 구합니다.
hull = ConvexHull(rotated_points)

# 3) 각 면의 법선 벡터(Normal Vector)를 확인합니다.
# Plotly의 카메라는 기본적으로 +Z 방향에서 -Z 방향을 보거나, 사용자가 설정하기 나름입니다.
# 여기서는 편의상 "화면을 뚫고 나오는 방향(+Z)"을 관측자 시점이라고 가정합니다.
# 따라서 법선의 Z값이 양수면(>0) 우리 눈에 보이는 면, 음수면 뒤로 돌아간 면입니다.
normals = hull.equations[:, :3]
visible_faces = [normal[2] > 0 for normal in normals] 

# 4) 모든 모서리(Edge)를 분류합니다.
visible_edges = set()
hidden_edges = set()

for simplex_idx, simplex in enumerate(hull.simplices):
    is_visible = visible_faces[simplex_idx]
    
    # 면을 이루는 각 선분에 대해
    n_pts = len(simplex)
    for i in range(n_pts):
        p1, p2 = simplex[i], simplex[(i+1)%n_pts]
        edge = tuple(sorted((p1, p2))) # (작은인덱스, 큰인덱스)로 통일
        
        # 볼록 다면체의 성질:
        # 두 면이 공유하는 모서리는, "두 면 중 하나라도 보이면" 실선입니다.
        # "두 면이 모두 안 보일 때만" 점선입니다.
        if is_visible:
            # 보이는 면에 속한 모서리는 무조건 실선
            if edge in hidden_edges: hidden_edges.remove(edge) # 혹시 숨김으로 처리됐었다면 취소
            visible_edges.add(edge)
        else:
            # 안 보이는 면에 속함. 단, 이미 실선으로 판명난 녀석은 건드리지 않음
            if edge not in visible_edges:
                hidden_edges.add(edge)

# --- 5. 시각화 (그리기) ---
fig = go.Figure()

# (1) 숨은 선 (점선 그리기)
x_dash, y_dash, z_dash = [], [], []
for p1, p2 in hidden_edges:
    pts = rotated_points[[p1, p2]]
    x_dash.extend([pts[0][0], pts[1][0], None])
    y_dash.extend([pts[0][1], pts[1][1], None])
    z_dash.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_dash, y=y_dash, z=z_dash, mode='lines',
    line=dict(color='gray', width=4, dash='dash'), # 회색 점선
    name='보이지 않는 모서리', hoverinfo='none'
))

# (2) 보이는 선 (실선 그리기)
x_solid, y_solid, z_solid = [], [], []
for p1, p2 in visible_edges:
    pts = rotated_points[[p1, p2]]
    x_solid.extend([pts[0][0], pts[1][0], None])
    y_solid.extend([pts[0][1], pts[1][1], None])
    z_solid.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_solid, y=y_solid, z=z_solid, mode='lines',
    line=dict(color='black', width=6), # 검은 실선
    name='보이는 모서리', hoverinfo='none'
))

# (3) 면 채우기 (입체감을 위해 연하게)
fig.add_trace(go.Mesh3d(
    x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
    i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2],
    color='#dceefc', opacity=0.3, # 아주 연한 하늘색
    lighting=dict(ambient=0.9), # 그림자 최소화
    hoverinfo='none', name='면'
))

# 레이아웃 고정 (카메라는 고정하고 물체를 돌렸으므로)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode='data',
        camera=dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)) # 정면 뷰 고정
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600,
    dragmode=False # 마우스 드래그 방지 (슬라이더 사용 유도)
)

st.plotly_chart(fig, use_container_width=True)
