import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

st.set_page_config(page_title="진짜 겨냥도 생성기", layout="wide")
st.title("📐 3D 입체도형 관측소 (Real-time 겨냥도)")
st.markdown("""
**마우스 드래그**는 단순히 카메라 위치만 바꿉니다.
**반드시 왼쪽 사이드바의 [도형 회전] 슬라이더를 움직이세요.** 그래야 실선/점선이 수학적으로 다시 계산됩니다.
""")

# --- 사이드바 설정 ---
st.sidebar.header("1. 도형 선택")
category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (근사)", "정다면체"])

st.sidebar.header("2. 도형 회전 (필수)")
rot_x = st.sidebar.slider("X축 회전 (위아래)", 0, 360, 30)
rot_y = st.sidebar.slider("Y축 회전 (좌우)", 0, 360, 45)
# Z축은 겨냥도에서 큰 의미 없으므로 생략하거나 필요시 추가

# --- 수학 함수: 회전 행렬 ---
def rotate_points(points, rx, ry):
    # 라디안 변환
    rad_x = np.radians(rx)
    rad_y = np.radians(ry)
    
    # 회전 행렬 정의
    mat_x = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    mat_y = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    # Y축 회전 후 X축 회전 적용
    rotated = points @ mat_y.T
    rotated = rotated @ mat_x.T
    return rotated

# --- 포인트 생성 로직 ---
points = []

if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = 4.0
    rb = 2.0
    
    if sub_type == "각기둥": rt = rb
    elif sub_type == "각뿔": rt = 0.001 # 0이면 ConvexHull 계산시 에러 가능성 있어 아주 작은 값
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, 1.9, 1.0)
    
    # 점 생성
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    # 윗면
    for t in theta: points.append([rt*np.cos(t), rt*np.sin(t), h/2])
    # 아랫면
    for t in theta: points.append([rb*np.cos(t), rb*np.sin(t), -h/2])
    
elif category == "원기둥/원뿔 (근사)":
    # 원기둥도 다각형으로 근사하여 처리 (n=40 정도면 충분히 원 같음)
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    n = 40 
    h = 4.0
    rb = 2.0
    if sub_type == "원기둥": rt = rb
    elif sub_type == "원뿔": rt = 0.001
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, 1.9, 1.0)
    
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    for t in theta: points.append([rt*np.cos(t), rt*np.sin(t), h/2])
    for t in theta: points.append([rb*np.cos(t), rb*np.sin(t), -h/2])

elif category == "정다면체":
    sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
    phi = (1 + np.sqrt(5)) / 2
    pts = []
    if sub_type == "정사면체": pts = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    elif sub_type == "정육면체":
        for x in [-1,1]:
            for y in [-1,1]:
                for z in [-1,1]: pts.append([x,y,z])
    elif sub_type == "정팔면체": pts = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif sub_type == "정십이면체":
        for x in [-1,1]:
             for y in [-1,1]:
                 for z in [-1,1]: pts.append([x,y,z])
        for i in [-1,1]:
             for j in [-1,1]: pts.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif sub_type == "정이십면체":
        for i in [-1,1]:
            for j in [-1,1]: pts.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])
    points = pts

# --- 핵심 로직: ConvexHull & 가시성 판단 ---
points = np.array(points)
# 1. 사용자 입력대로 회전시킴
rotated_points = rotate_points(points, rot_x, rot_y)

# 2. ConvexHull 계산 (면과 이웃 정보 추출)
hull = ConvexHull(rotated_points)

# 3. 각 면(Simplex)의 법선 벡터 확인
# ConvexHull의 equations는 [nx, ny, nz, offset] 형태이며, 법선은 바깥쪽을 향함
normals = hull.equations[:, :3]

# 4. 카메라 시점 설정 (우리는 물체를 회전시켰으므로 카메라는 고정된 위치라고 가정)
# Plotly의 기본 뷰는 +Z 쪽에서 바라보는 것과 유사하지만, 
# 여기서는 직관성을 위해 "화면을 뚫고 나오는 방향(+Z)"을 시선으로 가정합니다.
# 면의 법선 z값이 > 0 이면 카메라를 향하는 것 (보임), < 0 이면 뒤로 숨은 것.
visible_faces = []
for i, normal in enumerate(normals):
    # 카메라가 (0,0,infinity)에 있다고 가정하고 Orthographic projection 관점
    # 법선의 z성분이 양수면 관측자를 향함
    is_visible = normal[2] > 0 
    visible_faces.append(is_visible)

# 5. 엣지(모서리) 분류
visible_edges = set()
hidden_edges = set()

# hull.simplices는 각 면을 이루는 점들의 인덱스
# 모든 면을 순회하며 엣지 정보를 수집
for simplex_idx, simplex in enumerate(hull.simplices):
    # simplex는 삼각형을 이루는 3개의 점 인덱스 (예: [0, 4, 2])
    # 이 면이 보이는지 확인
    is_face_visible = visible_faces[simplex_idx]
    
    # 면의 각 변(edge)에 대해
    num_points = len(simplex)
    for i in range(num_points):
        p1, p2 = simplex[i], simplex[(i+1)%num_points]
        edge = tuple(sorted((p1, p2))) # (작은수, 큰수) 형태로 통일
        
        # 로직:
        # 이 엣지는 두 면이 공유합니다.
        # 하나라도 보이는 면에 속하면 -> 실선 (Visible)
        # 만약 이 엣지가 이미 Visible로 등록되어 있다면 건드리지 않음
        # 만약 Hidden으로 등록되어 있는데 지금 보니 Visible 면에 속하면 -> Visible로 승격
        
        if is_face_visible:
            if edge in hidden_edges:
                hidden_edges.remove(edge)
            visible_edges.add(edge)
        else:
            # 안 보이는 면에 속함. 
            # 단, 이미 Visible 리스트에 있다면(다른 보이는 면과 공유중이라면) Hidden으로 넣지 않음
            if edge not in visible_edges:
                hidden_edges.add(edge)

# --- 시각화 (Plotly) ---
fig = go.Figure()

# 1. 점선 그리기 (Hidden Edges)
x_dash, y_dash, z_dash = [], [], []
for p1, p2 in hidden_edges:
    pts = rotated_points[[p1, p2]]
    x_dash.extend([pts[0][0], pts[1][0], None])
    y_dash.extend([pts[0][1], pts[1][1], None])
    z_dash.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_dash, y=y_dash, z=z_dash,
    mode='lines',
    line=dict(color='gray', width=4, dash='dash'), # 회색 점선
    name='보이지 않는 선',
    hoverinfo='none'
))

# 2. 실선 그리기 (Visible Edges)
x_solid, y_solid, z_solid = [], [], []
for p1, p2 in visible_edges:
    pts = rotated_points[[p1, p2]]
    x_solid.extend([pts[0][0], pts[1][0], None])
    y_solid.extend([pts[0][1], pts[1][1], None])
    z_solid.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_solid, y=y_solid, z=z_solid,
    mode='lines',
    line=dict(color='black', width=6), # 검은 실선
    name='보이는 선',
    hoverinfo='none'
))

# 3. 면 그리기 (옵션: 면을 아주 연하게 깔아서 입체감 보조)
# 면을 그릴 때는 ConvexHull의 simplices를 그대로 사용
fig.add_trace(go.Mesh3d(
    x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],
    i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2],
    color='lightblue', opacity=0.1, # 아주 투명하게
    lighting=dict(ambient=0.8),
    hoverinfo='none',
    name='면'
))

# 레이아웃 설정
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        camera=dict(
            eye=dict(x=0, y=0, z=2.0), # 카메라를 정면(Z축 위)에 고정
            up=dict(x=0, y=1, z=0)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=600,
    dragmode=False # 마우스 드래그를 막는 것이 오해를 줄임 (선택사항)
)

st.plotly_chart(fig, use_container_width=True)
