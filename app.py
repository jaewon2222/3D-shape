import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="도형 생성기", layout="wide")
st.title("📐 수학 문제집 도형 생성기 (최종 수정)")

# --- 1. 설정 (사이드바) ---
st.sidebar.header("1. 보기 모드")
projection = st.sidebar.radio("투영", ["교과서(직교)", "눈으로 보는 것(원근)"], index=0)

st.sidebar.header("2. 도형 선택")
category = st.sidebar.radio("종류", ["각기둥/각뿔", "원기둥/원뿔", "정다면체"])

# --- 2. 도형 데이터 생성 ---
points = []
is_curved = False # 원기둥/원뿔인지 확인하는 태그

if category == "각기둥/각뿔":
    type_ = st.sidebar.selectbox("상세", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n각형 (밑면)", 3, 20, 3) # 기본값 3(삼각뿔)
    h, r = 4.0, 2.0
    if type_ == "각기둥": rt = r
    elif type_ == "각뿔": rt = 0.001
    else: rt = 1.0
    
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    for t in th: points.append([rt*np.cos(t), rt*np.sin(t), h/2]) # 윗면
    for t in th: points.append([r*np.cos(t), r*np.sin(t), -h/2])  # 아랫면

elif category == "원기둥/원뿔":
    is_curved = True # 곡면 모드 켜기 (세로줄 삭제용)
    type_ = st.sidebar.selectbox("상세", ["원기둥", "원뿔", "원뿔대"])
    n = 60 # 곡면을 부드럽게 표현하기 위해 점을 많이 찍음
    h, r = 4.0, 2.0
    if type_ == "원기둥": rt = r
    elif type_ == "원뿔": rt = 0.001
    else: rt = 1.0
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    for t in th: points.append([rt*np.cos(t), rt*np.sin(t), h/2])
    for t in th: points.append([r*np.cos(t), r*np.sin(t), -h/2])

elif category == "정다면체":
    type_ = st.sidebar.selectbox("상세", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
    phi = (1 + 5**0.5) / 2
    if type_ == "정사면체": points = [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
    elif type_ == "정육면체": points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
    elif type_ == "정팔면체": points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif type_ == "정십이면체":
        points = [[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        for i in [-1,1]:
             for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif type_ == "정이십면체":
        for i in [-1,1]:
            for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

points = np.array(points)

# --- 3. 회전 및 계산 ---
st.sidebar.markdown("---")
st.sidebar.header("3. 회전")
rx = st.sidebar.slider("X축", 0, 360, 20)
ry = st.sidebar.slider("Y축", 0, 360, 30)
rz = st.sidebar.slider("Z축", 0, 360, 0)

def rotate(p, x, y, z):
    ax, ay, az = np.radians(x), np.radians(y), np.radians(z)
    mx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    my = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    mz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    return p @ mx.T @ my.T @ mz.T

r_points = rotate(points, rx, ry, rz)
hull = ConvexHull(r_points)

# --- 4. 보이는 선 / 숨은 선 계산 ---
normals = []
for eq in hull.equations:
    normals.append(eq[:3] / np.linalg.norm(eq[:3]))

camera = np.array([0,0,10])
vis_mask = []
for i, s in enumerate(hull.simplices):
    if projection.startswith("교과서"):
        vis_mask.append(normals[i][2] > 0)
    else:
        center = np.mean(r_points[s], axis=0)
        vis_mask.append(np.dot(center - camera, normals[i]) < 0)

edges = {}
for i, s in enumerate(hull.simplices):
    for k in range(len(s)):
        edge = tuple(sorted((s[k], s[(k+1)%len(s)])))
        if edge not in edges: edges[edge] = []
        edges[edge].append(i)

vis_edges, hid_edges = [], []
for e, fs in edges.items():
    if len(fs) == 2:
        v1, v2 = vis_mask[fs[0]], vis_mask[fs[1]]
        n1, n2 = normals[fs[0]], normals[fs[1]]
        
        # [중요 1] 평면 위의 대각선 삭제 (사각형 면의 빗금 제거)
        if np.dot(n1, n2) > 0.999: continue
        
        # [중요 2] 원기둥 옆면 세로줄(바코드) 삭제 로직
        if is_curved and np.dot(n1, n2) > 0.8:
             # 곡면에서는 오직 '경계선(실루엣)'만 그린다
             if v1 != v2: vis_edges.append(e)
        
        # [중요 3] 일반적인 모서리 (각기둥, 각뿔 등)
        else:
            if v1 or v2: vis_edges.append(e)
            else: hid_edges.append(e)

# --- 5. 그리기 (여기가 수정된 핵심) ---
fig = go.Figure()

# 숨은 선 (Hidden Lines): 검은색 + 굵은 점선
x_h, y_h, z_h = [], [], []
for p1, p2 in hid_edges:
    pts = r_points[[p1, p2]]
    x_h.extend([pts[0][0], pts[1][0], None])
    y_h.extend([pts[0][1], pts[1][1], None])
    z_h.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_h, y=y_h, z=z_h, mode='lines',
    line=dict(color='black', width=4, dash='dash'), # 여기를 수정했습니다 (검정, 굵게)
    name='숨은 선', hoverinfo='none'
))

# 보이는 선 (Visible Lines): 검은색 + 굵은 실선
x_v, y_v, z_v = [], [], []
for p1, p2 in vis_edges:
    pts = r_points[[p1, p2]]
    x_v.extend([pts[0][0], pts[1][0], None])
    y_v.extend([pts[0][1], pts[1][1], None])
    z_v.extend([pts[0][2], pts[1][2], None])

fig.add_trace(go.Scatter3d(
    x=x_v, y=y_v, z=z_v, mode='lines',
    line=dict(color='black', width=5),
    name='보이는 선', hoverinfo='none'
))

# 카메라 뷰 설정
cam_dist = 2.0 if projection.startswith("교과서") else 2.5
proj_type = "orthographic" if projection.startswith("교과서") else "perspective"

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        bgcolor='white', aspectmode='data',
        camera=dict(projection=dict(type=proj_type), eye=dict(x=0, y=0, z=cam_dist), up=dict(x=0, y=1, z=0))
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=600, paper_bgcolor='white', showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
