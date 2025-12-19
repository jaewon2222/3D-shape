import streamlit as st

import plotly.graph_objects as go

import numpy as np

from scipy.spatial import ConvexHull



# --- 페이지 설정 ---

st.set_page_config(page_title="수학 문제집 생성기", layout="wide")

st.title("📐 수학 문제집 도형 생성기 (실루엣 알고리즘)")

st.markdown("""

**[긴급 수정]** * **바코드 현상 완전 제거:** 원기둥/원뿔의 옆면 내부 선을 수학적으로 0으로 만듭니다.

* **실루엣 알고리즘:** '보이는 면'과 '안 보이는 면'이 만나는 경계선만 그립니다.

""")



# --- 1. 사이드바 설정 ---

st.sidebar.header("1. 보기 설정")

projection_mode = st.sidebar.radio(

    "투영 방식", 

    ["교과서 모드 (직교 투영)", "현실 모드 (원근 투영)"],

    index=0

)



st.sidebar.header("2. 도형 선택")

category = st.sidebar.radio("카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔 (매끈함)", "정다면체"])



st.sidebar.header("3. 도형 회전")

rot_x = st.sidebar.slider("X축 회전", 0, 360, 20)

rot_y = st.sidebar.slider("Y축 회전", 0, 360, 30)

rot_z = st.sidebar.slider("Z축 회전", 0, 360, 0)



# --- 2. 수학 함수 ---

def rotate_points(points, rx, ry, rz):

    rad_x, rad_y, rad_z = np.radians(rx), np.radians(ry), np.radians(rz)

    mat_x = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])

    mat_y = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])

    mat_z = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])

    return points @ mat_x.T @ mat_y.T @ mat_z.T



# --- 3. 도형 데이터 생성 ---

points = []

# "매끈함" 옵션이 켜져 있으면, 인접한 면의 각도가 낮을 때 선을 아예 안 그립니다.

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



elif category == "원기둥/원뿔 (매끈함)":

    is_curved_surface = True

    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])

    n = 80 # 곡면을 표현하기 위해 점을 많이 찍음

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

rotated_points = rotate_points(points, rot_x, rot_y, rot_z)

hull = ConvexHull(rotated_points)



# 각 면의 법선 벡터 계산

normals = []

for eq in hull.equations:

    n_vec = eq[:3]

    normals.append(n_vec / np.linalg.norm(n_vec))

normals = np.array(normals)



# 카메라 설정 (직교/원근)

camera_pos = np.array([0, 0, 10.0])

visible_faces_mask = []



for i, simplex in enumerate(hull.simplices):

    if "교과서 모드" in projection_mode:

        # 직교 투영: 법선의 Z값이 양수면 보임

        is_visible = normals[i][2] > 0

    else:

        # 원근 투영: 면의 중심에서 카메라를 향한 벡터와 법선 내적

        face_center = np.mean(rotated_points[simplex], axis=0)

        view_vector = face_center - camera_pos 

        is_visible = np.dot(view_vector, normals[i]) < 0

    visible_faces_mask.append(is_visible)



# 엣지 정보 수집 (어떤 면들이 공유하는지)

edge_to_faces = {}

for face_idx, simplex in enumerate(hull.simplices):

    n_pts = len(simplex)

    for k in range(n_pts):

        p1, p2 = sorted((simplex[k], simplex[(k+1)%n_pts]))

        edge = (p1, p2)

        if edge not in edge_to_faces: edge_to_faces[edge] = []

        edge_to_faces[edge].append(face_idx)



# --- 5. 선 그리기 로직 (핵심 수정) ---

visible_edges = set()

hidden_edges = set()



for edge, faces in edge_to_faces.items():

    if len(faces) == 2:

        f1, f2 = faces

        n1, n2 = normals[f1], normals[f2]

        v1, v2 = visible_faces_mask[f1], visible_faces_mask[f2]

        

        # 두 면 사이의 각도 계산 (내적)

        # 값이 1에 가까울수록 평평하게 이어진 면 (곡면의 일부)

        dot_val = np.dot(n1, n2)

        is_smooth_edge = dot_val > 0.8  # 각도가 완만하면 '부드러운 모서리'로 간주



        if is_curved_surface and is_smooth_edge:

            # [원기둥/원뿔 해결책]

            # 부드러운 곡면에서는 '실루엣'만 그린다.

            # 실루엣의 정의: 한 면은 보이고, 다른 면은 안 보일 때 (v1 != v2)

            if v1 != v2:

                visible_edges.add(edge)

            # 둘 다 보이거나(배 부분), 둘 다 안 보이면(등 부분) -> 절대 그리지 않음!

        else:

            # [각기둥/각뿔 해결책]

            # 각진 모서리는 평범하게 처리

            # 하지만 평면 위의 대각선(완벽히 평평함, dot_val > 0.999)은 지움

            if dot_val > 0.999:

                continue

            

            if v1 or v2:

                visible_edges.add(edge)

            else:

                hidden_edges.add(edge)



    else:

        # 경계면 처리 (드물지만 안전장치)

        if any(visible_faces_mask[f] for f in faces): visible_edges.add(edge)

        else: hidden_edges.add(edge)



visible_mesh_indices = []

for i, is_vis in enumerate(visible_faces_mask):

    if is_vis: visible_mesh_indices.append(hull.simplices[i])



# --- 6. 시각화 ---

fig = go.Figure()



# 숨은 선 (점선)

x_dash, y_dash, z_dash = [], [], []

for p1, p2 in hidden_edges:

    pts = rotated_points[[p1, p2]]

    x_dash.extend([pts[0][0], pts[1][0], None])

    y_dash.extend([pts[0][1], pts[1][1], None])

    z_dash.extend([pts[0][2], pts[1][2], None])



fig.add_trace(go.Scatter3d(

    x=x_dash, y=y_dash, z=z_dash, mode='lines',

    line=dict(color='silver', width=3, dash='dash'),

    name='숨은 선', hoverinfo='none'

))



# 보이는 선 (실선)

x_solid, y_solid, z_solid = [], [], []

for p1, p2 in visible_edges:

    pts = rotated_points[[p1, p2]]

    x_solid.extend([pts[0][0], pts[1][0], None])

    y_solid.extend([pts[0][1], pts[1][1], None])

    z_solid.extend([pts[0][2], pts[1][2], None])



fig.add_trace(go.Scatter3d(

    x=x_solid, y=y_solid, z=z_solid, mode='lines',

    line=dict(color='black', width=5),

    name='보이는 선', hoverinfo='none'

))



# 면 채우기 (흰색)

if visible_mesh_indices:

    visible_mesh_indices = np.array(visible_mesh_indices)

    fig.add_trace(go.Mesh3d(

        x=rotated_points[:,0], y=rotated_points[:,1], z=rotated_points[:,2],

        i=visible_mesh_indices[:,0], j=visible_mesh_indices[:,1], k=visible_mesh_indices[:,2],

        color='white', opacity=0.15,

        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), # 그림자 제거

        hoverinfo='none', name='면'

    ))



# 뷰 설정

if "교과서 모드" in projection_mode:

    proj_type = "orthographic"

    cam_dist = 2.0

else:

    proj_type = "perspective"

    cam_dist = 2.5



fig.update_layout(

    scene=dict(

        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),

        bgcolor='white',

        aspectmode='data',

        camera=dict(

            projection=dict(type=proj_type), 

            eye=dict(x=0, y=0, z=cam_dist),

            up=dict(x=0, y=1, z=0)

        )

    ),

    margin=dict(l=0, r=0, b=0, t=0), height=600, dragmode=False,

    paper_bgcolor='white'

)



st.plotly_chart(fig, use_container_width=True)
