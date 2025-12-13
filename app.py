import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 관측기", layout="wide")

# 스타일링: 여백 줄이기
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

st.title("📐 왜곡 없는 3D 입체도형 관측소")
st.markdown("도형이 잘리거나 찌그러지지 않도록 **1:1:1 비율**과 **넓은 시야**를 적용했습니다.")

# --- 사이드바: 메뉴 선택 ---
st.sidebar.header("설정")
category = st.sidebar.radio(
    "도형 카테고리",
    ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체 (Platonic)", "구 (Sphere)"]
)

# --- 공통 함수: 다각형/원형 기둥, 뿔, 대 생성 ---
def create_general_mesh(n, r_bottom, r_top, height, color='cyan', name='Shape'):
    theta = np.linspace(0, 2*np.pi, n+1)
    
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta)

    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height)

    x = np.concatenate([x_top, x_bottom, [0], [0]])
    y = np.concatenate([y_top, y_bottom, [0], [0]])
    z = np.concatenate([z_top, z_bottom, [height], [0]])
    
    i_list, j_list, k_list = [], [], []

    # 옆면
    for idx in range(n):
        i_list.extend([idx, idx])
        j_list.extend([n + 1 + idx, n + 1 + idx + 1])
        k_list.extend([n + 1 + idx + 1, idx + 1])

    # 윗면
    if r_top > 0:
        center_top = 2 * n + 2
        for idx in range(n):
            i_list.extend([idx, idx + 1, center_top])

    # 아랫면
    if r_bottom > 0:
        center_bottom = 2 * n + 3
        for idx in range(n):
            i_list.extend([n + 1 + idx, center_bottom, n + 1 + idx + 1])

    return go.Mesh3d(x=x, y=y, z=z, i=i_list, j=j_list, k=k_list, opacity=0.8, color=color, flatshading=True, name=name)

# --- 공통 함수: 정다면체 ---
def get_platonic_solid(name, size):
    phi = (1 + np.sqrt(5)) / 2
    points = []
    
    if "정사면체" in name:
        points = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    elif "정육면체" in name:
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]: points.append([x, y, z])
    elif "정팔면체" in name:
        points = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
    elif "정십이면체" in name:
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]: points.append([x, y, z])
        for i in [-1, 1]:
            for j in [-1, 1]:
                points.extend([[0, i*phi, j/phi], [j/phi, 0, i*phi], [i*phi, j/phi, 0]])
    elif "정이십면체" in name:
        for i in [-1, 1]:
            for j in [-1, 1]:
                points.extend([[0, i, j*phi], [j*phi, 0, i], [i, j*phi, 0]])
    
    points = np.array(points) * size
    hull = ConvexHull(points)
    
    color_map = {"정사면체": "magenta", "정육면체": "cyan", "정팔면체": "orange", "정십이면체": "lime", "정이십면체": "violet"}
    color = "gray"
    for key in color_map:
        if key in name: color = color_map[key]

    return go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2], i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2], color=color, opacity=0.8, flatshading=True, name=name)

# --- 메인 로직 ---
fig = go.Figure()

if category == "각기둥/각뿔/각뿔대":
    st.sidebar.subheader("상세 설정")
    sub = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("각 수 (n)", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 15.0, 5.0) # 높이 최대값 15로 증가
    rb = st.sidebar.slider("밑면 반지름", 1.0, 8.0, 3.0)

    if sub == "각기둥": fig.add_trace(create_general_mesh(n, rb, rb, h, 'skyblue', f"{n}각기둥"))
    elif sub == "각뿔": fig.add_trace(create_general_mesh(n, rb, 0, h, 'salmon', f"{n}각뿔"))
    elif sub == "각뿔대":
        rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)
        fig.add_trace(create_general_mesh(n, rb, rt, h, 'lightgreen', f"{n}각뿔대"))

elif category == "원기둥/원뿔/원뿔대":
    st.sidebar.subheader("상세 설정")
    sub = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 15.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 8.0, 3.0)
    res = 60

    if sub == "원기둥": fig.add_trace(create_general_mesh(res, rb, rb, h, 'gold', "원기둥"))
    elif sub == "원뿔": fig.add_trace(create_general_mesh(res, rb, 0, h, 'tomato', "원뿔"))
    elif sub == "원뿔대":
        rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)
        fig.add_trace(create_general_mesh(res, rb, rt, h, 'lime', "원뿔대"))

elif category == "정다면체 (Platonic)":
    sub = st.sidebar.selectbox("도형", ["정사면체 (Tetrahedron)", "정육면체 (Cube)", "정팔면체 (Octahedron)", "정십이면체 (Dodecahedron)", "정이십면체 (Icosahedron)"])
    s = st.sidebar.slider("크기", 1.0, 8.0, 3.0)
    fig.add_trace(get_platonic_solid(sub, s))

elif category == "구 (Sphere)":
    r = st.sidebar.slider("반지름", 1.0, 8.0, 4.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.8))

# --- [핵심 수정] 1:1:1 비율 유지 및 잘림 방지 설정 ---
# 슬라이더 최대값 등을 고려하여 '가상의 방' 크기를 고정합니다.
max_range = 15.0 

fig.update_layout(
    scene=dict(
        # 1. 시야각(Aspect Ratio)을 수동(manual)으로 설정하여 비율 왜곡 방지
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1), # X:Y:Z 비율을 1:1:1로 강제

        # 2. 축의 범위(Range)를 넉넉하고 동일하게 설정하여 잘림 방지
        # X, Y는 중심이 0이므로 -15 ~ 15
        # Z는 바닥이 0이거나 중심이 0일 수 있으므로 넉넉하게 -5 ~ 25로 잡되, 
        # 화면상 1:1을 유지하려면 범위의 '길이(Span)'가 같아야 함.
        # 여기서는 단순화를 위해 모든 축을 -15 ~ 15로 통일하고 Z축만 높이를 고려해 이동시킴.
        xaxis=dict(range=[-max_range, max_range], title='X'),
        yaxis=dict(range=[-max_range, max_range], title='Y'),
        zaxis=dict(range=[-5, 25], title='Z'), # 높이가 15까지 가므로 여유 있게 25까지
    ),
    margin=dict(r=0, l=0, b=0, t=40), # 불필요한 여백 제거
    height=700 # 화면 높이 확보
)

st.plotly_chart(fig, use_container_width=True)
