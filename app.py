import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull  # 정다면체 면 구성을 위해 필요

# --- 페이지 설정 ---
st.set_page_config(page_title="종합 입체도형 관측기", layout="wide")
st.title("📐 종합 입체도형 관측소")
st.markdown("왼쪽 사이드바에서 **도형의 종류**와 **각 수(n)**를 설정해보세요.")

# --- 사이드바: 메뉴 선택 ---
st.sidebar.header("설정")
category = st.sidebar.radio(
    "도형 카테고리 선택",
    ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체 (Platonic Solids)", "구 (Sphere)"]
)

# --- 함수 1: 기둥, 뿔, 뿔대 생성 (다각형 및 원형 공통) ---
def create_general_mesh(n, r_bottom, r_top, height, color='cyan', name='Shape'):
    """
    n: 각형의 수 (원은 60 이상)
    r_bottom: 밑면 반지름
    r_top: 윗면 반지름 (0이면 뿔, r_bottom과 같으면 기둥)
    height: 높이
    """
    # 각도 생성 (0 ~ 2pi)
    theta = np.linspace(0, 2*np.pi, n+1)
    
    # 좌표 계산
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta) # 바닥은 z=0

    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height) # 윗면은 z=height

    # 모든 점 합치기: [윗면 테두리... 아랫면 테두리... 윗면 중심, 아랫면 중심]
    # 인덱스: 0~n(윗면), n+1~2n+1(아랫면), 2n+2(윗면중심), 2n+3(아랫면중심)
    x = np.concatenate([x_top, x_bottom, [0], [0]])
    y = np.concatenate([y_top, y_bottom, [0], [0]])
    z = np.concatenate([z_top, z_bottom, [height], [0]])
    
    # 면(Triangle) 구성 인덱스 리스트
    i_list, j_list, k_list = [], [], []

    # 1. 옆면 구성
    for idx in range(n):
        # 사각형을 삼각형 2개로 쪼개서 옆면을 만듦
        # 삼각형 1
        i_list.append(idx)
        j_list.append(n + 1 + idx)
        k_list.append(n + 1 + idx + 1)
        # 삼각형 2
        i_list.append(idx)
        j_list.append(n + 1 + idx + 1)
        k_list.append(idx + 1)

    # 2. 윗면 뚜껑 (반지름이 0보다 클 때만)
    if r_top > 0:
        center_top_idx = 2 * n + 2
        for idx in range(n):
            i_list.append(idx)
            j_list.append(idx + 1)
            k_list.append(center_top_idx)

    # 3. 아랫면 바닥 (반지름이 0보다 클 때만)
    if r_bottom > 0:
        center_bottom_idx = 2 * n + 3
        for idx in range(n):
            i_list.append(n + 1 + idx)
            j_list.append(center_bottom_idx)
            k_list.append(n + 1 + idx + 1)

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_list, j=j_list, k=k_list,
        opacity=0.7,
        color=color,
        flatshading=True,
        name=name
    )

# --- 함수 2: 정다면체 생성 (ConvexHull 사용) ---
def get_platonic_solid(name, size):
    phi = (1 + np.sqrt(5)) / 2  # 황금비

    points = []
    
    # 1. 정사면체
    if "정사면체" in name:
        points = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

    # 2. 정육면체
    elif "정육면체" in name:
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    points.append([x, y, z])

    # 3. 정팔면체
    elif "정팔면체" in name:
        points = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], 
            [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]

    # 4. 정십이면체 (황금비 이용)
    elif "정십이면체" in name:
        # (±1, ±1, ±1)
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    points.append([x, y, z])
        # (0, ±phi, ±1/phi) 순환
        for i in [-1, 1]:
            for j in [-1, 1]:
                points.append([0, i*phi, j/phi])
                points.append([j/phi, 0, i*phi])
                points.append([i*phi, j/phi, 0])

    # 5. 정이십면체 (황금비 이용)
    elif "정이십면체" in name:
        # (0, ±1, ±phi) 순환
        for i in [-1, 1]:
            for j in [-1, 1]:
                points.append([0, i, j*phi])
                points.append([j*phi, 0, i])
                points.append([i, j*phi, 0])
    
    # --- 점들을 이용해 면 자동 생성 (ConvexHull) ---
    points = np.array(points) * size
    hull = ConvexHull(points) 
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    i, j, k = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]

    # 색상 지정
    color_map = {
        "정사면체": "magenta", "정육면체": "cyan", "정팔면체": "orange",
        "정십이면체": "lime", "정이십면체": "violet"
    }
    # 이름에서 키워드 추출하여 색상 결정
    color = "gray"
    for key in color_map:
        if key in name:
            color = color_map[key]

    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=0.8, flatshading=True, name=name
    )

# --- 메인 로직 ---
fig = go.Figure()

# 1. 각기둥 / 각뿔 / 각뿔대
if category == "각기둥/각뿔/각뿔대":
    st.sidebar.subheader("상세 설정")
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n_sides = st.sidebar.number_input("밑면의 각 수 (n)", min_value=3, max_value=20, value=4, step=1)
    height = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    r_bottom = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)

    if sub_type == "각기둥":
        fig.add_trace(create_general_mesh(n_sides, r_bottom, r_bottom, height, 'skyblue', f"{n_sides}각기둥"))
    elif sub_type == "각뿔":
        fig.add_trace(create_general_mesh(n_sides, r_bottom, 0, height, 'salmon', f"{n_sides}각뿔"))
    elif sub_type == "각뿔대":
        r_top = st.sidebar.slider("윗면 반지름", 0.1, r_bottom-0.1, r_bottom/2)
        fig.add_trace(create_general_mesh(n_sides, r_bottom, r_top, height, 'lightgreen', f"{n_sides}각뿔대"))

# 2. 원기둥 / 원뿔 / 원뿔대
elif category == "원기둥/원뿔/원뿔대":
    st.sidebar.subheader("상세 설정")
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    height = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    r_bottom = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    res = 60 # 원을 표현하기 위한 해상도

    if sub_type == "원기둥":
        fig.add_trace(create_general_mesh(res, r_bottom, r_bottom, height, 'gold', "원기둥"))
    elif sub_type == "원뿔":
        fig.add_trace(create_general_mesh(res, r_bottom, 0, height, 'tomato', "원뿔"))
    elif sub_type == "원뿔대":
        r_top = st.sidebar.slider("윗면 반지름", 0.1, r_bottom-0.1, r_bottom/2)
        fig.add_trace(create_general_mesh(res, r_bottom, r_top, height, 'lime', "원뿔대"))

# 3. 정다면체
elif category == "정다면체 (Platonic Solids)":
    solid_type = st.sidebar.selectbox(
        "도형 선택", 
        ["정사면체 (Tetrahedron)", "정육면체 (Cube)", "정팔면체 (Octahedron)", 
         "정십이면체 (Dodecahedron)", "정이십면체 (Icosahedron)"]
    )
    size = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
    fig.add_trace(get_platonic_solid(solid_type, size))
    
    if "십이면체" in solid_type or "이십면체" in solid_type:
         st.info("💡 Tip: 이 도형은 황금비(Phi ≈ 1.618) 좌표계를 사용하여 그려집니다.")

# 4. 구
elif category == "구 (Sphere)":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.8, name="Sphere"))

# --- 그래프 공통 레이아웃 ---
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5], visible=True),
        yaxis=dict(range=[-5, 5], visible=True),
        zaxis=dict(range=[-5, 8], visible=True),
        aspectmode='data' # 비율 유지
    ),
    margin=dict(r=10, l=10, b=10, t=10)
)

st.plotly_chart(fig, use_container_width=True)
