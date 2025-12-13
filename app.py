import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="종합 입체도형 관측기", layout="wide")
st.title("📐 종합 입체도형 관측소")

# --- 사이드바: 메뉴 선택 ---
st.sidebar.header("설정")
category = st.sidebar.radio(
    "도형 종류 선택",
    ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구 (Sphere)"]
)

# --- 공통 함수: 다각형/원형 기둥, 뿔, 대 생성 로직 ---
def create_general_mesh(n, r_bottom, r_top, height, color='cyan'):
    """
    n: 각형의 수 (원은 50 이상)
    r_bottom: 밑면의 반지름 (중심에서 꼭짓점까지 거리)
    r_top: 윗면의 반지름 (0이면 뿔, r_bottom과 같으면 기둥)
    height: 높이
    """
    # 각도 생성 (0부터 2pi까지 n등분)
    theta = np.linspace(0, 2*np.pi, n+1)
    
    # 좌표 계산
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta)

    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height)

    # 점 합치기 (윗면 점들 + 아랫면 점들 + 위/아래 중심점)
    # 인덱스 구성: 0~n (윗면 테두리), n+1~2n+1 (아랫면 테두리), 2n+2 (윗면 중심), 2n+3 (아랫면 중심)
    x = np.concatenate([x_top, x_bottom, [0], [0]])
    y = np.concatenate([y_top, y_bottom, [0], [0]])
    z = np.concatenate([z_top, z_bottom, [height], [0]])
    
    # 면(Triangle) 구성 (i, j, k 인덱스)
    i_list, j_list, k_list = [], [], []

    # 옆면 구성
    for idx in range(n):
        # 사각형을 두 개의 삼각형으로 분할
        # 삼각형 1: (top[idx], bottom[idx], bottom[idx+1])
        i_list.append(idx)
        j_list.append(n + 1 + idx)
        k_list.append(n + 1 + idx + 1)

        # 삼각형 2: (top[idx], bottom[idx+1], top[idx+1])
        i_list.append(idx)
        j_list.append(n + 1 + idx + 1)
        k_list.append(idx + 1)

    # 윗면 뚜껑 (r_top > 0 일 때만)
    if r_top > 0:
        center_top_idx = 2 * n + 2
        for idx in range(n):
            i_list.append(idx)
            j_list.append(idx + 1)
            k_list.append(center_top_idx)

    # 아랫면 바닥 (r_bottom > 0 일 때만)
    if r_bottom > 0:
        center_bottom_idx = 2 * n + 3
        for idx in range(n):
            i_list.append(n + 1 + idx)
            j_list.append(center_bottom_idx)
            k_list.append(n + 1 + idx + 1)

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_list, j=j_list, k=k_list,
        opacity=0.6,
        color=color,
        flatshading=True, # 각진 느낌 살리기
        name='Shape'
    )

# --- 정다면체 데이터 ---
def get_platonic_solid(name, size):
    # 간단한 구현을 위해 정사면체, 정육면체, 정팔면체만 예시로 구현
    # 정십이면체/정이십면체는 좌표가 복잡하여 생략하거나 라이브러리(scipy) 활용 권장
    if name == "정사면체 (Tetrahedron)":
        # 꼭짓점 4개
        x = [1, -1, 1, -1]
        y = [1, 1, -1, -1]
        z = [1, -1, -1, 1]
        x = np.array(x) * size
        y = np.array(y) * size
        z = np.array(z) * size
        i = [0, 0, 0, 1]
        j = [1, 2, 3, 2]
        k = [2, 3, 1, 3]
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='magenta', opacity=0.8)

    elif name == "정육면체 (Cube)":
        return create_general_mesh(4, size, size, size*2, 'cyan') # 4각기둥 활용

    elif name == "정팔면체 (Octahedron)":
        x = [0, 0, size, -size, 0, 0]
        y = [0, 0, 0, 0, size, -size]
        z = [size, -size, 0, 0, 0, 0]
        i = [0, 0, 0, 0, 1, 1, 1, 1]
        j = [2, 4, 3, 5, 2, 5, 3, 4]
        k = [4, 3, 5, 2, 5, 3, 4, 2]
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='orange', opacity=0.8)
    
    return None

# --- 메인 로직 ---
fig = go.Figure()

if category == "각기둥/각뿔/각뿔대":
    st.sidebar.subheader("상세 설정")
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n_sides = st.sidebar.number_input("밑면의 각 수 (n)", min_value=3, max_value=20, value=4, step=1)
    height = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    radius_bottom = st.sidebar.slider("밑면 크기(반지름)", 1.0, 5.0, 3.0)

    if sub_type == "각기둥":
        # 윗면 크기 = 아랫면 크기
        fig.add_trace(create_general_mesh(n_sides, radius_bottom, radius_bottom, height, 'skyblue'))
    elif sub_type == "각뿔":
        # 윗면 크기 = 0
        fig.add_trace(create_general_mesh(n_sides, radius_bottom, 0, height, 'salmon'))
    elif sub_type == "각뿔대":
        # 윗면 크기 < 아랫면 크기 (사용자 입력)
        radius_top = st.sidebar.slider("윗면 크기(반지름)", 0.1, radius_bottom-0.1, radius_bottom/2)
        fig.add_trace(create_general_mesh(n_sides, radius_bottom, radius_top, height, 'lightgreen'))

elif category == "원기둥/원뿔/원뿔대":
    st.sidebar.subheader("상세 설정")
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    height = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    radius_bottom = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    resolution = 60 # 원을 표현하기 위한 다각형 수

    if sub_type == "원기둥":
        fig.add_trace(create_general_mesh(resolution, radius_bottom, radius_bottom, height, 'gold'))
    elif sub_type == "원뿔":
        fig.add_trace(create_general_mesh(resolution, radius_bottom, 0, height, 'tomato'))
    elif sub_type == "원뿔대":
        radius_top = st.sidebar.slider("윗면 반지름", 0.1, radius_bottom-0.1, radius_bottom/2)
        fig.add_trace(create_general_mesh(resolution, radius_bottom, radius_top, height, 'lime'))

elif category == "정다면체":
    solid_type = st.sidebar.selectbox("도형 선택", ["정사면체 (Tetrahedron)", "정육면체 (Cube)", "정팔면체 (Octahedron)"])
    size = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
    fig.add_trace(get_platonic_solid(solid_type, size))
    st.info("※ 참고: 정십이면체와 정이십면체는 복잡한 좌표 계산이 필요하여 이 데모에서는 제외되었습니다.")

elif category == "구 (Sphere)":
    radius = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    
    # 구 그리기 로직
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.8))

# --- 그래프 공통 레이아웃 ---
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-5, 5], title='X'),
        yaxis=dict(range=[-5, 5], title='Y'),
        zaxis=dict(range=[-2, 8], title='Z'), # 높이 고려하여 Z축 조정
        aspectmode='data' # 실제 비율대로 보이기
    ),
    margin=dict(r=10, l=10, b=10, t=10)
)

st.plotly_chart(fig, use_container_width=True)
