import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 시뮬레이터", layout="wide")
st.title("🧊 3D Geometry Simulator")

# --- 사이드바 설정 ---
st.sidebar.header("도형 설정")
shape_type = st.sidebar.selectbox(
    "도형을 선택하세요",
    ("다각형 기둥/뿔/대 (Prism/Pyramid)", "원형 기둥/뿔/대 (Cylinder/Cone)", "구 (Sphere)")
)

# --- 3D 그리기 함수 ---
def make_prism_like(n_sides, r_bottom, r_top, height):
    """
    각기둥, 각뿔, 각뿔대, 원기둥, 원뿔, 원뿔대를 그리는 통합 함수
    """
    # 각도 생성 (0부터 2pi까지)
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    
    # 밑면 좌표
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta)
    
    # 윗면 좌표
    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height)
    
    # Plotly Mesh3d를 위한 데이터 구성
    # 옆면을 구성하기 위해 좌표를 순서대로 연결
    x = np.concatenate([x_bottom, x_top])
    y = np.concatenate([y_bottom, y_top])
    z = np.concatenate([z_bottom, z_top])
    
    # i: 현재 점, n: 한 층의 점 개수
    # 면(Face)을 구성하는 점의 인덱스 계산 (삼각형 메쉬)
    i = np.arange(n_sides)
    n = n_sides + 1
    
    # 옆면 삼각형 1: (밑면i, 밑면i+1, 윗면i)
    i_list = np.concatenate([i, i])
    j_list = np.concatenate([i + 1, i + n])
    k_list = np.concatenate([i + n, i + n + 1])
    
    # 윗면과 아랫면 채우기 (중심점 추가 방식 대신 간단히 팬(fan) 방식 사용 가능하나 여기선 생략하고 옆면 위주로 시각화)
    # 완전한 닫힌 도형을 위해서는 위/아래 뚜껑용 메쉬를 추가해야 합니다.
    
    return go.Mesh3d(x=x, y=y, z=z, i=i_list, j=j_list, k=k_list, opacity=0.8, color='skyblue', name='Shape')

def make_sphere(radius):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return go.Surface(x=x, y=y, z=z, colorscale='Viridis', showscale=False)

# --- 메인 로직 ---
fig = go.Figure()

if shape_type == "다각형 기둥/뿔/대 (Prism/Pyramid)":
    sides = st.sidebar.slider("밑면의 변의 개수 (n)", 3, 12, 4)
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0이면 뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    
    mesh = make_prism_like(sides, r_b, r_t, h)
    fig.add_trace(mesh)
    
    # 캡션 생성
    shape_name = "각기둥" if r_b == r_t else ("각뿔" if r_t == 0 else "각뿔대")
    st.subheader(f"{sides}{shape_name} 시각화")

elif shape_type == "원형 기둥/뿔/대 (Cylinder/Cone)":
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0이면 원뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    
    # 원형은 변의 개수를 60개 정도로 많이 주어 부드럽게 표현
    mesh = make_prism_like(60, r_b, r_t, h)
    fig.add_trace(mesh)
    
    shape_name = "원기둥" if r_b == r_t else ("원뿔" if r_t == 0 else "원뿔대")
    st.subheader(f"{shape_name} 시각화")

elif shape_type == "구 (Sphere)":
    r = st.sidebar.slider("반지름", 1.0, 10.0, 5.0)
    surface = make_sphere(r)
    fig.add_trace(surface)
    st.subheader("구 시각화")

# --- 차트 레이아웃 업데이트 ---
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
        zaxis=dict(range=[0, 20]),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

st.plotly_chart(fig, use_container_width=True)

# --- 정보 표시 ---
st.info("마우스를 드래그하여 도형을 회전하고 휠을 굴려 확대/축소해보세요.")
