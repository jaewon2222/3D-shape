import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 시뮬레이터", layout="wide")
st.title("🧊 3D 입체도형 시뮬레이터")

# --- 사이드바 설정 ---
st.sidebar.header("도형 설정")
shape_type = st.sidebar.selectbox(
    "도형을 선택하세요",
    ("다각형 기둥/뿔/대 (Prism/Pyramid)", "원형 기둥/뿔/대 (Cylinder/Cone)", "구 (Sphere)")
)

# --- 3D 그리기 함수 (면 + 모서리 분리 버전) ---
def make_prism_like(n_sides, r_bottom, r_top, height):
    # 1. 좌표 생성 (0 ~ 2pi)
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    
    # 밑면과 윗면 좌표 생성
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta)
    
    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height)
    
    # --- [Step 1] 면(Face) 그리기 (Mesh3d) ---
    # 메쉬 구성을 위해 마지막 중복 점은 제외하고 슬라이싱
    xb_m, yb_m, zb_m = x_bottom[:-1], y_bottom[:-1], z_bottom[:-1]
    xt_m, yt_m, zt_m = x_top[:-1], y_top[:-1], z_top[:-1]
    
    # 좌표 합치기
    x_mesh = np.concatenate([xb_m, xt_m])
    y_mesh = np.concatenate([yb_m, yt_m])
    z_mesh = np.concatenate([zb_m, zt_m])
    
    # 인덱스 생성 (삼각형 2개로 사각형 면 만들기)
    n = n_sides
    i = np.arange(n)
    
    # 옆면을 구성하는 점들의 인덱스
    # 0~n-1: 밑면 점들, n~2n-1: 윗면 점들
    # 삼각형 1: 밑면(i) -> 밑면(i+1) -> 윗면(i)
    # 삼각형 2: 윗면(i) -> 밑면(i+1) -> 윗면(i+1)
    
    next_i = (i + 1) % n  # 마지막 점은 0번 점과 연결
    
    i_list = np.concatenate([i, i + n])
    j_list = np.concatenate([next_i, next_i])
    k_list = np.concatenate([i + n, next_i + n])
    
    # Mesh 객체 생성
    mesh = go.Mesh3d(
        x=x_mesh, y=y_mesh, z=z_mesh,
        i=i_list, j=j_list, k=k_list,
        color='skyblue',
        opacity=0.8,
        flatshading=True,  # 각진 느낌을 살림
        name='Face'
    )
    
    # --- [Step 2] 모서리 선(Edge Lines) 그리기 ---
    # 원형(변이 많음)일 때는 테두리를 굳이 그리지 않음 (너무 복잡해짐)
    lines = None
    if n_sides < 30: 
        x_lines, y_lines, z_lines = [], [], []
        
        # 밑면 테두리
        x_lines.extend(x_bottom); x_lines.append(None)
        y_lines.extend(y_bottom); y_lines.append(None)
        z_lines.extend(z_bottom); z_lines.append(None)
        
        # 윗면 테두리
        x_lines.extend(x_top); x_lines.append(None)
        y_lines.extend(y_top); y_lines.append(None)
        z_lines.extend(z_top); z_lines.append(None)
        
        # 옆면 세로선 (각 모서리)
        # 마지막 닫는 점까지 포함된 theta 배열 길이 사용
        for k in range(n_sides):
            x_lines.extend([x_bottom[k], x_top[k], None])
            y_lines.extend([y_bottom[k], y_top[k], None])
            z_lines.extend([z_bottom[k], z_top[k], None])

        lines = go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='black', width=4),
            name='Edge'
        )
    
    # 리스트로 반환 (lines가 없으면 mesh만)
    return [mesh, lines] if lines else [mesh]

def make_sphere(radius):
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi, theta = np.meshgrid(phi, theta)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi) + radius # 구의 중심을 z=radius로 올려서 바닥 위에 놓기
    
    return [go.Surface(x=x, y=y, z=z, colorscale='Blues', showscale=False, opacity=0.9)]

# --- 메인 로직 ---
fig = go.Figure()
traces = []

if shape_type == "다각형 기둥/뿔/대 (Prism/Pyramid)":
    sides = st.sidebar.slider("밑면의 변의 개수 (n)", 3, 12, 4)
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0이면 뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    
    traces = make_prism_like(sides, r_b, r_t, h)
    
    shape_name = "각기둥" if r_b == r_t else ("각뿔" if r_t == 0 else "각뿔대")
    st.subheader(f"{sides}{shape_name}")

elif shape_type == "원형 기둥/뿔/대 (Cylinder/Cone)":
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0이면 원뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    
    # 원형은 변의 개수를 60개로 설정
    traces = make_prism_like(60, r_b, r_t, h)
    
    shape_name = "원기둥" if r_b == r_t else ("원뿔" if r_t == 0 else "원뿔대")
    st.subheader(shape_name)

elif shape_type == "구 (Sphere)":
    r = st.sidebar.slider("반지름", 1.0, 10.0, 5.0)
    traces = make_sphere(r)
    st.subheader("구")

# --- [중요] 리스트로 받은 Trace들을 하나씩 추가 ---
for trace in traces:
    fig.add_trace(trace)

# --- 차트 레이아웃 ---
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False), # 축 눈금 숨기기 (깔끔하게)
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data' # 비율 왜곡 방지
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
