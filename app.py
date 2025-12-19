import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 입체도형 시뮬레이터", layout="wide")
st.title("🧊 3D 입체도형 시뮬레이터 (완성판)")

# --- 사이드바 설정 ---
st.sidebar.header("도형 설정")
shape_type = st.sidebar.selectbox(
    "도형을 선택하세요",
    ("다각형 기둥/뿔/대", "원형 기둥/뿔/대", "구 (Sphere)")
)

# --- 유틸리티 함수: 다각형 뚜껑/바닥 만들기 ---
def create_cap(r, height, n_sides, is_top=True):
    """
    중심점과 테두리를 연결하여 다각형/원형의 면을 채우는 함수
    """
    if r <= 0: return None # 반지름이 0이면(뾰족한 뿔의 끝) 면을 만들 필요 없음

    # 1. 테두리 점 좌표 생성
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    x_edge = r * np.cos(theta)
    y_edge = r * np.sin(theta)
    z_val = height if is_top else 0
    z_edge = np.full_like(theta, z_val)

    # 2. 중심점 추가 (리스트의 맨 마지막에 추가)
    x = np.append(x_edge, 0)
    y = np.append(y_edge, 0)
    z = np.append(z_edge, z_val)

    # 3. 인덱스 생성 (Triangle Fan 방식)
    # 중심점(마지막 인덱스) -> i -> i+1
    center_idx = len(x) - 1
    i = np.arange(n_sides)
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=np.full(n_sides, center_idx), # 모든 삼각형의 시작은 중심점
        j=i,                            # 테두리 현재 점
        k=(i + 1) % (n_sides + 1),      # 테두리 다음 점
        color='skyblue',
        opacity=0.8,
        flatshading=True,
        name='Top' if is_top else 'Bottom'
    )

# --- 메인 그리기 함수 ---
def make_prism_like(n_sides, r_bottom, r_top, height):
    traces = []
    
    # 기본 좌표 생성
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta)
    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height)
    
    # 1. 옆면 (Side Walls) 그리기
    # 좌표 합치기
    x_side = np.concatenate([x_bottom[:-1], x_top[:-1]])
    y_side = np.concatenate([y_bottom[:-1], y_top[:-1]])
    z_side = np.concatenate([z_bottom[:-1], z_top[:-1]])
    
    i = np.arange(n_sides)
    n = n_sides
    next_i = (i + 1) % n
    
    # 옆면 삼각형 구성
    mesh_side = go.Mesh3d(
        x=x_side, y=y_side, z=z_side,
        i=np.concatenate([i, i + n]),
        j=np.concatenate([next_i, next_i]),
        k=np.concatenate([i + n, next_i + n]),
        color='skyblue',
        opacity=0.8,
        flatshading=True,
        name='Side'
    )
    traces.append(mesh_side)

    # 2. 바닥면 (Bottom Cap) 채우기
    bottom_cap = create_cap(r_bottom, 0, n_sides, is_top=False)
    if bottom_cap: traces.append(bottom_cap)

    # 3. 윗면 (Top Cap) 채우기
    top_cap = create_cap(r_top, height, n_sides, is_top=True)
    if top_cap: traces.append(top_cap)
    
    # 4. 모서리 선 (Wireframe) 그리기
    # 원형(n_sides >= 30)일 때는 테두리 선을 생략하여 깔끔하게 표현
    if n_sides < 30:
        x_lines, y_lines, z_lines = [], [], []
        
        # 바닥 테두리
        x_lines.extend(x_bottom); x_lines.append(None)
        y_lines.extend(y_bottom); y_lines.append(None)
        z_lines.extend(z_bottom); z_lines.append(None)
        
        # 윗면 테두리
        x_lines.extend(x_top); x_lines.append(None)
        y_lines.extend(y_top); y_lines.append(None)
        z_lines.extend(z_top); z_lines.append(None)
        
        # 옆면 세로선
        for k in range(n_sides):
            x_lines.extend([x_bottom[k], x_top[k], None])
            y_lines.extend([y_bottom[k], y_top[k], None])
            z_lines.extend([z_bottom[k], z_top[k], None])

        lines = go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='black', width=3),
            name='Edge'
        )
        traces.append(lines)
    
    return traces

def make_sphere(radius):
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi) + radius 
    return [go.Surface(x=x, y=y, z=z, colorscale='Blues', showscale=False, opacity=0.9)]

# --- 메인 실행 로직 ---
fig = go.Figure()
traces = []

if shape_type == "다각형 기둥/뿔/대":
    sides = st.sidebar.slider("밑면의 변의 개수 (n)", 3, 12, 4)
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0=뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    traces = make_prism_like(sides, r_b, r_t, h)
    
    name = "각기둥" if r_b == r_t else ("각뿔" if r_t == 0 else "각뿔대")
    st.subheader(f"{sides}{name}")

elif shape_type == "원형 기둥/뿔/대":
    r_b = st.sidebar.slider("밑면 반지름", 0.0, 10.0, 5.0)
    r_t = st.sidebar.slider("윗면 반지름 (0=원뿔)", 0.0, 10.0, 5.0)
    h = st.sidebar.slider("높이", 1.0, 20.0, 10.0)
    traces = make_prism_like(60, r_b, r_t, h) # 변 60개로 원 표현
    
    name = "원기둥" if r_b == r_t else ("원뿔" if r_t == 0 else "원뿔대")
    st.subheader(name)

elif shape_type == "구 (Sphere)":
    r = st.sidebar.slider("반지름", 1.0, 10.0, 5.0)
    traces = make_sphere(r)
    st.subheader("구")

# Trace 추가
for trace in traces:
    fig.add_trace(trace)

# 레이아웃 업데이트
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
