import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 1. Scipy 안전하게 불러오기 ---
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (완성판)")

# --- 사이드바 설정 ---
st.sidebar.header("설정")
category = st.sidebar.radio(
    "도형 카테고리",
    ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"]
)

# --- [핵심] 입체감을 살리는 조명 설정 ---
# 이 설정이 없으면 도형이 색종이처럼 평평하게 보입니다.
light_config = dict(ambient=0.4, diffuse=0.6, roughness=0.1, specular=0.3)
light_pos = dict(x=10, y=10, z=10)

# --- 도형 생성 함수 1: 기둥/뿔/대 ---
def create_mesh(n, rb, rt, h, color, name):
    theta = np.linspace(0, 2*np.pi, n+1)
    
    # 좌표 계산 (변수명 오류 수정 완료)
    x_bottom = rb * np.cos(theta)
    y_bottom = rb * np.sin(theta)
    x_top = rt * np.cos(theta)
    y_top = rt * np.sin(theta)
    
    # 점들을 합치기
    x = np.concatenate([x_top, x_bottom, [0], [0]])
    y = np.concatenate([y_top, y_bottom, [0], [0]])
    z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
    
    i, j, k = [], [], []
    
    # 옆면 만들기
    for idx in range(n):
        i.extend([idx, idx])
        j.extend([n+1+idx, n+1+idx+1])
        k.extend([n+1+idx+1, idx+1])
    
    # 뚜껑 만들기 (반지름이 있을 때만)
    if rt > 0:
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
        
    # 바닥 만들기 (반지름이 있을 때만)
    if rb > 0:
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k, 
        color=color, 
        opacity=1.0, 
        flatshading=True,
        lighting=light_config, # 조명 적용
        lightposition=light_pos,
        name=name
    )

# --- 도형 생성 함수 2: 정다면체 ---
def create_platonic(name, size):
    if not has_scipy: return go.Mesh3d()
    phi = (1 + np.sqrt(5)) / 2
    points = []
    
    if "정사면체" in name: points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
    elif "정육면체" in name: 
        for x in [-1,1]: 
            for y in [-1,1]: 
                for z in [-1,1]: points.append([x,y,z])
    elif "정팔면체" in name: points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    elif "정십이면체" in name:
        for x in [-1,1]: 
            for y in [-1,1]: 
                for z in [-1,1]: points.append([x,y,z])
        for i in [-1,1]: 
            for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
    elif "정이십면체" in name:
        for i in [-1,1]: 
            for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

    points = np.array(points) * size
    hull = ConvexHull(points)
    
    return go.Mesh3d(
        x=points[:,0], y=points[:,1], z=points[:,2], 
        i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2], 
        color='#FF00FF', # 마젠타 색상
        opacity=1.0, 
        flatshading=True,
        lighting=light_config, # 조명 적용
        lightposition=light_pos,
        name=name
    )

# --- 메인 실행 로직 ---
fig = go.Figure()
max_limit = 5.0 # 카메라 범위를 결정할 변수

if category == "각기둥/각뿔/각뿔대":
    sub = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4, step=1)
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    
    rt = rb if sub == "각기둥" else (0 if sub == "각뿔" else st.sidebar.slider("윗면 반지름", 0.1, rb, rb/2))
    color = "#00CCFF" if sub=="각기둥" else ("#FF6666" if sub=="각뿔" else "#66FF66")
    
    fig.add_trace(create_mesh(n, rb, rt, h, color, sub))
    max_limit = max(h, rb) * 1.5 # 도형 크기에 맞춰 시야 확보

elif category == "원기둥/원뿔/원뿔대":
    sub = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    rt = rb if sub == "원기둥" else (0 if sub == "원뿔" else st.sidebar.slider("윗면 반지름", 0.1, rb, rb/2))
    
    color = "#FFD700" if sub=="원기둥" else ("#FF4500" if sub=="원뿔" else "#32CD32")
    
    fig.add_trace(create_mesh(60, rb, rt, h, color, sub))
    max_limit = max(h, rb) * 1.5

elif category == "정다면체":
    if has_scipy:
        sub = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        s = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
        fig.add_trace(create_platonic(sub, s))
        max_limit = s * 2.0
    else:
        st.error("scipy 라이브러리가 필요합니다. requirements.txt를 확인해주세요.")

elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', lighting=light_config))
    max_limit = r * 1.5

# --- [핵심] 카메라 자동 조정 및 배경 설정 ---
fig.update_layout(
    scene=dict(
        # 축의 범위를 도형 크기(max_limit)에 맞춰서 강제로 넓혀줍니다.
        xaxis=dict(range=[-max_limit, max_limit], title='X', backgroundcolor="white"),
        yaxis=dict(range=[-max_limit, max_limit], title='Y', backgroundcolor="white"),
        zaxis=dict(range=[-max_limit/2, max_limit*1.2], title='Z', backgroundcolor="white"),
        aspectmode='cube' # 찌그러짐 방지
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
