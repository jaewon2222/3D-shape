import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 1. Scipy 안전 로딩 ---
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (완전 고정판)")

# --- 사이드바 ---
st.sidebar.header("설정")
category = st.sidebar.radio("도형 카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"])

# --- [핵심 1] 조명 설정 (이게 있어야 입체감이 생깁니다) ---
# 그림자를 강하게 줘서 모양이 잘 보이게 하는 설정입니다.
light_config = dict(
    ambient=0.3,    # 전체 밝기
    diffuse=0.8,    # 빛 반사
    roughness=0.1,  # 표면 거칠기
    specular=0.5    # 하이라이트
)
# 조명을 오른쪽 위에서 비춥니다.
light_pos = dict(x=10, y=10, z=10)

# --- 도형 생성 함수 ---
def create_mesh(n, rb, rt, h, color, name):
    theta = np.linspace(0, 2*np.pi, n+1)
    
    # [핵심 2] 변수명 꼬이지 않게 또박또박 씀
    x_bottom = rb * np.cos(theta)
    y_bottom = rb * np.sin(theta)
    x_top = rt * np.cos(theta)
    y_top = rt * np.sin(theta)
    
    # 좌표 합치기
    x = np.concatenate([x_top, x_bottom, [0], [0]])
    y = np.concatenate([y_top, y_bottom, [0], [0]])
    z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
    
    i, j, k = [], [], []
    
    # 옆면 만들기
    for idx in range(n):
        i.extend([idx, idx])
        j.extend([n+1+idx, n+1+idx+1])
        k.extend([n+1+idx+1, idx+1])
        
    # 뚜껑 덮기
    if rt > 0:
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
        
    # 바닥 깔기
    if rb > 0:
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k, 
        color=color, 
        opacity=1.0, 
        flatshading=True,  # 각진 느낌 살리기
        lighting=light_config, # 조명 켜기!
        lightposition=light_pos,
        name=name
    )

def create_platonic(name, size):
    if not has_scipy: return go.Mesh3d()
    phi = (1 + np.sqrt(5)) / 2
    points = []
    
    if "정사면체" in name: points = [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
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
        color='#FF8800', # 잘 보이는 주황색
        opacity=1.0, 
        flatshading=True,
        lighting=light_config, # 조명 켜기!
        lightposition=light_pos,
        name=name
    )

# --- 메인 실행 로직 ---
fig = go.Figure()

# 1. 각기둥/각뿔
if category == "각기둥/각뿔/각뿔대":
    sub = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 8.0, 5.0)
    rb = st.sidebar.slider("밑면", 1.0, 5.0, 3.0)
    rt = rb if sub == "각기둥" else (0 if sub == "각뿔" else st.sidebar.slider("윗면", 0.1, rb, rb/2))
    
    # 색상: 시원한 하늘색
    fig.add_trace(create_mesh(n, rb, rt, h, '#00BFFF', sub))

# 2. 원기둥/원뿔
elif category == "원기둥/원뿔/원뿔대":
    sub = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 8.0, 5.0)
    rb = st.sidebar.slider("밑면", 1.0, 5.0, 3.0)
    rt = rb if sub == "원기둥" else (0 if sub == "원뿔" else st.sidebar.slider("윗면", 0.1, rb, rb/2))
    
    # 색상: 밝은 노랑색
    fig.add_trace(create_mesh(60, rb, rt, h, '#FFD700', sub))

# 3. 정다면체
elif category == "정다면체":
    if has_scipy:
        sub = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        s = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
        fig.add_trace(create_platonic(sub, s))
    else:
        st.error("Scipy 라이브러리가 필요합니다.")

# 4. 구
elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis', lighting=light_config))

# --- [핵심 3] 카메라 & 방 크기 완전 고정 ---
# 이것 때문에 안 보이는 일은 절대 없습니다.
fixed_range = [-10, 10]

fig.update_layout(
    scene=dict(
        # X, Y, Z축을 -10부터 10까지 강제로 고정
        xaxis=dict(range=fixed_range, title='X'),
        yaxis=dict(range=fixed_range, title='Y'),
        zaxis=dict(range=[-5, 15], title='Z'), # 높이는 조금 더 위로
        
        # 비율을 1:1:1로 강제 (찌그러짐 방지)
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
