import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 1. Scipy 안전하게 불러오기 (에러 방지) ---
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

# --- 페이지 설정 ---
st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (수정판)")

# 경고 메시지: Scipy가 없을 경우
if not has_scipy:
    st.error("⚠️ 'scipy' 라이브러리가 설치되지 않았습니다. 정다면체가 보이지 않을 수 있습니다.")
    st.info("GitHub의 requirements.txt 파일에 'scipy'를 추가하고 앱을 재부팅(Reboot)해주세요.")

# --- 사이드바 ---
st.sidebar.header("설정")
category = st.sidebar.radio(
    "도형 카테고리",
    ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"]
)

# --- 도형 생성 함수들 ---
def create_mesh(n, rb, rt, h, color, name):
    theta = np.linspace(0, 2*np.pi, n+1)
    x_bot, y_bot = rb * np.cos(theta), rb * np.sin(theta)
    x_top, y_top = rt * np.cos(theta), rt * np.sin(theta)
    
    # 좌표 배열
    x = np.concatenate([x_top, x_bot, [0], [0]])
    y = np.concatenate([y_top, y_bot, [0], [0]])
    z = np.concatenate([np.full_like(theta, h), np.zeros_like(theta), [h], [0]])
    
    i, j, k = [], [], []
    for idx in range(n):
        # 옆면
        i.extend([idx, idx])
        j.extend([n+1+idx, n+1+idx+1])
        k.extend([n+1+idx+1, idx+1])
    
    # 뚜껑/바닥
    if rt > 0:
        for idx in range(n): i.extend([idx, idx+1, 2*n+2])
    if rb > 0:
        for idx in range(n): i.extend([n+1+idx, 2*n+3, n+1+idx+1])

    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=1.0, flatshading=True, name=name)

def create_platonic(name, size):
    if not has_scipy: return go.Mesh3d() # 에러 방지용 빈 객체
    
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
        color='cyan', opacity=1.0, flatshading=True, name=name
    )

# --- 메인 실행 로직 ---
fig = go.Figure()

if category == "각기둥/각뿔/각뿔대":
    sub = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면", 1.0, 5.0, 3.0)
    if sub=="각기둥": fig.add_trace(create_mesh(n, rb, rb, h, 'skyblue', sub))
    elif sub=="각뿔": fig.add_trace(create_mesh(n, rb, 0, h, 'salmon', sub))
    elif sub=="각뿔대": 
        rt = st.sidebar.slider("윗면", 0.1, rb, rb/2)
        fig.add_trace(create_mesh(n, rb, rt, h, 'lightgreen', sub))

elif category == "원기둥/원뿔/원뿔대":
    sub = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면", 1.0, 5.0, 3.0)
    if sub=="원기둥": fig.add_trace(create_mesh(60, rb, rb, h, 'gold', sub))
    elif sub=="원뿔": fig.add_trace(create_mesh(60, rb, 0, h, 'tomato', sub))
    elif sub=="원뿔대":
        rt = st.sidebar.slider("윗면", 0.1, rb, rb/2)
        fig.add_trace(create_mesh(60, rb, rt, h, 'lime', sub))

elif category == "정다면체":
    if has_scipy:
        sub = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        s = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
        fig.add_trace(create_platonic(sub, s))
    else:
        st.warning("Scipy 라이브러리가 없어서 정다면체를 표시할 수 없습니다.")

elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50))
    x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Viridis'))

# --- [중요] 레이아웃 설정: 왜곡 방지 + 자동 시점 ---
fig.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        # 'cube' 모드는 X,Y,Z 축 길이를 시각적으로 동일하게 강제합니다.
        # 비율은 맞추되, 범위는 데이터에 따라 자동으로 조절됩니다.
        aspectmode='cube' 
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
