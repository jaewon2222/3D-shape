import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- Scipy 체크 ---
try:
    from scipy.spatial import ConvexHull
    has_scipy = True
except ImportError:
    has_scipy = False

st.set_page_config(page_title="3D 도형 관측기", layout="wide")
st.title("📐 3D 입체도형 관측소 (두꺼운 윤곽선)")
st.markdown("윤곽선을 **두껍게(Bold)** 만들고, 원형 도형의 **옆면 이음새 선을 제거**했습니다.")

# --- 사이드바 ---
st.sidebar.header("설정")
category = st.sidebar.radio("도형 카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"])

fig = go.Figure()

# --- 설정값 ---
line_width = 8  # 선 두께 (여기서 조절하세요)
line_color = 'black'
mesh_opacity = 1.0
lighting_effects = dict(ambient=0.7, diffuse=0.5, roughness=0.1, specular=0.2)

# ========================================================
# 1. 각기둥 / 각뿔 / 각뿔대
# ========================================================
if category == "각기둥/각뿔/각뿔대":
    sub_type = st.sidebar.selectbox("종류", ["각기둥", "각뿔", "각뿔대"])
    n = st.sidebar.number_input("n (각형)", 3, 20, 4)
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)

    if sub_type == "각기둥": rt = rb
    elif sub_type == "각뿔": rt = 0
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)

    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
    x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)

    x = np.concatenate([x_top, x_bot, [0], [0]])
    y = np.concatenate([y_top, y_bot, [0], [0]])
    z = np.concatenate([np.full(n, h), np.zeros(n), [h], [0]])

    i, j, k = [], [], []
    top_start, bot_start = 0, n
    top_center, bot_center = 2*n, 2*n+1

    for idx in range(n):
        next_idx = (idx + 1) % n
        i.extend([top_start + idx, top_start + idx])
        j.extend([bot_start + idx, bot_start + next_idx])
        k.extend([bot_start + next_idx, top_start + next_idx])
        if rt > 0:
            i.extend([top_start + idx]); j.extend([top_start + next_idx]); k.extend([top_center])
        if rb > 0:
            i.extend([bot_start + idx]); j.extend([bot_center]); k.extend([bot_start + next_idx])

    # 윤곽선 데이터 (각기둥은 모서리가 뚜렷해야 하므로 세로선 포함)
    x_lines, y_lines, z_lines = [], [], []
    
    # (1) 윗면 테두리
    if rt > 0: # 뿔이 아닐 때만
        x_lines.extend(list(x_top) + [x_top[0]] + [None])
        y_lines.extend(list(y_top) + [y_top[0]] + [None])
        z_lines.extend([h]*(n+1) + [None])
    
    # (2) 아랫면 테두리
    x_lines.extend(list(x_bot) + [x_bot[0]] + [None])
    y_lines.extend(list(y_bot) + [y_bot[0]] + [None])
    z_lines.extend([0]*(n+1) + [None])

    # (3) 세로 모서리 (각기둥/각뿔은 꼭지점이 있으므로 그려야 함)
    for idx in range(n):
        x_lines.extend([x_top[idx], x_bot[idx], None])
        y_lines.extend([y_top[idx], y_bot[idx], None])
        z_lines.extend([h, 0, None])

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#00BFFF', opacity=mesh_opacity, flatshading=True, lighting=lighting_effects, name='면'))
    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=line_color, width=line_width), name='윤곽선'))


# ========================================================
# 2. 원기둥 / 원뿔 / 원뿔대 (옆면 선 제거!)
# ========================================================
elif category == "원기둥/원뿔/원뿔대":
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    n = 60 # 매끄러운 원
    
    if sub_type == "원기둥": rt = rb
    elif sub_type == "원뿔": rt = 0
    else: rt = st.sidebar.slider("윗면 반지름", 0.1, rb-0.1, rb/2)

    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x_bot = rb * np.cos(theta); y_bot = rb * np.sin(theta)
    x_top = rt * np.cos(theta); y_top = rt * np.sin(theta)

    x = np.concatenate([x_top, x_bot, [0], [0]])
    y = np.concatenate([y_top, y_bot, [0], [0]])
    z = np.concatenate([np.full(n, h), np.zeros(n), [h], [0]])

    i, j, k = [], [], []
    for idx in range(n):
        next_idx = (idx + 1) % n
        i.extend([idx, idx]); j.extend([n + idx, n + next_idx]); k.extend([n + next_idx, next_idx])
        if rt > 0: i.extend([idx]); j.extend([next_idx]); k.extend([2*n])
        if rb > 0: i.extend([n+idx]); j.extend([2*n+1]); k.extend([n+next_idx])

    # [핵심 수정] 원형 도형은 세로 선(옆면)을 그리지 않습니다!
    x_lines, y_lines, z_lines = [], [], []
    
    # 윗면 동그라미
    if rt > 0:
        x_lines.extend(list(x_top) + [x_top[0]] + [None])
        y_lines.extend(list(y_top) + [y_top[0]] + [None])
        z_lines.extend([h]*(n+1) + [None])
    
    # 아랫면 동그라미
    x_lines.extend(list(x_bot) + [x_bot[0]] + [None])
    y_lines.extend(list(y_bot) + [y_bot[0]] + [None])
    z_lines.extend([0]*(n+1) + [None])

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FFD700', opacity=mesh_opacity, flatshading=True, lighting=lighting_effects, name='면'))
    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=line_color, width=line_width), name='윤곽선'))

# ========================================================
# 3. 정다면체
# ========================================================
elif category == "정다면체":
    if not has_scipy:
        st.error("Scipy가 필요합니다.")
    else:
        sub_type = st.sidebar.selectbox("도형", ["정사면체", "정육면체", "정팔면체", "정십이면체", "정이십면체"])
        size = st.sidebar.slider("크기", 1.0, 5.0, 3.0)
        phi = (1 + np.sqrt(5)) / 2
        points = []

        if sub_type == "정사면체": points = [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]
        elif sub_type == "정육면체":
            for x in [-1,1]:
                for y in [-1,1]:
                    for z in [-1,1]: points.append([x,y,z])
        elif sub_type == "정팔면체": points = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
        elif sub_type == "정십이면체":
            for x in [-1,1]:
                for y in [-1,1]:
                    for z in [-1,1]: points.append([x,y,z])
            for i in [-1,1]:
                for j in [-1,1]: points.extend([[0,i*phi,j/phi], [j/phi,0,i*phi], [i*phi,j/phi,0]])
        elif sub_type == "정이십면체":
            for i in [-1,1]:
                for j in [-1,1]: points.extend([[0,i,j*phi], [j*phi,0,i], [i,j*phi,0]])

        points = np.array(points) * size
        hull = ConvexHull(points) 
        
        x_lines, y_lines, z_lines = [], [], []
        for simplex in hull.simplices:
            for v_idx in list(simplex) + [simplex[0]]:
                x_lines.append(points[v_idx][0])
                y_lines.append(points[v_idx][1])
                z_lines.append(points[v_idx][2])
            x_lines.append(None); y_lines.append(None); z_lines.append(None)

        fig.add_trace(go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2], 
                                i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2], 
                                color='#FF8800', opacity=mesh_opacity, flatshading=True, lighting=lighting_effects))
        
        fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=line_color, width=line_width), name='윤곽선'))

# ========================================================
# 4. 구
# ========================================================
elif category == "구":
    r = st.sidebar.slider("반지름", 1.0, 5.0, 3.0)
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, 
        colorscale='Viridis', 
        lighting=lighting_effects,
        contours = {
            # 구의 격자선도 조금 두껍게(size는 간격이라 width 옵션이 제한적이지만 색상을 진하게)
            "x": {"show": True, "start": -r, "end": r, "size": r/4, "color":"black", "width": 4},
            "y": {"show": True, "start": -r, "end": r, "size": r/4, "color":"black", "width": 4},
            "z": {"show": True, "start": -r, "end": r, "size": r/4, "color":"black", "width": 4}
        }
    ))

# ========================================================
# [레이아웃]
# ========================================================
fig.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
