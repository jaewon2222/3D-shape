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
st.title("📐 3D 입체도형 관측소 (깔끔한 윤곽선)")
st.markdown("정다면체의 **불필요한 대각선을 제거**하여 더욱 깔끔하게 다듬었습니다.")

# --- 사이드바 ---
st.sidebar.header("설정")
category = st.sidebar.radio("도형 카테고리", ["각기둥/각뿔/각뿔대", "원기둥/원뿔/원뿔대", "정다면체", "구"])

fig = go.Figure()

# --- 설정값 ---
line_width = 8
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

    x_lines, y_lines, z_lines = [], [], []
    if rt > 0:
        x_lines.extend(list(x_top) + [x_top[0]] + [None])
        y_lines.extend(list(y_top) + [y_top[0]] + [None])
        z_lines.extend([h]*(n+1) + [None])
    x_lines.extend(list(x_bot) + [x_bot[0]] + [None])
    y_lines.extend(list(y_bot) + [y_bot[0]] + [None])
    z_lines.extend([0]*(n+1) + [None])
    for idx in range(n):
        x_lines.extend([x_top[idx], x_bot[idx], None])
        y_lines.extend([y_top[idx], y_bot[idx], None])
        z_lines.extend([h, 0, None])

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#00BFFF', opacity=mesh_opacity, flatshading=True, lighting=lighting_effects, name='면'))
    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=line_color, width=line_width), name='윤곽선'))


# ========================================================
# 2. 원기둥 / 원뿔 / 원뿔대
# ========================================================
elif category == "원기둥/원뿔/원뿔대":
    sub_type = st.sidebar.selectbox("종류", ["원기둥", "원뿔", "원뿔대"])
    h = st.sidebar.slider("높이", 1.0, 10.0, 5.0)
    rb = st.sidebar.slider("밑면 반지름", 1.0, 5.0, 3.0)
    n = 60
    
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

    x_lines, y_lines, z_lines = [], [], []
    if rt > 0:
        x_lines.extend(list(x_top) + [x_top[0]] + [None])
        y_lines.extend(list(y_top) + [y_top[0]] + [None])
        z_lines.extend([h]*(n+1) + [None])
    x_lines.extend(list(x_bot) + [x_bot[0]] + [None])
    y_lines.extend(list(y_bot) + [y_bot[0]] + [None])
    z_lines.extend([0]*(n+1) + [None])

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='#FFD700', opacity=mesh_opacity, flatshading=True, lighting=lighting_effects, name='면'))
    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=line_color, width=line_width), name='윤곽선'))

# ========================================================
# 3. 정다면체 (대각선 제거 로직 적용)
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
        
        # [핵심 로직] 대각선 제거하기
        # 1. 모든 가능한 선(Triangulation Edge)을 수집합니다.
        # 2. 선의 길이를 잽니다.
        # 3. 정다면체에서 '진짜 모서리'는 길이가 가장 짧습니다. 대각선은 더 깁니다.
        # 4. 가장 짧은 길이와 비슷한 선만 그립니다.
        
        # 모든 엣지 수집
        edges = set()
        for simplex in hull.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[2], simplex[0]))))
            
        # 길이 계산 및 필터링
        x_lines, y_lines, z_lines = [], [], []
        
        # 최소 길이 찾기 (이게 진짜 모서리 길이)
        min_dist = float('inf')
        edge_list = list(edges)
        distances = []
        
        for p1_idx, p2_idx in edge_list:
            dist = np.linalg.norm(points[p1_idx] - points[p2_idx])
            distances.append(dist)
            if dist < min_dist:
                min_dist = dist
        
        # 진짜 모서리만 그리기 (오차 허용 0.01)
        for i, (p1_idx, p2_idx) in enumerate(edge_list):
            if abs(distances[i] - min_dist) < 0.01:
                x_lines.extend([points[p1_idx][0], points[p2_idx][0], None])
                y_lines.extend([points[p1_idx][1], points[p2_idx][1], None])
                z_lines.extend([points[p1_idx][2], points[p2_idx][2], None])

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
