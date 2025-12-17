import streamlit as st
import plotly.graph_objects as go
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="원기둥 맞춤형 생성기", layout="wide")
st.title("📐 원기둥/원뿔 전용 깔끔한 생성기")
st.caption("지저분한 선을 모두 없애고, 교과서처럼 '윤곽선'과 '점선'만 그립니다.")

# 스타일 설정 (빨간 버튼 등)
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.error("⚠️ **중요:** 마우스로 회전하면 '점선'의 위치가 어긋나 보입니다! (파이썬 계산 한계) **반드시 왼쪽 슬라이더로 회전시켜주세요.**")

# --- 1. 사이드바 설정 ---
with st.sidebar:
    st.header("1. 도형 설정")
    # 원기둥이 메인이므로 맨 앞에 배치
    shape_type = st.radio("도형 종류", ["원기둥", "원뿔", "원뿔대"], horizontal=True)
    
    # 파라미터
    radius_top = 0.0
    radius_bottom = 2.0
    height = st.slider("높이", 1.0, 10.0, 4.0)
    
    if shape_type == "원기둥":
        r = st.slider("반지름", 0.5, 5.0, 2.0)
        radius_top = radius_bottom = r
    elif shape_type == "원뿔":
        radius_bottom = st.slider("밑면 반지름", 0.5, 5.0, 2.0)
        radius_top = 0.0
    elif shape_type == "원뿔대":
        radius_bottom = st.slider("밑면 반지름", 0.5, 5.0, 3.0)
        radius_top = st.slider("윗면 반지름", 0.5, 5.0, 1.5)

    st.write("---")
    st.header("2. 뷰(시점) 설정")
    st.info("여기를 조절해야 점선이 정확하게 나옵니다.")
    
    # 카메라 각도 (Degree)
    azimuth = st.slider("가로 회전 (Azimuth)", 0, 360, 45)
    elevation = st.slider("세로 회전 (Elevation)", 0, 90, 30)
    
    # 뷰 옵션
    show_surface = st.checkbox("면 색칠하기 (흰색 반투명)", value=True)
    line_color = "black"

# --- 2. 수학적 계산 (윤곽선 추출) ---

def get_cylinder_geometry(rt, rb, h, az_deg, el_deg):
    # 각도를 라디안으로 변환
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    
    # 1. 윤곽선 (Silhouette Lines) 계산
    # 카메라가 az 각도에 있을 때, 원기둥의 윤곽선은 az + 90도, az - 90도 위치에 존재함
    # 수학적으로 접평면이 시선과 평행한 지점
    
    t_left = az + np.pi/2
    t_right = az - np.pi/2
    
    lines = []
    
    # 왼쪽 윤곽선
    lines.append({
        'x': [rb * np.cos(t_left), rt * np.cos(t_left)],
        'y': [rb * np.sin(t_left), rt * np.sin(t_left)],
        'z': [-h/2, h/2],
        'type': 'solid'
    })
    
    # 오른쪽 윤곽선
    lines.append({
        'x': [rb * np.cos(t_right), rt * np.cos(t_right)],
        'y': [rb * np.sin(t_right), rt * np.sin(t_right)],
        'z': [-h/2, h/2],
        'type': 'solid'
    })
    
    # 2. 밑면/윗면 원 그리기
    # 원을 카메라 기준 '앞쪽(visible)'과 '뒤쪽(hidden)'으로 나눔
    # 카메라 벡터 (x, y) 방향 = (cos(az), sin(az))
    # 원 위의 점 (cos(t), sin(t))
    # 내적(Dot Product)을 통해 앞/뒤 판별: cos(t-az) > 0 이면 앞, < 0 이면 뒤
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    def split_circle(r, z_pos, is_top=False):
        # 윗면은 보통 다 보임 (Elevation > 0 일 때)
        # 아랫면은 앞만 보이고 뒤는 가려짐
        
        # 원 좌표
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, z_pos)
        
        # 카메라 방향과의 내적 계산을 위한 각도 차이
        # Elevation이 90도(위에서 수직)면 다 실선, 0도면 앞뒤 구분 필요
        # 간단한 로직: 아랫면(Bottom)의 경우 뒤쪽 절반은 점선
        
        if is_top:
            # 윗면은 전체 실선 (우리가 위에서 내려다보므로)
            return [{'x': x, 'y': y, 'z': z, 'type': 'solid'}]
        else:
            # 아랫면: 카메라 반대편(뒤쪽)은 점선
            # 카메라가 az 방향에 있음. 
            # 점의 각도 t에 대해, cos(t - az)가 양수면 카메라 쪽, 음수면 반대쪽
            
            # 배열 마스킹
            # 각도 차이 정규화 (-pi ~ pi)
            angle_diff = (theta - az + np.pi) % (2*np.pi) - np.pi
            
            # 카메라 쪽 (앞면)
            mask_front = (np.abs(angle_diff) <= np.pi/2)
            # 반대 쪽 (뒷면)
            mask_back = ~mask_front
            
            # 끊어진 선을 연결하지 않기 위해 None 삽입 로직은 생략하고,
            # 단순히 Scatter로 그릴 때 점들을 분리해서 처리해야 함.
            # 여기서는 편의상 마스크된 좌표를 그대로 반환 (Plotly가 알아서 끊음)
            
            res = []
            # 실선 부분 (앞)
            res.append({
                'x': x[mask_front], 'y': y[mask_front], 'z': z[mask_front], 'type': 'solid'
            })
            # 점선 부분 (뒤)
            res.append({
                'x': x[mask_back], 'y': y[mask_back], 'z': z[mask_back], 'type': 'dotted'
            })
            return res

    circle_lines = []
    # 윗면 (반지름 > 0 일 때만)
    if rt > 0.01:
        circle_lines.extend(split_circle(rt, h/2, is_top=True))
        
    # 아랫면
    circle_lines.extend(split_circle(rb, -h/2, is_top=False))
    
    return lines + circle_lines

# --- 3. 시각화 ---

data = []

# (1) 선 그리기
lines_data = get_cylinder_geometry(radius_top, radius_bottom, height, azimuth, elevation)

for line in lines_data:
    mode = "lines"
    line_style = dict(color="black", width=4)
    
    if line['type'] == 'dotted':
        line_style['dash'] = 'dash' # 점선 설정
        line_style['width'] = 3     # 점선은 조금 얇게
    
    data.append(go.Scatter3d(
        x=line['x'], y=line['y'], z=line['z'],
        mode=mode,
        line=line_style,
        showlegend=False,
        hoverinfo='skip'
    ))

# (2) 면 색칠하기 (옵션)
if show_surface:
    # 원기둥 옆면 메쉬 생성
    n_mesh = 60
    t_mesh = np.linspace(0, 2*np.pi, n_mesh)
    z_mesh = np.linspace(-height/2, height/2, 10)
    t_grid, z_grid = np.meshgrid(t_mesh, z_mesh)
    
    # 선형 보간 (원뿔대 대응)
    # z가 -h/2일 때 r=rb, z가 h/2일 때 r=rt
    # 비율 alpha = (z - (-h/2)) / h = (z + h/2) / h
    alpha = (z_grid + height/2) / height
    r_grid = radius_bottom * (1 - alpha) + radius_top * alpha
    
    x_surf = r_grid * np.cos(t_grid)
    y_surf = r_grid * np.sin(t_grid)
    z_surf = z_grid
    
    data.append(go.Surface(
        x=x_surf, y=y_surf, z=z_surf,
        colorscale=[[0, '#eeeeee'], [1, '#eeeeee']], # 흰색/회색
        showscale=False,
        opacity=0.7, # 반투명
        lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.1, specular=0.1)
    ))

# --- 4. 카메라 설정 ---
# 구면 좌표 -> 직교 좌표 (카메라 위치)
cam_r = 2.5 * height # 거리 자동 조절
cam_x = cam_r * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
cam_y = cam_r * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
cam_z = cam_r * np.sin(np.radians(elevation))

layout = go.Layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=dict(
            eye=dict(x=cam_x/height, y=cam_y/height, z=cam_z/height), # 정규화된 좌표 필요
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    height=700
)

fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig, use_container_width=True)
