def make_prism_like(n_sides, r_bottom, r_top, height):
    """
    깔끔한 모서리(Wireframe)와 평평한 면(Flat Shading)을 적용한 버전
    """
    # 1. 좌표 생성
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    
    # 밑면과 윗면 좌표 (마지막 점은 0번 점과 같아 도형을 닫음)
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)
    z_bottom = np.zeros_like(theta)
    
    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top = np.full_like(theta, height)
    
    # --- [Step 1] 면(Face) 그리기 (Mesh3d) ---
    # 옆면 구성을 위한 데이터 준비 (마지막 중복 점 제외하고 연결)
    # n_sides 개수만큼만 사용하여 인덱싱
    
    # 좌표 배열 합치기 (마지막 닫는 점 제외)
    xb_mesh = x_bottom[:-1]
    yb_mesh = y_bottom[:-1]
    zb_mesh = z_bottom[:-1]
    xt_mesh = x_top[:-1]
    yt_mesh = y_top[:-1]
    zt_mesh = z_top[:-1]
    
    x_mesh = np.concatenate([xb_mesh, xt_mesh])
    y_mesh = np.concatenate([yb_mesh, yt_mesh])
    z_mesh = np.concatenate([zb_mesh, zt_mesh])
    
    # 인덱스 생성
    i = np.arange(n_sides)
    n = n_sides
    
    # 삼각형 1 (아랫면i -> 아랫면i+1 -> 윗면i)
    # % n 을 사용하여 마지막 점이 다시 0번 점과 연결되도록 함
    i_list = np.concatenate([i, i])
    j_list = np.concatenate([(i + 1) % n, i + n])
    k_list = np.concatenate([i + n, (i + 1) % n + n])
    
    # Mesh 생성: flatshading=True가 핵심
    mesh = go.Mesh3d(
        x=x_mesh, y=y_mesh, z=z_mesh,
        i=i_list, j=j_list, k=k_list,
        color='skyblue',
        opacity=0.8,
        flatshading=True,  # <--- 중요: 삼각형 경계를 부드럽게 숨김
        lighting=dict(ambient=0.5, diffuse=0.5), # 조명 조정
        name='Face'
    )
    
    # --- [Step 2] 모서리 선(Edge Lines) 그리기 ---
    # Plotly에서 선을 끊어서 그리려면 좌표 사이에 None을 넣어야 함
    
    x_lines = []
    y_lines = []
    z_lines = []
    
    # 1) 밑면 테두리 & 2) 윗면 테두리
    # 이미 theta가 한바퀴 돌아서 닫혀있으므로 그대로 사용
    x_lines.extend(x_bottom); x_lines.append(None)
    y_lines.extend(y_bottom); y_lines.append(None)
    z_lines.extend(z_bottom); z_lines.append(None)
    
    x_lines.extend(x_top); x_lines.append(None)
    y_lines.extend(y_top); y_lines.append(None)
    z_lines.extend(z_top); z_lines.append(None)
    
    # 3) 옆면 수직 선 (기둥/뿔의 옆 모서리)
    for k in range(n_sides):
        x_lines.extend([x_bottom[k], x_top[k], None])
        y_lines.extend([y_bottom[k], y_top[k], None])
        z_lines.extend([z_bottom[k], z_top[k], None])
        
    lines = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='black', width=3), # 검은색 굵은 선
        name='Edge'
    )
    
    return [mesh, lines] # 두 개의 Trace를 리스트로 반환
