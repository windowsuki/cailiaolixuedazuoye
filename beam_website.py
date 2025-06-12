import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

# 初始化模式
if 'mode' not in st.session_state:
    st.session_state.mode = '计算模式'

# 模式切换按钮
mode = st.radio("请选择模式：", ["教学模式", "计算模式"], index=0 if st.session_state.mode == '教学模式' else 1)
st.session_state.mode = mode

# ------------------- 定义符号 -------------------
x, L = sp.symbols('x L', real=True)
F1, M1 = sp.symbols('F1 M1')
EI = sp.symbols('EI')
C1, C2 = sp.symbols('C1 C2')

# ------------------- 计算模式 -------------------
if st.session_state.mode == '计算模式':

# ------------------- 标题 -------------------
    st.title("🧮 计算模式")
    st.markdown("输入梁参数，计算弯矩、转角、挠度（支持多种载荷类型）")

    # ------------------- 参数输入区域 -------------------
    col1, col2 = st.columns(2)

    with col1:
        left_support = st.selectbox("左端支撑", ['fixed', 'pinned', 'free'])
        right_support = st.selectbox("右端支撑", ['fixed', 'pinned', 'free'])
        L_val = st.number_input("梁长 L", value=10.0)
        EI_val = st.number_input("弯曲刚度 EI", value=20000.0)

    with col2:
        num_point_loads = st.number_input("集中力数量", min_value=0, max_value=5, value=1, step=1)
        num_point_moments = st.number_input("集中力偶矩数量", min_value=0, max_value=5, value=0, step=1)
        num_uniform_loads = st.number_input("均匀分布力数量", min_value=0, max_value=5, value=0, step=1)
        num_linear_loads = st.number_input("线性分布力数量", min_value=0, max_value=5, value=0, step=1)

    # ------------------- 载荷输入 -------------------
    st.markdown("### 集中力")
    point_loads = []
    for i in range(num_point_loads):
        col1, col2 = st.columns(2)
        with col1:
            P_val = st.number_input(f"P{i+1} (大小)", value=100.0)
        with col2:
            a_val = st.number_input(f"a{i+1} (位置)", value=5.0)
        point_loads.append((P_val, a_val))

    st.markdown("### 集中力偶矩")
    point_moments = []
    for i in range(num_point_moments):
        col1, col2 = st.columns(2)
        with col1:
            M0_val = st.number_input(f"M0{i+1} (大小)", value=50.0)
        with col2:
            m0_pos_val = st.number_input(f"M0位置{i+1}", value=4.0)
        point_moments.append((M0_val, m0_pos_val))

    st.markdown("### 均匀分布力")
    uniform_loads = []
    for i in range(num_uniform_loads):
        col1, col2 = st.columns(2)
        with col1:
            q_val = st.number_input(f"q{i+1} (强度)", value=10.0)
        with col2:
            x1_val = st.number_input(f"起点x1_{i+1}", value=2.0)
            x2_val = st.number_input(f"终点x2_{i+1}", value=6.0)
        uniform_loads.append((q_val, x1_val, x2_val))

    st.markdown("### 线性分布力 (q = q0 + k(x - x1))")
    linear_loads = []
    for i in range(num_linear_loads):
        col1, col2 = st.columns(2)
        with col1:
            q0_val = st.number_input(f"q0_{i+1} (起点强度)", value=5.0)
            k_val = st.number_input(f"k_{i+1} (斜率)", value=1.0)
        with col2:
            lx1_val = st.number_input(f"起点x1_{i+1}", value=1.0)
            lx2_val = st.number_input(f"终点x2_{i+1}", value=4.0)
        linear_loads.append((q0_val, k_val, lx1_val, lx2_val))

    # ------------------- 生成边界条件 -------------------
    def generate_boundary_conditions(left_support, right_support, y, theta, M_expr):
        conditions = []
        if left_support == 'fixed':
            conditions.append(sp.Eq(y.subs(x, 0), 0))
            conditions.append(sp.Eq(theta.subs(x, 0), 0))
        elif left_support == 'pinned':
            conditions.append(sp.Eq(y.subs(x, 0), 0))
            conditions.append(sp.Eq(M_expr.subs(x, 0), 0))
        elif left_support == 'free':
            conditions.append(sp.Eq(F1, 0))
            conditions.append(sp.Eq(M1, 0))

        if right_support == 'fixed':
            conditions.append(sp.Eq(y.subs(x, L), 0))
            conditions.append(sp.Eq(theta.subs(x, L), 0))
        elif right_support == 'pinned':
            conditions.append(sp.Eq(y.subs(x, L), 0))
            conditions.append(sp.Eq(M_expr.subs(x, L), 0))
        elif right_support == 'free':
            conditions.append(sp.Eq(M_expr.subs(x, L), 0))
            #左侧杆支撑力等于整体受力
            total_force = 0
            # 添加集中力
            for P_i, a_i in point_loads:
                total_force += P_i
            # 添加均布载荷
            for q_i, x1_i, x2_i in uniform_loads:
                total_force += q_i * (x2_i - x1_i)
            # 添加线性分布力（一次函数）
            for q0_i, k_i, lx1_i, lx2_i in linear_loads:
                # 对 q(x) = q0_i + k_i*(x - lx1_i) 在区间 [lx1_i, lx2_i] 积分
                total_force += sp.integrate(q0_i + k_i * (x - lx1_i), (x, lx1_i, lx2_i))
            conditions.append(sp.Eq(F1, total_force))

        return conditions

    # ------------------- 计算 -------------------
    if st.button("开始计算"):
        M_expr = -M1 + F1 * x

        # 添加集中力
        for P_i, a_i in point_loads:
            M_expr -= P_i * sp.Heaviside(x - a_i) * (x - a_i)

        # 添加集中力偶矩
        for M0_i, m0_pos_i in point_moments:
            M_expr -= M0_i * sp.Heaviside(x - m0_pos_i)

        # 添加均匀分布力
        for q_i, x1_i, x2_i in uniform_loads:
            M_expr -= q_i * ((x - x1_i)**2 * sp.Heaviside(x - x1_i) - (x - x2_i)**2 * sp.Heaviside(x - x2_i)) / 2

        # 添加线性分布力（一次函数）
        for q0_i, k_i, lx1_i, lx2_i in linear_loads:
            x_ = sp.Symbol('x')
            deflin = lambda x_: q0_i + k_i*(x_ - lx1_i)
            expr = sp.integrate((x - x_)*deflin(x_), (x_, lx1_i, sp.Min(x, lx2_i)))
            M_expr -= sp.Piecewise((expr, x >= lx1_i), (0, True))

        theta = sp.integrate(M_expr / EI, x) + C1
        y = sp.integrate(theta, x) + C2

        param_subs = {L: L_val, EI: EI_val}
        equations = generate_boundary_conditions(left_support, right_support, y, theta, M_expr)
        equations_subs = [eq.subs(param_subs) for eq in equations]
        unknowns = [F1, M1, C1, C2]
        solution = sp.solve(equations_subs, unknowns, dict=True)

        if not solution:
            st.error("解不出来！检查边界条件或参数是否合理。")
        else:
            solution = solution[0]

            M_final = M_expr.subs(param_subs).subs(solution)
            theta_final = theta.subs(param_subs).subs(solution)
            y_final = y.subs(param_subs).subs(solution)

            M_func = sp.lambdify(x, M_final, 'numpy')
            theta_func = sp.lambdify(x, theta_final, 'numpy')
            y_func = sp.lambdify(x, y_final, 'numpy')

            x_vals = np.linspace(0, L_val, 400)
            M_vals = M_func(x_vals)
            theta_vals = theta_func(x_vals)
            y_vals = y_func(x_vals)

            fig, axs = plt.subplots(3, 1, figsize=(8, 10))

            axs[0].plot(x_vals, M_vals, label="Bending Moment M(x)")
            axs[0].set_ylabel("Moment")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(x_vals, theta_vals, label="Rotation θ(x)", color='orange')
            axs[1].set_ylabel("Rotation")
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(x_vals, y_vals, label="Deflection y(x)", color='green')
            axs[2].set_ylabel("Deflection")
            axs[2].set_xlabel("x")
            axs[2].legend()
            axs[2].grid(True)

            st.pyplot(fig)

            # ------------------- 绘制3d图形 -------------------
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')


            # 固定端（用一个矩形或者立方体）
            def draw_fixed_support(ax, x_pos, width=0.5, height=0.3, depth=0.3):
                y_range = np.linspace(-width / 2, width / 2, 2)
                z_range = np.linspace(0, height, 2)
                Y, Z = np.meshgrid(y_range, z_range)
                X = np.full_like(Y, x_pos)
                ax.plot_surface(X, Y, Z, color='gray', alpha=0.8)


            # 铰支座（简单地用一个三角锥或球替代示意）
            def draw_pinned_support(ax, x_pos, radius=0.1):
                u = np.linspace(0, 2 * np.pi, 12)
                v = np.linspace(0, np.pi, 12)
                x = radius * np.outer(np.cos(u), np.sin(v)) + x_pos
                y = radius * np.outer(np.sin(u), np.sin(v))
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='brown', alpha=0.9)


            if left_support == 'fixed':
                draw_fixed_support(ax, x_pos=0)
            elif left_support == 'pinned':
                draw_pinned_support(ax, x_pos=0)

            if right_support == 'fixed':
                draw_fixed_support(ax, x_pos=L_val)
            elif right_support == 'pinned':
                draw_pinned_support(ax, x_pos=L_val)

            # 梁的宽度方向
            w = 0.5
            y_width = np.linspace(-w / 2, w / 2, 5)
            X, Y = np.meshgrid(x_vals, y_width)
            Z = np.tile(y_vals, (len(y_width), 1))  # 将挠度值复制为二维

            # 画挠度梁
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

            # 原始未变形梁
            ax.plot(x_vals, np.zeros_like(x_vals), np.zeros_like(x_vals), 'k--', label='origin', linewidth=2)

            # 设置标签
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title("3D Deflection")
            ax.view_init(elev=30, azim=45)
            ax.legend()

            st.pyplot(fig)

            st.subheader("解出的未知量:")
            for var in unknowns:
                st.write(f"{var} = {solution[var]}")

# ------------------- 教学模式 -------------------

elif st.session_state.mode == '教学模式':
    st.title("🎓 教学模式")

    # ------ Step 0: 重新生成题目按钮 ------
    if st.button("重新生成题目"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    # ------ Step 1: 随机生成边界条件 ------
    if 'boundary_initialized' not in st.session_state:
        supports = ['fixed', 'pinned']
        st.session_state.left_support = random.choice(supports)
        st.session_state.right_support = random.choice(supports)
        st.session_state.boundary_initialized = True

    left_support = st.session_state.left_support
    right_support = st.session_state.right_support

    # ------ Step 2: 随机生成载荷类型 ------
    if 'load_type' not in st.session_state:
        st.session_state.load_type = random.choice(['point', 'moment', 'uniform', 'linear'])

    load_type = st.session_state.load_type

    # 设置常数
    L_val = 10.0
    EI_val = 20000.0

    # ------ Step 3: 随机生成载荷参数 ------
    point_loads, point_moments, uniform_loads, linear_loads = [], [], [], []

    if load_type == 'point':
        if 'P_val' not in st.session_state:
            st.session_state.P_val = random.randint(50, 150)
            st.session_state.a_val = random.uniform(2, 8)
        point_loads.append((st.session_state.P_val, st.session_state.a_val))

    elif load_type == 'moment':
        if 'M0_val' not in st.session_state:
            st.session_state.M0_val = random.randint(30, 100)
            st.session_state.M0_pos = random.uniform(2, 8)
        point_moments.append((st.session_state.M0_val, st.session_state.M0_pos))

    elif load_type == 'uniform':
        if 'q_val' not in st.session_state:
            st.session_state.q_val = random.randint(5, 20)
            st.session_state.x1 = random.uniform(1, 4)
            st.session_state.x2 = st.session_state.x1 + random.uniform(2, 5)
        uniform_loads.append((st.session_state.q_val, st.session_state.x1, st.session_state.x2))

    elif load_type == 'linear':
        if 'q0' not in st.session_state:
            st.session_state.q0 = random.randint(5, 15)
            st.session_state.k = random.uniform(-2, 2)
            st.session_state.x1 = random.uniform(1, 4)
            st.session_state.x2 = st.session_state.x1 + random.uniform(2, 5)
        linear_loads.append((st.session_state.q0, st.session_state.k, st.session_state.x1, st.session_state.x2))

    # ------ Step 4: 构建弯矩表达式 ------
    M_expr = -M1 + F1 * x

    for P, a in point_loads:
        M_expr -= P * sp.Heaviside(x - a) * (x - a)

    for M0, pos in point_moments:
        M_expr -= M0 * sp.Heaviside(x - pos)

    for q, x1, x2 in uniform_loads:
        M_expr -= q * ((x - x1)**2 * sp.Heaviside(x - x1) - (x - x2)**2 * sp.Heaviside(x - x2)) / 2

    for q0, k, x1, x2 in linear_loads:
        x_ = sp.Symbol('x_')
        deflin = lambda x_: q0 + k * (x_ - x1)
        expr = sp.integrate((x - x_) * deflin(x_), (x_, x1, sp.Min(x, x2)))
        M_expr -= sp.Piecewise((expr, x >= x1), (0, True))

    # 积分得到转角和挠度表达式
    theta = sp.integrate(M_expr / EI, x) + C1
    y = sp.integrate(theta, x) + C2

    # ------ Step 5: 生成边界条件 ------
    def generate_boundary_conditions(left, right, y, theta, M_expr):
        conds = []
        if left == 'fixed':
            conds += [sp.Eq(y.subs(x, 0), 0), sp.Eq(theta.subs(x, 0), 0)]
        elif left == 'pinned':
            conds += [sp.Eq(y.subs(x, 0), 0), sp.Eq(M_expr.subs(x, 0), 0)]

        if right == 'fixed':
            conds += [sp.Eq(y.subs(x, L), 0), sp.Eq(theta.subs(x, L), 0)]
        elif right == 'pinned':
            conds += [sp.Eq(y.subs(x, L), 0), sp.Eq(M_expr.subs(x, L), 0)]

        return conds

    param_subs = {L: L_val, EI: EI_val}
    eqs = [eq.subs(param_subs) for eq in generate_boundary_conditions(left_support, right_support, y, theta, M_expr)]

    # ------ Step 6: 解方程求常数 ------
    unknowns = [F1, M1, C1, C2]
    sol = sp.solve(eqs, unknowns, dict=True)

    if not sol:
        st.warning("该配置无法求解，请点击重新生成题目")
    else:
        sol = sol[0]
        M_final = M_expr.subs(param_subs).subs(sol)
        theta_final = theta.subs(param_subs).subs(sol)
        y_final = y.subs(param_subs).subs(sol)

        # ------ Step 7: 数值化并绘图展示 ------
        M_func = sp.lambdify(x, M_final, 'numpy')
        theta_func = sp.lambdify(x, theta_final, 'numpy')
        y_func = sp.lambdify(x, y_final, 'numpy')

        x_vals = np.linspace(0, L_val, 400)
        M_vals = M_func(x_vals)
        theta_vals = theta_func(x_vals)
        y_vals = y_func(x_vals)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        axs[0].plot(x_vals, M_vals, label="Bending Moment M(x)")
        axs[0].set_ylabel("Moment")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(x_vals, theta_vals, label="Rotation θ(x)", color='orange')
        axs[1].set_ylabel("Rotation")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(x_vals, y_vals, label="Deflection y(x)", color='green')
        axs[2].set_ylabel("Deflection")
        axs[2].set_xlabel("x")
        axs[2].legend()
        axs[2].grid(True)

        st.pyplot(fig)

        # ------ Step 8: 选择题交互 ------
        st.subheader("问题：你能判断出以下哪些是当前梁的特性？")

        support_option = st.selectbox("请选择该梁的支撑类型", [
            "fixed-fixed", "fixed-pinned", "pinned-fixed", "pinned-pinned"
        ])
        load_option = st.selectbox("请选择该梁的载荷类型", [
            "集中力", "集中偶矩", "均匀分布力", "线性分布力"
        ])

        # ------ Step 9: 显示正确答案并反馈 ------
        if st.button("显示正确答案"):
            correct_support = f"{left_support}-{right_support}"
            correct_load = {
                'point': '集中力',
                'moment': '集中偶矩',
                'uniform': '均匀分布力',
                'linear': '线性分布力'
            }[load_type]

            st.success(f"✅ 正确边界条件: 左 = `{left_support}`, 右 = `{right_support}`")
            st.success(f"✅ 正确载荷类型: `{correct_load}`")

            if support_option == correct_support and load_option == correct_load:
                st.balloons()
                st.success("🎉 回答正确！")
            else:
                st.warning("❌ 回答不正确，再试试？")
