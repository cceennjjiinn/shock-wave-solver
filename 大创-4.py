import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import least_squares
from io import BytesIO
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# --------------------------
# 核心优化：页面跳转回调函数
# --------------------------
def set_page(page_name):
    """设置页面状态的回调函数，确保状态立即更新"""
    st.session_state.page = page_name
    # 强制页面重渲染
    st.rerun()  # 使用st.rerun()替代experimental版本，适用于较新版本Streamlit


# --------------------------
# 初始化会话状态
# --------------------------
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'material_fits' not in st.session_state:
        st.session_state.material_fits = {}
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'fitted_parameters' not in st.session_state:
        st.session_state.fitted_parameters = None


# --------------------------
# 数据库相关函数
# --------------------------
@st.cache_resource
def create_sql_engine():
    """创建数据库引擎"""
    db_path = os.path.join(os.path.dirname(__file__), "shock_wave_data.db")
    engine = create_engine(f"sqlite:///{db_path}")
    return engine

def init_database(engine):
    """初始化数据库表结构"""
    with engine.connect() as conn:
        # 创建材料表
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            density REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        
        # 创建冲击波数据表
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS shock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER NOT NULL,
            shock_velocity REAL NOT NULL,
            particle_velocity REAL NOT NULL,
            pressure REAL,
            temperature REAL,
            source TEXT,
            FOREIGN KEY (material_id) REFERENCES materials (id)
        )
        """))
        
        # 插入示例数据（如果不存在）
        if not pd.read_sql(text("SELECT * FROM materials"), conn).shape[0]:
            materials = [
                ("铜", "工业纯铜，密度8.96g/cm³", 8.96),
                ("铝", "工业纯铝，密度2.70g/cm³", 2.70),
                ("铁", "工业纯铁，密度7.87g/cm³", 7.87)
            ]
            
            for name, desc, density in materials:
                conn.execute(
                    text("INSERT INTO materials (name, description, density) VALUES (:name, :desc, :density)"),
                    {"name": name, "desc": desc, "density": density}
                )
            
            # 铜的示例数据 (Us, Up)
            copper_id = conn.execute(text("SELECT id FROM materials WHERE name = '铜'")).scalar()
            copper_data = [
                (copper_id, 3.94, 0.50),
                (copper_id, 4.50, 0.80),
                (copper_id, 5.00, 1.05),
                (copper_id, 5.50, 1.28),
                (copper_id, 6.00, 1.50),
                (copper_id, 6.50, 1.71),
                (copper_id, 7.00, 1.91),
                (copper_id, 7.50, 2.10),
                (copper_id, 8.00, 2.28)
            ]
            
            for mid, us, up in copper_data:
                p = 8.96 * us * up  # 计算压力
                conn.execute(
                    text("INSERT INTO shock_data (material_id, shock_velocity, particle_velocity, pressure) VALUES (:mid, :us, :up, :p)"),
                    {"mid": mid, "us": us, "up": up, "p": p}
                )
        
        conn.commit()
    st.session_state.db_initialized = True

def get_all_materials(engine):
    """获取所有材料列表"""
    with engine.connect() as conn:
        materials = pd.read_sql(text("SELECT * FROM materials"), conn)
    return materials

def get_material_data(engine, material_id):
    """获取特定材料的冲击波数据"""
    with engine.connect() as conn:
        data = pd.read_sql(
            text("SELECT * FROM shock_data WHERE material_id = :id"),
            conn,
            params={"id": material_id}
        )
    return data

def add_material_data(engine, material_name, density, shock_data):
    """添加新材料及数据"""
    with engine.connect() as conn:
        # 检查材料是否已存在
        result = conn.execute(
            text("SELECT id FROM materials WHERE name = :name"),
            {"name": material_name}
        )
        material_id = result.scalar()
        
        if not material_id:
            # 插入新材料
            result = conn.execute(
                text("INSERT INTO materials (name, density) VALUES (:name, :density) RETURNING id"),
                {"name": material_name, "density": density}
            )
            material_id = result.scalar()
        
        # 插入冲击波数据
        for us, up in shock_data:
            p = density * us * up  # 计算压力
            conn.execute(
                text("INSERT INTO shock_data (material_id, shock_velocity, particle_velocity, pressure) VALUES (:mid, :us, :up, :p)"),
                {"mid": material_id, "us": us, "up": up, "p": p}
            )
        
        conn.commit()
    return material_id


# --------------------------
# 冲击波参数计算函数
# --------------------------
def fit_hugoniot(us_data, up_data):
    """拟合Hugoniot关系 Us = C0 + s*Up"""
    model = LinearRegression()
    up_data = np.array(up_data).reshape(-1, 1)
    model.fit(up_data, us_data)
    
    c0 = model.intercept_  # 截距，C0
    s = model.coef_[0]     # 斜率，s
    r2 = r2_score(us_data, model.predict(up_data))  # 决定系数
    
    return {
        'c0': c0,
        's': s,
        'r2': r2,
        'model': model
    }

def calculate_shock_parameters(rho0, us, up):
    """根据Rankine-Hugoniot方程计算冲击波参数"""
    # 密度压缩比
    rho_ratio = us / (us - up)
    rho = rho0 * rho_ratio
    
    # 压力
    pressure = rho0 * us * up
    
    # 比内能变化
    delta_e = 0.5 * up**2 * (1 - (rho0/rho)**2)
    
    return {
        'rho0': rho0,
        'rho': rho,
        'us': us,
        'up': up,
        'pressure': pressure,
        'delta_e': delta_e,
        'rho_ratio': rho_ratio
    }

def solve_hugoniot(rho0, c0, s, p_known=None, us_known=None, up_known=None):
    """已知一个参数，求解其他冲击波参数"""
    if sum(1 for x in [p_known, us_known, up_known] if x is not None) != 1:
        raise ValueError("必须且只能提供一个已知参数：p_known, us_known 或 up_known")
    
    # 根据已知参数计算其他参数
    if up_known is not None:
        us = c0 + s * up_known
        p = rho0 * us * up_known
    
    elif us_known is not None:
        if s == 0:
            raise ValueError("s不能为0")
        up = (us_known - c0) / s if abs(s) > 1e-9 else 0
        p = rho0 * us_known * up
    
    elif p_known is not None:
        # 定义残差函数，用于最小二乘求解
        def residuals(x):
            up = x[0]
            us = c0 + s * up
            return [rho0 * us * up - p_known]
        
        # 初始猜测值
        x0 = [0.1]  # 初始粒子速度猜测
        result = least_squares(residuals, x0)
        up = result.x[0]
        us = c0 + s * up
    
    return calculate_shock_parameters(rho0, us, up)


# --------------------------
# 绘图函数
# --------------------------
def plot_hugoniot(us_data, up_data, fit_params, material_name):
    """绘制Hugoniot关系图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点
    ax.scatter(up_data, us_data, color='blue', label='实验数据')
    
    # 绘制拟合线
    up_range = np.linspace(min(up_data) - 0.1, max(up_data) + 0.1, 100)
    us_fit = fit_params['c0'] + fit_params['s'] * up_range
    ax.plot(up_range, us_fit, 'r--', label=f'拟合线: Us = {fit_params["c0"]:.2f} + {fit_params["s"]:.2f}·Up')
    
    ax.set_xlabel('粒子速度 Up (km/s)')
    ax.set_ylabel('冲击波速度 Us (km/s)')
    ax.set_title(f'{material_name}的Hugoniot关系')
    ax.grid(True)
    ax.legend()
    
    # 保存到 BytesIO
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    
    return buf

def plot_pressure_vs_velocity(params_list, material_name):
    """绘制压力与速度关系图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取数据
    us_values = [p['us'] for p in params_list]
    up_values = [p['up'] for p in params_list]
    p_values = [p['pressure'] for p in params_list]
    
    # 绘制压力-冲击波速度关系
    ax.plot(us_values, p_values, 'bo-', label='压力-冲击波速度')
    
    ax.set_xlabel('冲击波速度 Us (km/s)')
    ax.set_ylabel('压力 P (GPa)')
    ax.set_title(f'{material_name}的压力与冲击波速度关系')
    ax.grid(True)
    ax.legend()
    
    # 保存到 BytesIO
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    
    return buf


# --------------------------
# 页面函数
# --------------------------
def home_page():
    st.title("冲击波参数计算与分析系统")
    st.write("""
    本系统用于分析冲击波在材料中传播的特性参数，基于Rankine-Hugoniot守恒方程进行计算。
    
    ### 核心功能
    - 基于Hugoniot关系（Us = C0 + s·Up）拟合材料参数
    - 计算冲击波压力、密度变化、粒子速度等关键参数
    - 可视化冲击波特性曲线
    - 支持数据库存储和管理多种材料的冲击波数据
    """)
    
    st.write("请选择操作模式：")
    col1, col2 = st.columns(2)
    
    with col1:
        st.button(
            "使用数据库数据",
            on_click=set_page,
            args=("database_mode",),
            key="db_mode_btn",
            use_container_width=True
        )
    
    with col2:
        st.button(
            "手动输入参数",
            on_click=set_page,
            args=("manual_mode",),
            key="manual_mode_btn",
            use_container_width=True
        )


def database_mode_page(engine):
    st.title("数据库模式")
    st.write("从数据库加载材料数据，基于Hugoniot关系拟合参数并求解冲击波特性")
    
    # 获取所有材料
    materials = get_all_materials(engine)
    
    if materials.empty:
        st.error("数据库中没有可用材料，请先添加材料数据")
        st.button(
            "返回主页",
            on_click=set_page,
            args=("home",),
            key="db_empty_back_btn"
        )
        return
    
    # 选择材料
    material_names = materials["name"].tolist()
    selected_material = st.selectbox("选择材料", material_names)
    
    # 获取所选材料信息
    material_info = materials[materials["name"] == selected_material].iloc[0]
    material_id = material_info["id"]
    rho0 = material_info["density"]
    
    st.write(f"**材料密度**: {rho0} g/cm³")
    
    # 获取材料数据
    material_data = get_material_data(engine, material_id)
    
    if material_data.empty:
        st.warning("所选材料没有可用的冲击波数据")
    else:
        st.subheader("冲击波数据")
        st.dataframe(material_data[["shock_velocity", "particle_velocity", "pressure"]], 
                    column_config={
                        "shock_velocity": "冲击波速度 Us (km/s)",
                        "particle_velocity": "粒子速度 Up (km/s)",
                        "pressure": "压力 P (GPa)"
                    })
        
        # 拟合Hugoniot关系
        us_data = material_data["shock_velocity"].values
        up_data = material_data["particle_velocity"].values
        fit_params = fit_hugoniot(us_data, up_data)
        
        st.subheader("Hugoniot关系拟合结果")
        st.write(f"拟合公式: Us = C0 + s·Up")
        st.write(f"C0 = {fit_params['c0']:.4f} km/s")
        st.write(f"s = {fit_params['s']:.4f}")
        st.write(f"决定系数 R² = {fit_params['r2']:.4f}")
        
        # 保存拟合参数到会话状态
        st.session_state.fitted_parameters = fit_params
        
        # 显示Hugoniot关系图
        hugoniot_plot = plot_hugoniot(us_data, up_data, fit_params, selected_material)
        st.image(hugoniot_plot, caption=f"{selected_material}的Hugoniot关系图")
        
        # 参数计算区域
        st.subheader("冲击波参数计算")
        st.write("输入已知参数，计算其他冲击波特性参数")
        
        calc_option = st.radio("选择已知参数类型", 
                             ["冲击波速度 (Us)", "粒子速度 (Up)", "压力 (P)"])
        
        if calc_option == "冲击波速度 (Us)":
            us_input = st.number_input("输入冲击波速度 Us (km/s)", 
                                     min_value=0.01, value=5.0, step=0.1)
            if st.button("计算", key="calc_us_btn"):
                try:
                    result = solve_hugoniot(rho0, fit_params['c0'], fit_params['s'], us_known=us_input)
                    st.session_state.current_results = result
                except Exception as e:
                    st.error(f"计算出错: {str(e)}")
        
        elif calc_option == "粒子速度 (Up)":
            up_input = st.number_input("输入粒子速度 Up (km/s)", 
                                     min_value=0.01, value=1.0, step=0.1)
            if st.button("计算", key="calc_up_btn"):
                try:
                    result = solve_hugoniot(rho0, fit_params['c0'], fit_params['s'], up_known=up_input)
                    st.session_state.current_results = result
                except Exception as e:
                    st.error(f"计算出错: {str(e)}")
        
        elif calc_option == "压力 (P)":
            p_input = st.number_input("输入压力 P (GPa)", 
                                    min_value=0.01, value=50.0, step=1.0)
            if st.button("计算", key="calc_p_btn"):
                try:
                    result = solve_hugoniot(rho0, fit_params['c0'], fit_params['s'], p_known=p_input)
                    st.session_state.current_results = result
                except Exception as e:
                    st.error(f"计算出错: {str(e)}")
        
        # 显示计算结果
        if st.session_state.current_results is not None:
            res = st.session_state.current_results
            st.subheader("计算结果")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"冲击波速度 Us: {res['us']:.4f} km/s")
                st.write(f"粒子速度 Up: {res['up']:.4f} km/s")
                st.write(f"初始密度 rho0: {res['rho0']:.4f} g/cm³")
            with col2:
                st.write(f"冲击波压力 P: {res['pressure']:.4f} GPa")
                st.write(f"压缩后密度 rho: {res['rho']:.4f} g/cm³")
                st.write(f"密度压缩比: {res['rho_ratio']:.4f}")
            st.write(f"比内能变化 delta_e: {res['delta_e']:.4f} (km/s)²")
    
    # 添加新材料数据
    with st.expander("添加新材料数据"):
        new_material_name = st.text_input("新材料名称")
        new_density = st.number_input("材料密度 (g/cm³)", min_value=0.01, value=1.0)
        
        st.write("输入冲击波数据 (Us, Up)，每行一对数值，用逗号分隔")
        data_text = st.text_area("数据输入", "5.0, 1.0\n6.0, 1.5\n7.0, 2.0")
        
        if st.button("添加数据", key="add_data_btn"):
            if not new_material_name:
                st.error("请输入材料名称")
            else:
                try:
                    # 解析输入数据
                    shock_data = []
                    lines = [line.strip() for line in data_text.split('\n') if line.strip()]
                    for line in lines:
                        us, up = map(float, line.split(','))
                        shock_data.append((us, up))
                    
                    # 添加到数据库
                    material_id = add_material_data(engine, new_material_name, new_density, shock_data)
                    st.success(f"成功添加材料 '{new_material_name}' 及其数据")
                    st.rerun()  # 刷新页面以显示新添加的材料
                except Exception as e:
                    st.error(f"添加数据出错: {str(e)}")
    
    # 返回主页按钮
    st.button(
        "返回主页",
        on_click=set_page,
        args=("home",),
        key="db_back_btn"
    )


def manual_mode_page(engine):
    st.title("手动输入模式")
    st.write("手动输入参数计算冲击波特性，适用于无数据库数据的场景")
    
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input("材料名称", value="铜")
        rho0 = st.number_input("材料初始密度 (g/cm³)", min_value=0.01, value=8.96)
        
        st.subheader("Hugoniot参数")
        c0 = st.number_input("C0 (km/s)", min_value=0.01, value=3.93)
        s = st.number_input("s", min_value=0.01, value=1.49)
    
    with col2:
        st.subheader("已知参数 (输入其中一项)")
        input_type = st.radio("选择输入类型", 
                            ["冲击波速度 (Us)", "粒子速度 (Up)", "压力 (P)"],
                            key="input_type_radio")
        
        if input_type == "冲击波速度 (Us)":
            input_value = st.number_input("输入冲击波速度 Us (km/s)", 
                                        min_value=0.01, value=5.0)
        elif input_type == "粒子速度 (Up)":
            input_value = st.number_input("输入粒子速度 Up (km/s)", 
                                        min_value=0.01, value=1.0)
        else:  # 压力
            input_value = st.number_input("输入压力 P (GPa)", 
                                        min_value=0.01, value=50.0)
    
    # 计算按钮
    if st.button("计算冲击波参数", key="manual_calc_btn"):
        try:
            # 根据输入类型计算
            if input_type == "冲击波速度 (Us)":
                result = solve_hugoniot(rho0, c0, s, us_known=input_value)
            elif input_type == "粒子速度 (Up)":
                result = solve_hugoniot(rho0, c0, s, up_known=input_value)
            else:  # 压力
                result = solve_hugoniot(rho0, c0, s, p_known=input_value)
            
            st.session_state.current_results = result
            
            # 显示结果
            st.subheader("计算结果")
            res = result
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"冲击波速度 Us: {res['us']:.4f} km/s")
                st.write(f"粒子速度 Up: {res['up']:.4f} km/s")
                st.write(f"初始密度 rho0: {res['rho0']:.4f} g/cm³")
            with col2:
                st.write(f"冲击波压力 P: {res['pressure']:.4f} GPa")
                st.write(f"压缩后密度 rho: {res['rho']:.4f} g/cm³")
                st.write(f"密度压缩比: {res['rho_ratio']:.4f}")
            st.write(f"比内能变化 delta_e: {res['delta_e']:.4f} (km/s)²")
            
        except Exception as e:
            st.error(f"计算出错: {str(e)}")
    
    # 保存到数据库选项
    if st.session_state.current_results is not None:
        with st.expander("保存到数据库"):
            st.write("将当前计算结果和参数保存到数据库")
            if st.button("保存", key="save_to_db_btn"):
                try:
                    res = st.session_state.current_results
                    # 准备要保存的数据点
                    shock_data = [(res['us'], res['up'])]
                    # 添加到数据库
                    material_id = add_material_data(engine, material_name, rho0, shock_data)
                    st.success(f"成功将数据保存到数据库中的 '{material_name}'")
                except Exception as e:
                    st.error(f"保存失败: {str(e)}")
    
    # 返回主页按钮
    st.button(
        "返回主页",
        on_click=set_page,
        args=("home",),
        key="manual_back_btn"
    )


# --------------------------
# 主函数
# --------------------------
def main():
    init_session_state()
    st.set_page_config(
        page_title="冲击波参数计算与分析系统",
        page_icon="⚡",
        layout="wide"
    )
    
    engine = create_sql_engine()
    if not st.session_state.db_initialized:
        init_database(engine)
    
    # 根据会话状态显示相应页面
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "database_mode":
        database_mode_page(engine)
    elif st.session_state.page == "manual_mode":
        manual_mode_page(engine)


if __name__ == "__main__":
    main()
