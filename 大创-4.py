import streamlit as st
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sympy import symbols, Eq, solve
from io import BytesIO
from PIL import Image
import itertools
import os

# 设置中文字体
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建SQLite引擎
sqlite_path = os.path.abspath('shock_wave_data.db')
sqlite_engine = create_engine(f'sqlite:///{sqlite_path}')

# SQLite性能优化
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # 写前日志
    cursor.execute('PRAGMA synchronous=NORMAL;')  # 同步模式
    cursor.execute('PRAGMA temp_store=MEMORY;')   # 临时存储
    cursor.close()

# 初始化数据库 - 表名修改为shock_wave_all_data
def init_database():
    try:
        with sqlite_engine.connect() as conn:
            if not conn.dialect.has_table(conn, 'shock_wave_all_data'):
                conn.execute(text("""
                    CREATE TABLE shock_wave_all_data (
                        id INTEGER PRIMARY KEY,
                        material TEXT,
                        rho0 REAL,       -- 初始密度 (g/cm³)
                        Us REAL,         -- 冲击波速度 (km/s)
                        Up REAL,         -- 粒子速度 (km/s)
                        P REAL,          -- 冲击压力 (GPa)
                        V REAL,          -- 比容 (cm³/g)
                        rho REAL,        -- 压缩后密度 (g/cm³)
                        V_V0 REAL,       -- 比容比 (V/V0)
                        exp_method TEXT, -- 实验方法/数据来源
                        gamma REAL,      -- Grüneisen系数
                        T REAL           -- 冲击温度 (K)
                    )
                """))
                conn.commit()
    except Exception as e:
        st.error(f"数据库初始化失败: {str(e)}")

init_database()

# 数据库操作函数 - 统一使用shock_wave_all_data表
def get_all_materials():
    try:
        query = text("SELECT DISTINCT material FROM shock_wave_all_data")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['material'].tolist()
    except Exception as e:
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material_name):
    try:
        query = text("SELECT * FROM shock_wave_all_data WHERE material = :material")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'material': material_name})
        return df
    except Exception as e:
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    if not results:
        st.warning("没有数据可保存")
        return
        
    try:
        with sqlite_engine.begin() as conn:
            for result in results:
                data = {
                    'material': material_name,
                    'rho0': result.get('rh0f', 0),
                    'Us': result.get('Df', 0),
                    'Up': result.get('uf', 0),
                    'P': result.get('Pf', 0),
                    'V': result.get('V', 0),
                    'rho': result.get('rhf', 0),
                    'V_V0': result.get('V_V0', 0),
                    'exp_method': 'calculated',
                    'gamma': result.get('gamma', 0),
                    'T': result.get('T', 0)
                }
                stmt = text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, data)
        st.success(f"成功保存 {len(results)} 条计算结果到数据库")
    except Exception as e:
        st.error(f"保存失败: {str(e)}")

def save_input_data_to_db(input_data, material_name, exp_method="manual_input"):
    try:
        with sqlite_engine.begin() as conn:
            data = {
                'material': material_name,
                'rho0': input_data.get('rho0', 0),
                'Us': input_data.get('Us', 0),
                'Up': input_data.get('Up', 0),
                'P': input_data.get('P', 0),
                'V': input_data.get('V', 0),
                'rho': input_data.get('rho', 0),
                'V_V0': input_data.get('V_V0', 0),
                'exp_method': exp_method,
                'gamma': input_data.get('gamma', 0),
                'T': input_data.get('T', 0)
            }
            stmt = text("""
                INSERT INTO shock_wave_all_data 
                (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
            """)
            conn.execute(stmt, data)
        st.success(f"成功保存输入数据到数据库 (材料: {material_name})")
    except Exception as e:
        st.error(f"保存输入数据失败: {str(e)}")

# 冲击波参数计算（含温度计算）
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300):
    """
    基于Rankine-Hugoniot守恒关系计算冲击波参数
    新增温度计算：基于Mie-Grüneisen状态方程
    """
    # 动量守恒：P = rho0 * U_s * u_p
    P = rho0 * U_s * u_p
    # 质量守恒推导比容：V = (1/rho0) * (1 - u_p/U_s)
    V = (1 / rho0) * (1 - u_p / U_s)
    # 压缩后密度：rho = rho0 * U_s/(U_s - u_p)
    rho = rho0 * U_s / (U_s - u_p)
    # 比容比：V/V0 = 1 - u_p/U_s
    V_V0 = V * rho0  # 因V0 = 1/rho0，故V/V0 = V * rho0
    
    # 温度计算（Mie-Grüneisen方程近似）
    # 单位转换：1 GPa·cm³/g = 1e5 J/kg
    E_shock = 0.5 * P * (1/rho0 - V) * 1e5  # 冲击内能 (J/kg)
    T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
    
    return P, V, rho, V_V0, T

# Hugoniot关系拟合
def fit_hugoniot(df):
    U_s = df['Us'].values
    u_p = df['Up'].values
    coeffs = np.polyfit(u_p, U_s, 1)
    S = coeffs[0]    # 斜率参数
    C0 = coeffs[1]   # 截距（零压声速）
    return C0, S

def fit_material_data(df, material_name):
    if df is None or df.empty:
        st.warning(f"材料 '{material_name}' 没有数据")
        return None
    
    X = df['Up'].values.reshape(-1, 1)
    y = df['Us'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 拟合参数
    C0 = model.intercept_    # 体积声速 (km/s)
    lambda_val = model.coef_[0]  # Hugoniot参数S
    y_pred = model.predict(X)
    
    # 拟合误差计算
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # 均方根误差
    mae = np.mean(np.abs(y - y_pred))              # 平均绝对误差
    
    st.info(f"{material_name} 拟合结果: Us = {C0:.4f} + {lambda_val:.4f}*Up")
    st.info(f"拟合误差: R² = {r2:.4f}, RMSE = {rmse:.4f} km/s, MAE = {mae:.4f} km/s")
    st.info(f"平均参数: ρ₀ = {df['rho0'].mean():.4f} g/cm³, 平均压力 = {df['P'].mean():.4f} GPa")
    
    return {
        "C0": C0, "lambda": lambda_val, "rho0": df['rho0'].mean(), 
        "r2": r2, "rmse": rmse, "mae": mae
    }

# 误差传递计算
def calculate_error(params, param_errors):
    """计算输出参数的误差（基于误差传递公式）"""
    rho0, Us, Up = params['rho0'], params['Us'], params['Up']
    rho0_err, Us_err, Up_err = param_errors['rho0'], param_errors['Us'], param_errors['Up']
    
    # 压力误差：P = rho0*Us*Up → 相对误差平方和
    P_rel_err = (rho0_err/rho0)**2 + (Us_err/Us)** 2 + (Up_err/Up)**2
    P_err = rho0*Us*Up * np.sqrt(P_rel_err)
    
    # 冲击波速度误差（简化）
    Us_err = np.sqrt(Us_err**2 + (0.01*Us)** 2)  # 新增1%模型误差
    
    return {
        "P_err": P_err,
        "Us_err": Us_err,
        "Up_err": Up_err
    }

# 输入函数
def get_input_streamlit(label, var_name, key, default=None, unit="", desc=""):
    st.caption(f"{desc} | 单位: {unit}")
    input_type = st.radio(
        f"{label} 输入类型",
        ["单个值", "多个值(逗号分隔)", "范围(带可选步长)"],
        key=f"{key}_type",
        horizontal=True
    )
    
    default_val = str(default) if default is not None else ""
    
    if input_type == "单个值":
        val = st.text_input(label, default_val, key=f"{key}_single")
        if val == "":
            return symbols(var_name)
        try:
            return [float(val)]
        except ValueError:
            st.error("请输入有效的数值")
            return None
    elif input_type == "多个值(逗号分隔)":
        val = st.text_input(label, default_val, key=f"{key}_multi")
        if val == "":
            return symbols(var_name)
        try:
            return [float(x.strip()) for x in val.split(',')]
        except ValueError:
            st.error("请输入有效的逗号分隔数值")
            return None
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.text_input(f"{label} 起始值", default_val, key=f"{key}_start")
        with col2:
            end = st.text_input(f"{label} 结束值", "", key=f"{key}_end")
        with col3:
            step = st.text_input(f"{label} 步长(可选)", "0.5", key=f"{key}_step")
            
        if not start or not end:
            return symbols(var_name)
            
        try:
            start = float(start)
            end = float(end)
            step = float(step) if step else 0.5
            values = []
            current = start
            epsilon = 1e-9
            while current <= end + epsilon:
                values.append(round(current, 6))
                current += step
            return values
        except ValueError:
            st.error("请输入有效的范围数值")
            return None

# 冲击波关系图
def generate_shock_plots(df, C0, S):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Us vs Up
    axs[0, 0].scatter(df['Up'], df['Us'], label='实验数据')
    u_p_range = np.linspace(0, df['Up'].max()*1.1, 100)
    U_s_fit = C0 + S * u_p_range
    axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'拟合: Us = {C0:.2f} + {S:.2f}·Up')
    axs[0, 0].set_xlabel('粒子速度 Up (km/s)')
    axs[0, 0].set_ylabel('冲击波速度 Us (km/s)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # P vs Up
    axs[0, 1].scatter(df['Up'], df['P'], label='实验数据')
    rho0 = df['rho0'].iloc[0] if not df.empty else 8.96
    P_range = rho0 * U_s_fit * u_p_range  # P = rho0 * Us * Up
    axs[0, 1].plot(u_p_range, P_range, 'r-', label='理论曲线: P = ρ0·Us·Up')
    axs[0, 1].set_xlabel('粒子速度 Up (km/s)')
    axs[0, 1].set_ylabel('压力 P (GPa)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P vs V/V0
    axs[1, 0].scatter(df['V_V0'], df['P'], label='实验数据')
    V_V0_range = 1 - u_p_range / U_s_fit  # V/V0 = 1 - Up/Us
    axs[1, 0].plot(V_V0_range, P_range, 'r-', label='理论曲线')
    axs[1, 0].set_xlabel('比容比 V/V0')
    axs[1, 0].set_ylabel('压力 P (GPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # rho vs P
    axs[1, 1].scatter(df['P'], df['rho'], label='实验数据')
    rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)  # rho = rho0·Us/(Us-Up)
    axs[1, 1].plot(P_range, rho_range, 'r-', label='理论曲线')
    axs[1, 1].set_xlabel('压力 P (GPa)')
    axs[1, 1].set_ylabel('密度 ρ (g/cm³)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def save_plot_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

# 绘图函数（含温度和误差）
def plot_results_streamlit(results):
    if not results:
        st.warning("没有数据可绘制")
        return
        
    fig = plt.figure(figsize=(18, 9))
    
    # 新增温度相关数据
    tf_values = [r.get('Tf', 0) for r in results]
    tb_values = [r.get('Tb', 0) for r in results]
    ts_values = [r.get('Ts', 0) for r in results]
    
    # 原有数据
    pf_values = [r.get('Pf', 0) for r in results]
    uf_values = [r.get('uf', 0) for r in results]
    df_values = [r.get('Df', 0) for r in results]
    rhf_values = [r.get('rhf', 0) for r in results]
    
    # 1. 压力-粒子速度图（含误差棒）
    ax1 = fig.add_subplot(221)
    ax1.errorbar(uf_values, pf_values, 
                 yerr=[r.get('Pf_err', 0.1) for r in results],  # 压力误差
                 xerr=[r.get('uf_err', 0.05) for r in results], # 粒子速度误差
                 fmt='bo', ecolor='r', capsize=5, label='飞片数据')
    ax1.set_xlabel('粒子速度 Up (km/s)')
    ax1.set_ylabel('冲击压力 P (GPa)')
    ax1.set_title('压力-粒子速度关系（含误差范围）')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 温度-压力图（新增）
    ax2 = fig.add_subplot(222)
    ax2.scatter(pf_values, tf_values, c='orange', label='飞片温度')
    ax2.set_xlabel('冲击压力 P (GPa)')
    ax2.set_ylabel('冲击温度 T (K)')
    ax2.set_title('温度-压力关系')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 冲击波速度-粒子速度图
    ax3 = fig.add_subplot(223)
    ax3.scatter(uf_values, df_values, c='blue', label='飞片')
    ax3.set_xlabel('粒子速度 Up (km/s)')
    ax3.set_ylabel('冲击波速度 Us (km/s)')
    ax3.set_title('冲击波速度-粒子速度关系')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 密度-压力图
    ax4 = fig.add_subplot(224)
    ax4.scatter(pf_values, rhf_values, c='green', label='飞片')
    ax4.set_xlabel('冲击压力 P (GPa)')
    ax4.set_ylabel('压缩后密度 (g/cm³)')
    ax4.set_title('密度-压力关系')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    
    st.image(img, caption="分析结果图表（含温度和误差）")
    
    # 下载按钮
    buf2 = BytesIO()
    plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
    buf2.seek(0)
    st.download_button(
        label="下载图表",
        data=buf2,
        file_name="analysis_with_temp_error.png",
        mime="image/png"
    )
    
    return fig

# 页面函数
def home_page():
    st.title("冲击波参数计算与分析系统")
    st.info("""
    系统核心模型说明：
    1. 基于Rankine-Hugoniot守恒方程（质量、动量、能量守恒）
    2. 假设条件：平面冲击波、稳态传播、忽略初始压力
    3. 单位体系：密度(g/cm³)、速度(km/s)、压力(GPa)
    """)
    st.write("选择操作模式：")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("使用数据库数据"):
            st.session_state.page = "database_mode"
    with col2:
        if st.button("手动输入参数"):
            st.session_state.page = "manual_mode"

def database_mode_page():
    st.title("数据库模式")
    st.write("从数据库加载材料数据，基于Hugoniot关系拟合参数并求解")
    
    materials = get_all_materials()
    if not materials:
        st.error("数据库中没有可用材料数据")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.selectbox("飞片材料", materials, key="flyer_material")
    with col2:
        base_material = st.selectbox("基板材料", materials, key="base_material")
    with col3:
        sample_material = st.selectbox("样品材料", materials, key="sample_material")
    
    flyer_df = get_material_data(flyer_material)
    base_df = get_material_data(base_material)
    sample_df = get_material_data(sample_material)
    
    with st.spinner(f"拟合 {flyer_material} 数据..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material)
    with st.spinner(f"拟合 {base_material} 数据..."):
        base_fit = fit_material_data(base_df, base_material)
    with st.spinner(f"拟合 {sample_material} 数据..."):
        sample_fit = fit_material_data(sample_df, sample_material)
    
    # 冲击波参数分析
    st.subheader("冲击波参数分析（Hugoniot关系）")
    st.caption("""
    分析基于线性Hugoniot关系 Us = C0 + S·Up，其中：
    - C0：材料体积声速（零压下的声速，km/s）
    - S：Hugoniot参数（描述冲击波速度随粒子速度的变化率，无量纲）
    - 适用提示：高压力下（如>100 GPa）可能出现偏差，需考虑相变或非线性项
    """)
    if not flyer_df.empty:
        C0_flyer, S_flyer = fit_hugoniot(flyer_df)
        fig = generate_shock_plots(flyer_df, C0_flyer, S_flyer)
        st.pyplot(fig)
        buf = save_plot_to_bytes(fig)
        st.download_button(
            label="下载冲击波关系图",
            data=buf,
            file_name=f"{flyer_material}_shock_relations.png",
            mime="image/png"
        )
    
    default_params = {"f": flyer_fit, "b": base_fit, "s": sample_fit}
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "nubdaf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "nubdab", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "nubdas", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片参数
    with st.expander(f"{flyer_material} 飞片参数", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "初始密度",
            "rhf": "压缩后密度",
            "Df": "冲击波速度（对应Us）",
            "C0f": "体积声速（Hugoniot拟合）",
            "nubdaf": "Hugoniot参数S（无量纲）",
            "E0f": "初始内能（单位体积）",
            "Ef": "压缩后内能（单位体积）",
            "uf": "粒子速度（对应Up）",
            "w": "飞片初始撞击速度",
            "Pf": "冲击压力",
            "gammaf": "Grüneisen系数",
            "Tf": "冲击温度 (K)"
        }
        var_units = {
            "rh0f": "g/cm³",
            "rhf": "g/cm³",
            "Df": "km/s",
            "C0f": "km/s",
            "nubdaf": "无量纲",
            "E0f": "GPa",
            "Ef": "GPa",
            "uf": "km/s",
            "w": "km/s",
            "Pf": "GPa",
            "gammaf": "无量纲",
            "Tf": "K"
        }
        for i, var in enumerate(variables["f"]):
            with cols[i % 3]:
                default_val = None
                if default_params["f"] and var in ["rh0f", "C0f", "nubdaf"]:
                    if var == "rh0f":
                        default_val = default_params["f"]["rho0"]
                    elif var == "C0f":
                        default_val = default_params["f"]["C0"]
                    elif var == "nubdaf":
                        default_val = default_params["f"]["lambda"]
                elif var == "gammaf":
                    default_val = 2.0  # 默认Grüneisen系数
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"f_{var}",
                    default=default_val,
                    unit=var_units[var],
                    desc=var_descs[var]
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数
    with st.expander(f"{base_material} 基板参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                default_val = None
                if default_params["b"] and var in ["rh0b", "C0b", "nubdab"]:
                    if var == "rh0b":
                        default_val = default_params["b"]["rho0"]
                    elif var == "C0b":
                        default_val = default_params["b"]["C0"]
                    elif var == "nubdab":
                        default_val = default_params["b"]["lambda"]
                elif var == "gammab":
                    default_val = 2.0  # 默认Grüneisen系数
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"b_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Db", "C0b", "ub"] else 
                         "GPa" if var in ["E0b", "Eb", "Pb"] else 
                         "K" if var == "Tb" else "无量纲",
                    desc="初始密度" if var == "rh0b" else
                         "压缩后密度" if var == "rhb" else
                         "冲击波速度" if var == "Db" else
                         "体积声速" if var == "C0b" else
                         "Hugoniot参数" if var == "nubdab" else
                         "初始内能" if var == "E0b" else
                         "压缩后内能" if var == "Eb" else
                         "粒子速度" if var == "ub" else
                         "冲击压力" if var == "Pb" else
                         "Grüneisen系数" if var == "gammab" else
                         "冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数
    with st.expander(f"{sample_material} 样品参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                default_val = None
                if default_params["s"] and var in ["rh0s", "C0s", "nubdas"]:
                    if var == "rh0s":
                        default_val = default_params["s"]["rho0"]
                    elif var == "C0s":
                        default_val = default_params["s"]["C0"]
                    elif var == "nubdas":
                        default_val = default_params["s"]["lambda"]
                elif var == "gammas":
                    default_val = 2.0  # 默认Grüneisen系数
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"s_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Ds", "C0s", "us"] else 
                         "GPa" if var in ["E0s", "Es", "Ps"] else 
                         "K" if var == "Ts" else "无量纲",
                    desc="初始密度" if var == "rh0s" else
                         "压缩后密度" if var == "rhs" else
                         "冲击波速度" if var == "Ds" else
                         "体积声速" if var == "C0s" else
                         "Hugoniot参数" if var == "nubdas" else
                         "初始内能" if var == "E0s" else
                         "压缩后内能" if var == "Es" else
                         "粒子速度" if var == "us" else
                         "冲击压力" if var == "Ps" else
                         "Grüneisen系数" if var == "gammas" else
                         "冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    if st.button("开始求解"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} 输入无效，请检查")
        
        if not valid:
            return
            
        range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        results = []
        progress_bar = st.progress(0)
        total = len(list(itertools.product(*[v for v in range_params.values()]))) if range_params else 1
        count = 0
        
        for combo in combinations:
            count += 1
            progress_bar.progress(count / total)
            
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # 方程组
            eqs = [
                # 飞片质量守恒：rho0f·Df = rhf·(Df - uf)
                Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # 基板质量守恒：rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒：Pf = rho0f·Df·(w - uf)
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*(sym_vars['w'] - sym_vars['uf']), 0),
                # 基板动量守恒：Pb = rho0b·Db·ub
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # 飞片能量守恒：Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # 基板能量守恒：Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # 飞片Hugoniot关系：Df = C0f + nubdaf·(w - uf)
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['nubdaf']*(sym_vars['w'] - sym_vars['uf']), 0),
                # 基板Hugoniot关系：Db = C0b + nubdab·ub
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['nubdab']*sym_vars['ub'], 0),
                # 界面压力连续：Pf = Pb
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # 界面粒子速度连续：uf = ub
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            try:
                # 检查样品与基板是否为同种材料
                cond = all([
                    current_subs.get(sym_vars['rh0s'], sym_vars['rh0s']) == current_subs.get(sym_vars['rh0b'], sym_vars['rh0b']),
                    current_subs.get(sym_vars['C0b'], sym_vars['C0b']) == current_subs.get(sym_vars['C0s'], sym_vars['C0s']),
                    current_subs.get(sym_vars['nubdab'], sym_vars['nubdab']) == current_subs.get(sym_vars['nubdas'], sym_vars['nubdas']),
                    current_subs.get(sym_vars['E0b'], sym_vars['E0b']) == current_subs.get(sym_vars['E0s'], sym_vars['E0s'])
                ])
            except TypeError:
                cond = False
                
            if cond:
                # 样品与基板同种材料：参数与基板一致
                eqs += [
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续
                    Eq(sym_vars['ub'] - sym_vars['us'], 0),  # 速度连续
                    Eq(sym_vars['rhb'] - sym_vars['rhs'], 0), # 密度连续
                    Eq(sym_vars['Db'] - sym_vars['Ds'], 0),  # 冲击波速度连续
                    # 样品能量守恒
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # 温度参数连续
                    Eq(sym_vars['Tb'] - sym_vars['Ts'], 0),
                    Eq(sym_vars['gammab'] - sym_vars['gammas'], 0)
                ]
            else:
                # 样品与基板不同材料：单独计算
                eqs += [
                    # 样品质量守恒
                    Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    # 基板-样品界面动量守恒
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    # 样品动量守恒
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    # 样品能量守恒
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # 样品Hugoniot关系
                    Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['nubdas']*sym_vars['us'], 0),
                    # 基板-样品界面Hugoniot关系
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['nubdab']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续
                    Eq(sym_vars['ub'] - sym_vars['us'], 0)   # 速度连续
                ]
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            try:
                solutions = solve(substituted_eqs, remaining_vars, dict=True)
                if solutions:
                    for sol in solutions:
                        record = {str(k): float(v) for k, v in sol.items()}
                        record.update({str(k): float(v) for k, v in current_subs.items()})
                        record['flyer_material'] = flyer_material
                        record['base_material'] = base_material
                        record['sample_material'] = sample_material
                        results.append(record)
            except Exception as e:
                st.warning(f"求解错误: {str(e)}（可能因高压力下非线性效应导致，请检查参数范围）")
        
        if results:
            st.success(f"求解完成，共找到 {len(results)} 个解（结果基于理想冲击波假设，实际应用需验证）")
            
            st.subheader("结果数据（单位：rho=g/cm³, D=km/s, u=km/s, P=GPa, T=K）")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载结果数据",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            st.subheader("结果可视化")
            plot_results_streamlit(results)
            
            if st.button("保存结果到数据库"):
                save_results_to_db(results, sample_material)
        else:
            st.warning("未找到有效解（请检查参数是否符合物理范围，如冲击波速度>粒子速度）")
    
    if st.button("返回主页"):
        st.session_state.page = "home"

def manual_mode_page():
    st.title("手动输入模式")
    st.write("手动输入参数求解，适用于无数据库数据的场景")
    
    # 材料参数输入
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input("材料名称", value="Copper", help="输入材料名称，如Copper、Aluminum等")
        gamma = st.number_input("Grüneisen系数 Γ", value=2.0, min_value=0.1, help="铜约2.0，铝约2.13")
    with col2:
        exp_method = st.text_input("实验方法/数据来源", value="manual_input", help="记录数据来源，如实验设备、文献等")
        Cv = st.number_input("定容比热容 Cv (J/(kg·K))", value=385, help="铜约385，铝约900")
    
    # 冲击波参数快速计算
    st.subheader("冲击波参数快速计算")
    st.caption("""
    基于Rankine-Hugoniot守恒方程，适用于理想平面冲击波：
    - 公式：P = ρ0·Us·Up, ρ = ρ0·Us/(Us-Up), V/V0 = 1 - Up/Us
    - 输入要求：Us > Up（冲击波速度必须大于粒子速度）
    - 单位：ρ0(g/cm³), Us(km/s), Up(km/s) → 输出P(GPa)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        U_s = st.number_input("冲击波速度 Us (km/s)", min_value=0.01, value=5.0, help="需大于粒子速度Up")
        Us_err = st.number_input("Us 误差 (km/s)", 0.1, help="测量误差")
    with col2:
        u_p = st.number_input("粒子速度 Up (km/s)", min_value=0.0, value=1.0, help="需小于冲击波速度Us")
        Up_err = st.number_input("Up 误差 (km/s)", 0.05, help="测量误差")
    with col3:
        rho0 = st.number_input("初始密度 ρ0 (g/cm³)", min_value=0.01, value=8.96, help="如铜的初始密度约8.96 g/cm³")
        rho0_err = st.number_input("ρ0 误差 (g/cm³)", 0.02, help="测量误差")
    
    # 存储计算结果用于保存
    calculation_result = None
    
    if st.button("计算冲击波参数"):
        if U_s <= u_p:
            st.error("物理参数错误：冲击波速度Us必须大于粒子速度Up")
        else:
            P, V, rho, V_V0, T = calculate_shock_parameters(
                U_s, u_p, rho0, gamma, Cv
            )
            
            # 计算误差
            error_params = calculate_error(
                params={'rho0': rho0, 'Us': U_s, 'Up': u_p},
                param_errors={'rho0': rho0_err, 'Us': Us_err, 'Up': Up_err}
            )
            
            calculation_result = {
                'rho0': rho0, 'Us': U_s, 'Up': u_p, 
                'P': P, 'V': V, 'rho': rho, 'V_V0': V_V0,
                'gamma': gamma, 'T': T,
                'P_err': error_params['P_err'],
                'Us_err': error_params['Us_err'],
                'Up_err': error_params['Up_err']
            }
            
            st.success(f"""
            计算结果（基于理想冲击波假设）：
            - 冲击压力 P = {P:.2f} ± {error_params['P_err']:.2f} GPa
            - 冲击温度 T = {T:.0f} K
            - 压缩后密度 ρ = {rho:.2f} g/cm³
            - 比容比 V/V0 = {V_V0:.4f}
            """)
    
    # 保存输入数据到数据库
    if calculation_result:
        if st.button("保存输入数据到数据库"):
            save_input_data_to_db(calculation_result, material_name, exp_method)
    
    # 参数输入
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "nubdaf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "nubdab", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "nubdas", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    with st.expander("飞片参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["f"]):
            with cols[i % 3]:
                default_val = 2.0 if var == "gammaf" else None
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=var,
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Df", "C0f", "uf", "w"] else 
                         "GPa" if var in ["E0f", "Ef", "Pf"] else
                         "K" if var == "Tf" else "无量纲",
                    desc="飞片初始密度" if var == "rh0f" else
                         "飞片压缩后密度" if var == "rhf" else
                         "飞片冲击波速度" if var == "Df" else
                         "飞片体积声速" if var == "C0f" else
                         "飞片Hugoniot参数" if var == "nubdaf" else
                         "飞片初始内能" if var == "E0f" else
                         "飞片压缩后内能" if var == "Ef" else
                         "飞片粒子速度" if var == "uf" else
                         "飞片初始撞击速度" if var == "w" else
                         "飞片冲击压力" if var == "Pf" else
                         "飞片Grüneisen系数" if var == "gammaf" else
                         "飞片冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数输入
    with st.expander("基板参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                default_val = 2.0 if var == "gammab" else None
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"b_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Db", "C0b", "ub"] else 
                         "GPa" if var in ["E0b", "Eb", "Pb"] else
                         "K" if var == "Tb" else "无量纲",
                    desc="基板初始密度" if var == "rh0b" else
                         "基板压缩后密度" if var == "rhb" else
                         "基板冲击波速度" if var == "Db" else
                         "基板体积声速" if var == "C0b" else
                         "基板Hugoniot参数" if var == "nubdab" else
                         "基板初始内能" if var == "E0b" else
                         "基板压缩后内能" if var == "Eb" else
                         "基板粒子速度" if var == "ub" else
                         "基板冲击压力" if var == "Pb" else
                         "基板Grüneisen系数" if var == "gammab" else
                         "基板冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数输入
    with st.expander("样品参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                default_val = 2.0 if var == "gammas" else None
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"s_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Ds", "C0s", "us"] else 
                         "GPa" if var in ["E0s", "Es", "Ps"] else
                         "K" if var == "Ts" else "无量纲",
                    desc="样品初始密度" if var == "rh0s" else
                         "样品压缩后密度" if var == "rhs" else
                         "样品冲击波速度" if var == "Ds" else
                         "样品体积声速" if var == "C0s" else
                         "样品Hugoniot参数" if var == "nubdas" else
                         "样品初始内能" if var == "E0s" else
                         "样品压缩后内能" if var == "Es" else
                         "样品粒子速度" if var == "us" else
                         "样品冲击压力" if var == "Ps" else
                         "样品Grüneisen系数" if var == "gammas" else
                         "样品冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    if st.button("开始求解方程组"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} 输入无效，请检查")
        
        if not valid:
            return
            
        range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        results = []
        progress_bar = st.progress(0)
        total = len(list(itertools.product(*[v for v in range_params.values()]))) if range_params else 1
        count = 0
        
        for combo in combinations:
            count += 1
            progress_bar.progress(count / total)
            
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # 检查物理合理性
            try:
                if current_subs.get(sym_vars['Df'], 0) <= current_subs.get(sym_vars['uf'], 0):
                    st.warning("飞片参数错误：Df（冲击波速度）必须大于uf（粒子速度）")
                    continue
                if current_subs.get(sym_vars['Db'], 0) <= current_subs.get(sym_vars['ub'], 0):
                    st.warning("基板参数错误：Db（冲击波速度）必须大于ub（粒子速度）")
                    continue
            except:
                pass
            
            # 方程组定义
            eqs = [
                Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*(sym_vars['w'] - sym_vars['uf']), 0),
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['nubdaf']*(sym_vars['w'] - sym_vars['uf']), 0),
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['nubdab']*sym_vars['ub'], 0),
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            try:
                # 检查样品与基板是否为同种材料
                cond = all([
                    current_subs.get(sym_vars['rh0s'], sym_vars['rh0s']) == current_subs.get(sym_vars['rh0b'], sym_vars['rh0b']),
                    current_subs.get(sym_vars['C0b'], sym_vars['C0b']) == current_subs.get(sym_vars['C0s'], sym_vars['C0s']),
                    current_subs.get(sym_vars['nubdab'], sym_vars['nubdab']) == current_subs.get(sym_vars['nubdas'], sym_vars['nubdas']),
                    current_subs.get(sym_vars['E0b'], sym_vars['E0b']) == current_subs.get(sym_vars['E0s'], sym_vars['E0s'])
                ])
            except TypeError:
                cond = False
                
            if cond:
                eqs += [
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),
                    Eq(sym_vars['ub'] - sym_vars['us'], 0),
                    Eq(sym_vars['rhb'] - sym_vars['rhs'], 0),
                    Eq(sym_vars['Db'] - sym_vars['Ds'], 0),
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    Eq(sym_vars['Tb'] - sym_vars['Ts'], 0),
                    Eq(sym_vars['gammab'] - sym_vars['gammas'], 0)
                ]
            else:
                eqs += [
                    Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['nubdas']*sym_vars['us'], 0),
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['nubdab']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),
                    Eq(sym_vars['ub'] - sym_vars['us'], 0)
                ]
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            try:
                solutions = solve(substituted_eqs, remaining_vars, dict=True)
                if solutions:
                    for sol in solutions:
                        record = {str(k): float(v) for k, v in sol.items()}
                        record.update({str(k): float(v) for k, v in current_subs.items()})
                        results.append(record)
            except Exception as e:
                st.warning(f"求解错误: {str(e)}（可能因参数范围超出模型适用条件）")
        
        if results:
            st.success(f"求解完成，共找到 {len(results)} 个解")
            
            st.subheader("结果数据")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载结果数据",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            st.subheader("结果可视化")
            plot_results_streamlit(results)
            
            if st.button("保存计算结果到数据库"):
                save_results_to_db(results, material_name)
        else:
            st.warning("未找到有效解（请检查参数是否符合物理规律，如冲击波速度>粒子速度）")
    
    if st.button("返回主页"):
        st.session_state.page = "home"

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    st.set_page_config(
        page_title="冲击波参数计算与分析系统",
        page_icon="✨",
        layout="wide"
    )
    
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "database_mode":
        database_mode_page()
    elif st.session_state.page == "manual_mode":
        manual_mode_page()

if __name__ == "__main__":
    main()
