import streamlit as st
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sympy import symbols, Eq, solve
from scipy.optimize import least_squares
from io import BytesIO
from PIL import Image
import itertools
import os

# 设置中文字体（保持界面中的中文显示）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建SQLite引擎 - 优化连接池配置
sqlite_path = os.path.abspath('shock_wave_data.db')
sqlite_engine = create_engine(
    f'sqlite:///{sqlite_path}',
    pool_size=5,          # 保持5个持久连接
    max_overflow=10,      # 最多创建10个额外临时连接
    pool_recycle=3600     # 1小时后回收连接
)

# SQLite性能优化
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # 预写日志
    cursor.execute('PRAGMA synchronous=NORMAL;')  # 同步模式
    cursor.execute('PRAGMA temp_store=MEMORY;')   # 临时存储
    cursor.execute('PRAGMA cache_size=-20000;')   # 增加缓存（20MB）
    cursor.close()

# 初始化数据库 - 添加材料字段索引
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
                        V REAL,          -- 比体积 (cm³/g)
                        rho REAL,        -- 压缩密度 (g/cm³)
                        V_V0 REAL,       -- 比体积比 (V/V0)
                        exp_method TEXT, -- 实验方法/数据来源
                        gamma REAL,      -- 格吕奈森系数
                        T REAL,          -- 冲击温度 (K)
                        INDEX idx_material (material)  -- 新增索引以加快查询
                    )
                """))
                conn.commit()
    except Exception as e:
        st.error(f"数据库初始化失败: {str(e)}")

init_database()

# 数据库操作函数 - 优化查询效率
@st.cache_data(ttl=3600)  # 缓存1小时
def get_all_materials():
    try:
        query = text("SELECT DISTINCT material FROM shock_wave_all_data")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['material'].tolist()
    except Exception as e:
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material_name, fields=None):
    """按需查询字段以减少数据传输"""
    try:
        if fields is None:
            fields = '*'  # 默认查询所有字段
        else:
            fields = ', '.join(fields)  # 按需指定字段
        query = text(f"SELECT {fields} FROM shock_wave_all_data WHERE material = :material")
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
        st.success(f"成功将{len(results)}个计算结果保存到数据库")
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
        st.success(f"成功将输入数据保存到数据库（材料: {material_name}）")
    except Exception as e:
        st.error(f"保存输入数据失败: {str(e)}")

# 冲击波参数计算（包含温度计算）
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300):
    """根据Rankine-Hugoniot守恒关系计算冲击波参数"""
    # 动量守恒: P = rho0 * U_s * u_p
    P = rho0 * U_s * u_p
    
    # 质量守恒推导比体积: V = (1/rho0) * (1 - u_p/U_s)
    V = (1 / rho0) * (1 - u_p / U_s)
    
    # 压缩密度: rho = rho0 * U_s/(U_s - u_p)
    rho = rho0 * U_s / (U_s - u_p)
    
    # 比体积比: V/V0 = 1 - u_p/U_s
    V_V0 = V * rho0  # 由于V0 = 1/rho0，V/V0 = V * rho0
    
    # 温度计算（Mie-Grüneisen方程近似）
    # 单位转换: 1 GPa·cm³/g = 1e5 J/kg
    E_shock = 0.5 * P * (1/rho0 - V) * 1e6  # 冲击内能 (J/kg)
    # 基于Mie-Grüneisen方程简化形式（适用于弱冲击，忽略体积修正项）
    T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
    
    return P, V, rho, V_V0, T

# Hugoniot关系拟合 - 优化数据预处理
def fit_hugoniot(df):
    # 过滤物理上无效的数据
    df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
    if len(df) < 2:
        return 0, 0  # 数据不足时返回默认值
        
    U_s = df['Us'].values
    u_p = df['Up'].values
    coeffs = np.polyfit(u_p, U_s, 1)
    S = coeffs[0]    # 斜率参数
    C0 = coeffs[1]   # 截距（零压声速）
    return C0, S

@st.cache_data(ttl=3600)  # 缓存拟合结果
def fit_material_data(df, material_name, material_type):
    if df is None or df.empty:
        st.warning(f"{material_type}材料'{material_name}'没有数据")
        return None
    
    # 过滤异常值
    df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
    if len(df) < 2:
        st.warning(f"{material_type}材料'{material_name}'有效数据不足，无法拟合")
        return None
    
    X = df['Up'].values.reshape(-1, 1)
    y = df['Us'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 拟合参数
    C0 = model.intercept_    # 体声速 (km/s)
    S = model.coef_[0]       # Hugoniot参数S
    y_pred = model.predict(X)
    
    # 拟合误差计算
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # 均方根误差
    mae = np.mean(np.abs(y - y_pred))              # 平均绝对误差
    
    st.info(f"{material_type}材料 {material_name} 拟合结果: Us = {C0:.4f} + {S:.4f}*Up")
    st.info(f"拟合误差: R² = {r2:.4f}, RMSE = {rmse:.4f} km/s, MAE = {mae:.4f} km/s")
    st.info(f"平均参数: ρ₀ = {df['rho0'].mean():.4f} g/cm³, 平均压力 = {df['P'].mean():.4f} GPa")
    
    return {
        "C0": C0, "S": S, "rho0": df['rho0'].mean(),
        "r2": r2, "rmse": rmse, "mae": mae
    }

# 误差传播计算
def calculate_error(params, param_errors):
    """计算输出参数的误差（基于误差传播公式）"""
    rho0, Us, Up = params['rho0'], params['Us'], params['Up']
    rho0_err, Us_err, Up_err = param_errors['rho0'], param_errors['Us'], param_errors['Up']
    
    # 压力误差: P = rho0*Us*Up → 相对误差平方和
    P_rel_err = (rho0_err/rho0)**2 + (Us_err/Us)** 2 + (Up_err/Up)**2
    P_err = rho0*Us*Up * np.sqrt(P_rel_err)
    
    # 冲击波速度误差（简化）
    Us_err = np.sqrt(Us_err**2 + (0.01*Us)** 2)  # 加入1%模型误差
    
    return {
        "P_err": P_err,
        "Us_err": Us_err,
        "Up_err": Up_err
    }

# 输入函数
def get_input_streamlit(label, var_name, key, default=None, unit="", desc=""):
    st.caption(f"{desc} | 单位: {unit}")
    input_type = st.radio(
        f"{label}输入类型",
        ["单一值", "多个值（逗号分隔）", "范围（可指定步长）"],
        key=f"{key}_type",
        horizontal=True
    )
    
    default_val = str(default) if default is not None else ""
    
    if input_type == "单一值":
        val = st.text_input(label, default_val, key=f"{key}_single")
        if val == "":
            return symbols(var_name)
        try:
            return [float(val)]
        except ValueError:
            st.error("请输入有效的数字")
            return None
    elif input_type == "多个值（逗号分隔）":
        val = st.text_input(label, default_val, key=f"{key}_multi")
        if val == "":
            return symbols(var_name)
        try:
            return [float(x.strip()) for x in val.split(',')]
        except ValueError:
            st.error("请输入有效的逗号分隔数字")
            return None
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.text_input(f"{label}起始值", default_val, key=f"{key}_start")
        with col2:
            end = st.text_input(f"{label}结束值", "", key=f"{key}_end")
        with col3:
            step = st.text_input(f"{label}步长（可选）", "0.5", key=f"{key}_step")
            
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
            st.error("请输入有效的范围数字")
            return None

# 数值求解器（替代符号求解以提高速度）
def solve_numerically(eqs, sym_vars, initial_guess):
    """使用数值方法求解方程组"""
    var_list = list(sym_vars.values())
    
    def residuals(x):
        """计算残差：方程组的误差"""
        substitutions = {var_list[i]: x[i] for i in range(len(x))}
        return [float(abs(eq.subs(substitutions).evalf())) for eq in eqs]
    
    # 执行最小二乘优化，根据物理参数调整边界范围
    result = least_squares(
        residuals,
        list(initial_guess.values()),
        bounds=([
            0.1,   # 密度下界 (g/cm³)
            0.1,   # 密度下界 (g/cm³)
            0.1,   # 速度下界 (km/s)
            0.1,   # 速度下界 (km/s)
            0.1,   # 速度下界 (km/s)
            0.1,   # 速度下界 (km/s)
            0.01,  # 压力下界 (GPa)
            0.01,  # 压力下界 (GPa)
            100    # 温度下界 (K)
        ], [
            20,    # 密度上界 (g/cm³)
            20,    # 密度上界 (g/cm³)
            30,    # 速度上界 (km/s)
            30,    # 速度上界 (km/s)
            30,    # 速度上界 (km/s)
            30,    # 速度上界 (km/s)
            5000,  # 压力上界 (GPa)
            5000,  # 压力上界 (GPa)
            1e5    # 温度上界 (K)
        ]),
        ftol=1e-6,
        max_nfev=1000
    )
    
    if result.success:
        return {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
    return None

# 冲击波关系图绘制 - 修改为在标题中包含材料类型，并支持中英文切换
@st.cache_data(ttl=3600)  # 缓存图像结果
def generate_shock_plots(df, C0, S, material_name, material_type, use_english=False):
    # 数据量大时进行采样
    if len(df) > 1000:
        df = df.sample(1000)
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 根据use_english参数决定使用中文还是英文标题
    if use_english:
        material_type_en = {
            "飞片": "Flyer",
            "基板": "Substrate",
            "样品": "Sample"
        }.get(material_type, material_type)
        fig.suptitle(f'{material_type_en} Material: {material_name} - Shock Wave Relationships', fontsize=16)
    else:
        fig.suptitle(f'{material_type}材料: {material_name} - 冲击波关系图', fontsize=16)
    
    # Us vs Up
    axs[0, 0].scatter(df['Up'], df['Us'], label='实验数据' if not use_english else 'Experimental data')
    u_p_range = np.linspace(0, df['Up'].max()*1.1, 100)
    U_s_fit = C0 + S * u_p_range
    
    if use_english:
        axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'Fit: Us = {C0:.2f} + {S:.2f}·Up')
        axs[0, 0].set_xlabel('Particle velocity Up (km/s)')
        axs[0, 0].set_ylabel('Shock wave velocity Us (km/s)')
    else:
        axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'拟合: Us = {C0:.2f} + {S:.2f}·Up')
        axs[0, 0].set_xlabel('粒子速度 Up (km/s)')
        axs[0, 0].set_ylabel('冲击波速度 Us (km/s)')
        
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # P vs Up
    axs[0, 1].scatter(df['Up'], df['P'], label='实验数据' if not use_english else 'Experimental data')
    # 使用数据中的平均密度而非硬编码值
    rho0 = df['rho0'].mean() if not df.empty else 8.96
    P_range = rho0 * U_s_fit * u_p_range  # P = rho0 * Us * Up
    
    if use_english:
        axs[0, 1].plot(u_p_range, P_range, 'r-', label='Theoretical curve: P = ρ0·Us·Up')
        axs[0, 1].set_xlabel('Particle velocity Up (km/s)')
        axs[0, 1].set_ylabel('Pressure P (GPa)')
    else:
        axs[0, 1].plot(u_p_range, P_range, 'r-', label='理论曲线: P = ρ0·Us·Up')
        axs[0, 1].set_xlabel('粒子速度 Up (km/s)')
        axs[0, 1].set_ylabel('压力 P (GPa)')
        
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P vs V/V0
    axs[1, 0].scatter(df['V_V0'], df['P'], label='实验数据' if not use_english else 'Experimental data')
    V_V0_range = 1 - u_p_range / U_s_fit  # V/V0 = 1 - Up/Us
    
    if use_english:
        axs[1, 0].plot(V_V0_range, P_range, 'r-', label='Theoretical curve')
        axs[1, 0].set_xlabel('Specific volume ratio V/V0')
        axs[1, 0].set_ylabel('Pressure P (GPa)')
    else:
        axs[1, 0].plot(V_V0_range, P_range, 'r-', label='理论曲线')
        axs[1, 0].set_xlabel('比体积比 V/V0')
        axs[1, 0].set_ylabel('压力 P (GPa)')
        
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # rho vs P
    axs[1, 1].scatter(df['P'], df['rho'], label='实验数据' if not use_english else 'Experimental data')
    rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)  # rho = rho0·Us/(Us-Up)
    
    if use_english:
        axs[1, 1].plot(P_range, rho_range, 'r-', label='Theoretical curve')
        axs[1, 1].set_xlabel('Pressure P (GPa)')
        axs[1, 1].set_ylabel('Density ρ (g/cm³)')
    else:
        axs[1, 1].plot(P_range, rho_range, 'r-', label='理论曲线')
        axs[1, 1].set_xlabel('压力 P (GPa)')
        axs[1, 1].set_ylabel('密度 ρ (g/cm³)')
        
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def save_plot_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')  # 降低分辨率以提高速度
    buf.seek(0)
    return buf

# 材料图像显示辅助函数
def display_material_plots(df, material_name, material_type, use_english=False):
    if not df.empty:
        with st.expander(f"查看{material_type}材料{material_name}的冲击波图像", expanded=True):
            C0, S = fit_hugoniot(df)
            # 根据参数决定是否使用英文
            fig = generate_shock_plots(df, C0, S, material_name, material_type, use_english)
            st.pyplot(fig)
            buf = save_plot_to_bytes(fig)
            
            if use_english:
                material_type_en = {
                    "飞片": "flyer",
                    "基板": "substrate",
                    "样品": "sample"
                }.get(material_type, material_type.lower())
                download_label = f"Download {material_type_en} {material_name} shock wave plots"
                file_name = f"{material_type_en}_{material_name}_shock_relations.png"
            else:
                download_label = f"下载{material_type.lower()}{material_name}的冲击波图像"
                file_name = f"{material_type.lower()}_{material_name}_shock_relations.png"
                
            st.download_button(
                label=download_label,
                data=buf,
                file_name=file_name,
                mime="image/png"
            )
    else:
        st.info(f"没有可用数据生成{material_type}材料{material_name}的图像")

# 结果绘图函数
@st.cache_data(ttl=3600)  # 缓存图像结果
def plot_results_streamlit(results):
    if not results:
        return None
        
    # 数据量大时进行采样
    if len(results) > 1000:
        results = results[:1000]
        
    fig = plt.figure(figsize=(18, 9))
    
    # 温度相关数据
    tf_values = [r.get('Tf', 0) for r in results]
    tb_values = [r.get('Tb', 0) for r in results]
    ts_values = [r.get('Ts', 0) for r in results]
    
    # 原始数据
    pf_values = [r.get('Pf', 0) for r in results]
    uf_values = [r.get('uf', 0) for r in results]
    df_values = [r.get('Df', 0) for r in results]
    rhf_values = [r.get('rhf', 0) for r in results]
    
    # 1. 压力-粒子速度图（带误差棒）
    ax1 = fig.add_subplot(221)
    ax1.errorbar(uf_values, pf_values, 
                 yerr=[r.get('Pf_err', 0.1) for r in results],
                 xerr=[r.get('uf_err', 0.05) for r in results],
                 fmt='bo', ecolor='r', capsize=5, label='飞片数据')
    ax1.set_xlabel('粒子速度 Up (km/s)')
    ax1.set_ylabel('冲击压力 P (GPa)')
    ax1.set_title('压力-粒子速度关系（带误差范围）')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 温度-压力图
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
    ax4.set_ylabel('压缩密度 (g/cm³)')
    ax4.set_title('密度-压力关系')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

# 页面函数
def home_page():
    st.title("冲击波参数计算与分析系统")
    st.info("""
    系统核心模型说明：
    1. 基于Rankine-Hugoniot守恒方程组（质量、动量、能量守恒）
    2. 假设条件：平面冲击波、稳态传播、忽略初始压力
    3. 单位系统：密度(g/cm³)、速度(km/s)、压力(GPa)
    """)
    st.write("选择操作模式：")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("使用数据库数据"):
            st.session_state.page = "database_mode"
            st.rerun()  # 立即刷新页面
    with col2:
        if st.button("手动输入参数"):
            st.session_state.page = "manual_mode"
            st.rerun()  # 立即刷新页面

def database_mode_page():
    st.title("数据库模式")
    st.write("从数据库加载材料数据，基于Hugoniot关系拟合参数并求解")
    
    materials = get_all_materials()
    if not materials:
        st.error("数据库中没有可用材料")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.selectbox("飞片材料", materials, key="flyer_material")
    with col2:
        base_material = st.selectbox("基板材料", materials, key="base_material")
    with col3:
        sample_material = st.selectbox("样品材料", materials, key="sample_material")
    
    # 按需查询字段以减少数据传输
    flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho'])
    base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho'])
    sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho'])
    
    # 为每种材料类型拟合数据并清晰标注
    with st.spinner(f"正在拟合飞片材料{flyer_material}的数据..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material, "飞片")
    with st.spinner(f"正在拟合基板材料{base_material}的数据..."):
        base_fit = fit_material_data(base_df, base_material, "基板")
    with st.spinner(f"正在拟合样品材料{sample_material}的数据..."):
        sample_fit = fit_material_data(sample_df, sample_material, "样品")
    
    # 冲击波参数分析部分，为每种材料单独绘图
    st.subheader("冲击波参数分析（Hugoniot关系）")
    st.caption("""
    基于线性Hugoniot关系Us = C0 + S·Up进行分析，其中：
    - C0：材料的体声速（零压状态下的声速，单位km/s）
    - S：Hugoniot参数（描述冲击波速度随粒子速度的变化率，无量纲）
    - 应用说明：在高压下（如>100 GPa）可能出现偏差，需考虑相变或非线性项
    """)
    
    # 为每种材料类型显示单独的图像，数据库模式下设置use_english=True
    display_material_plots(flyer_df, flyer_material, "飞片", use_english=True)
    display_material_plots(base_df, base_material, "基板", use_english=True)
    display_material_plots(sample_df, sample_material, "样品", use_english=True)
    
    default_params = {"f": flyer_fit, "b": base_fit, "s": sample_fit}
    # 参数定义
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片参数
    with st.expander(f"{flyer_material}飞片参数", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "初始密度",
            "rhf": "压缩密度",
            "Df": "冲击波速度（对应Us）",
            "C0f": "体声速（Hugoniot拟合）",
            "Sf": "Hugoniot参数S（无量纲）",
            "E0f": "初始内能密度",
            "Ef": "压缩内能密度",
            "uf": "粒子速度（对应Up）",
            "w": "飞片初始撞击速度",
            "Pf": "冲击压力",
            "gammaf": "格吕奈森系数",
            "Tf": "冲击温度 (K)"
        }
        var_units = {
            "rh0f": "g/cm³",
            "rhf": "g/cm³",
            "Df": "km/s",
            "C0f": "km/s",
            "Sf": "无量纲",
            "E0f": "GPa·cm³/g",
            "Ef": "GPa·cm³/g",
            "uf": "km/s",
            "w": "km/s",
            "Pf": "GPa",
            "gammaf": "无量纲",
            "Tf": "K"
        }
        for i, var in enumerate(variables["f"]):
            with cols[i % 3]:
                default_val = None
                if default_params["f"] and var in ["rh0f", "C0f", "Sf"]:
                    if var == "rh0f":
                        default_val = default_params["f"]["rho0"]
                    elif var == "C0f":
                        default_val = default_params["f"]["C0"]
                    elif var == "Sf":
                        default_val = default_params["f"]["S"]
                elif var == "gammaf":
                    default_val = 2.0  # 默认格吕奈森系数
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
    with st.expander(f"{base_material}基板参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                default_val = None
                if default_params["b"] and var in ["rh0b", "C0b", "Sb"]:
                    if var == "rh0b":
                        default_val = default_params["b"]["rho0"]
                    elif var == "C0b":
                        default_val = default_params["b"]["C0"]
                    elif var == "Sb":
                        default_val = default_params["b"]["S"]
                elif var == "gammab":
                    default_val = 2.0  # 默认格吕奈森系数
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"b_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Db", "C0b", "ub"] else 
                         "GPa·cm³/g" if var in ["E0b", "Eb"] else
                         "GPa" if var == "Pb" else 
                         "K" if var == "Tb" else "无量纲",
                    desc="初始密度" if var == "rh0b" else
                         "压缩密度" if var == "rhb" else
                         "冲击波速度" if var == "Db" else
                         "体声速" if var == "C0b" else
                         "Hugoniot参数" if var == "Sb" else
                         "初始内能密度" if var == "E0b" else
                         "压缩内能密度" if var == "Eb" else
                         "粒子速度" if var == "ub" else
                         "冲击压力" if var == "Pb" else
                         "格吕奈森系数" if var == "gammab" else
                         "冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数
    with st.expander(f"{sample_material}样品参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                default_val = None
                if default_params["s"] and var in ["rh0s", "C0s", "Ss"]:
                    if var == "rh0s":
                        default_val = default_params["s"]["rho0"]
                    elif var == "C0s":
                        default_val = default_params["s"]["C0"]
                    elif var == "Ss":
                        default_val = default_params["s"]["S"]
                elif var == "gammas":
                    default_val = 2.0  # 默认格吕奈森系数
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"s_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Ds", "C0s", "us"] else 
                         "GPa·cm³/g" if var in ["E0s", "Es"] else
                         "GPa" if var == "Ps" else 
                         "K" if var == "Ts" else "无量纲",
                    desc="初始密度" if var == "rh0s" else
                         "压缩密度" if var == "rhs" else
                         "冲击波速度" if var == "Ds" else
                         "体声速" if var == "C0s" else
                         "Hugoniot参数" if var == "Ss" else
                         "初始内能密度" if var == "E0s" else
                         "压缩内能密度" if var == "Es" else
                         "粒子速度" if var == "us" else
                         "冲击压力" if var == "Ps" else
                         "格吕奈森系数" if var == "gammas" else
                         "冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "最大参数组合数量（过多会影响速度）", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("开始求解"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var}输入无效，请检查")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # 截断过多的组合
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"参数组合过多（{len(combinations)}），截断为{max_combinations}以提高速度")
            combinations = combinations[:max_combinations]
        
        results = []
        progress_bar = st.progress(0)
        total = len(combinations)
        count = 0
        
        for combo in combinations:
            count += 1
            # 每10次更新一次进度条以减少UI开销
            if count % 10 == 0 or count == total:
                progress_bar.progress(count / total)
                
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # 方程组
            eqs = [
                # 飞片质量守恒: rho0f·Df = rhf·(Df - uf)
                Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·(w - uf)
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*(sym_vars['w'] - sym_vars['uf']), 0),
                # 基板动量守恒: Pb = rho0b·Db·ub
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # 飞片能量守恒: Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # 基板能量守恒: Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # 飞片Hugoniot关系: Df = C0f + Sf·(w - uf)
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*(sym_vars['w'] - sym_vars['uf']), 0),
                # 基板Hugoniot关系: Db = C0b + Sb·ub
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                # 界面压力连续性: Pf = Pb
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # 界面粒子速度连续性: uf = ub
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            try:
                # 检查样品和基板是否为同一材料
                cond = all([
                    current_subs.get(sym_vars['rh0s'], sym_vars['rh0s']) == current_subs.get(sym_vars['rh0b'], sym_vars['rh0b']),
                    current_subs.get(sym_vars['C0b'], sym_vars['C0b']) == current_subs.get(sym_vars['C0s'], sym_vars['C0s']),
                    current_subs.get(sym_vars['Sb'], sym_vars['Sb']) == current_subs.get(sym_vars['Ss'], sym_vars['Ss']),
                    current_subs.get(sym_vars['E0b'], sym_vars['E0b']) == current_subs.get(sym_vars['E0s'], sym_vars['E0s'])
                ])
            except TypeError:
                cond = False
                
            if cond:
                # 样品与基板为同一材料：参数与基板一致
                eqs += [
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Eq(sym_vars['ub'] - sym_vars['us'], 0),  # 速度连续性
                    Eq(sym_vars['rhb'] - sym_vars['rhs'], 0), # 密度连续性
                    Eq(sym_vars['Db'] - sym_vars['Ds'], 0),  # 冲击波速度连续性
                    # 样品能量守恒
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # 温度参数连续性
                    Eq(sym_vars['Tb'] - sym_vars['Ts'], 0),
                    Eq(sym_vars['gammab'] - sym_vars['gammas'], 0)
                ]
            else:
                # 样品与基板为不同材料：单独计算
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
                    Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                    # 基板-样品界面Hugoniot关系
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Eq(sym_vars['ub'] - sym_vars['us'], 0)   # 速度连续性
                ]
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # 构建初始猜测值（基于物理合理范围）
                initial_guess = {}
                for var in remaining_vars:
                    var_str = str(var)
                    if var_str.startswith(('rh0', 'rh')):  # 密度
                        initial_guess[var] = 8.0
                    elif var_str.startswith(('D', 'C0', 'u', 'w')):  # 速度
                        initial_guess[var] = 5.0
                    elif var_str.startswith(('P', 'E')):  # 压力/能量
                        initial_guess[var] = 100.0
                    elif var_str.startswith('gamma'):  # 格吕奈森系数
                        initial_guess[var] = 2.0
                    elif var_str.startswith('T'):  # 温度
                        initial_guess[var] = 3000.0
                    else:  # 其他参数
                        initial_guess[var] = 1.0
                
                # 使用数值方法求解
                solution = solve_numerically(substituted_eqs, {v:v for v in remaining_vars}, initial_guess)
                
                if solution:
                    record = solution.copy()
                    # 添加已知参数
                    for k, v in current_subs.items():
                        try:
                            record[str(k)] = float(v)
                        except:
                            pass
                    record['flyer_material'] = flyer_material
                    record['base_material'] = base_material
                    record['sample_material'] = sample_material
                    results.append(record)
            except Exception as e:
                st.warning(f"求解错误: {str(e)}（可能由高压下的非线性效应引起，请检查参数范围）")
        
        if results:
            st.success(f"求解完成，找到{len(results)}个解（结果基于理想冲击波假设，实际应用需验证）")
            
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
            fig = plot_results_streamlit(results)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="下载图表",
                    data=buf2,
                    file_name="analysis_with_temp_error.png",
                    mime="image/png"
                )
            
            if st.button("保存结果到数据库"):
                save_results_to_db(results, sample_material)
        else:
            st.warning("未找到有效解（请检查参数是否符合物理范围，如冲击波速度>粒子速度）")
    
    if st.button("返回首页"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

def manual_mode_page():
    st.title("手动输入模式")
    st.write("通过手动输入参数进行求解，适用于没有数据库数据的场景")
    
    # 材料参数输入
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input("材料名称", value="铜", help="输入材料名称，如：铜、铝等")
        gamma = st.number_input("格吕奈森系数Γ", value=2.0, min_value=0.1, help="铜约为2.0，铝约为2.13")
    with col2:
        exp_method = st.text_input("实验方法/数据来源", value="手动输入", help="记录数据来源，如：实验设备、文献等")
        Cv = st.number_input("定容比热容Cv (J/(kg·K))", value=385, help="铜约为385，铝约为900")
    
    # 冲击波参数快速计算
    st.subheader("冲击波参数快速计算")
    st.caption("""
    基于Rankine-Hugoniot守恒方程组，适用于理想平面冲击波：
    - 公式：P = ρ0·Us·Up，ρ = ρ0·Us/(Us-Up)，V/V0 = 1 - Up/Us
    - 输入要求：Us > Up（冲击波速度必须大于粒子速度）
    - 单位：ρ0(g/cm³)，Us(km/s)，Up(km/s) → 输出P(GPa)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        U_s = st.number_input("冲击波速度Us (km/s)", min_value=0.01, value=5.0, help="必须大于粒子速度Up")
        Us_err = st.number_input("Us误差 (km/s)", 0.1, help="测量误差")
    with col2:
        u_p = st.number_input("粒子速度Up (km/s)", min_value=0.0, value=1.0, help="必须小于冲击波速度Us")
        Up_err = st.number_input("Up误差 (km/s)", 0.05, help="测量误差")
    with col3:
        rho0 = st.number_input("初始密度ρ0 (g/cm³)", min_value=0.01, value=8.96, help="例如铜的初始密度约为8.96 g/cm³")
        rho0_err = st.number_input("ρ0误差 (g/cm³)", 0.02, help="测量误差")
    
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
            - 压缩密度 ρ = {rho:.2f} g/cm³
            - 比体积比 V/V0 = {V_V0:.4f}
            """)
    
    # 保存输入数据到数据库
    if calculation_result:
        if st.button("保存输入数据到数据库"):
            save_input_data_to_db(calculation_result, material_name, exp_method)
    
    # 参数输入
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
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
                         "GPa·cm³/g" if var in ["E0f", "Ef"] else
                         "GPa" if var == "Pf" else
                         "K" if var == "Tf" else "无量纲",
                    desc="飞片初始密度" if var == "rh0f" else
                         "飞片压缩密度" if var == "rhf" else
                         "飞片冲击波速度" if var == "Df" else
                         "飞片体声速" if var == "C0f" else
                         "飞片Hugoniot参数S" if var == "Sf" else
                         "飞片初始内能密度" if var == "E0f" else
                         "飞片压缩内能密度" if var == "Ef" else
                         "飞片粒子速度" if var == "uf" else
                         "飞片初始撞击速度" if var == "w" else
                         "飞片冲击压力" if var == "Pf" else
                         "飞片格吕奈森系数" if var == "gammaf" else
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
                         "GPa·cm³/g" if var in ["E0b", "Eb"] else
                         "GPa" if var == "Pb" else
                         "K" if var == "Tb" else "无量纲",
                    desc="基板初始密度" if var == "rh0b" else
                         "基板压缩密度" if var == "rhb" else
                         "基板冲击波速度" if var == "Db" else
                         "基板体声速" if var == "C0b" else
                         "基板Hugoniot参数" if var == "Sb" else
                         "基板初始内能密度" if var == "E0b" else
                         "基板压缩内能密度" if var == "Eb" else
                         "基板粒子速度" if var == "ub" else
                         "基板冲击压力" if var == "Pb" else
                         "基板格吕奈森系数" if var == "gammab" else
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
                         "GPa·cm³/g" if var in ["E0s", "Es"] else
                         "GPa" if var == "Ps" else
                         "K" if var == "Ts" else "无量纲",
                    desc="样品初始密度" if var == "rh0s" else
                         "样品压缩密度" if var == "rhs" else
                         "样品冲击波速度" if var == "Ds" else
                         "样品体声速" if var == "C0s" else
                         "样品Hugoniot参数S" if var == "Ss" else
                         "样品初始内能密度" if var == "E0s" else
                         "样品压缩内能密度" if var == "Es" else
                         "样品粒子速度" if var == "us" else
                         "样品冲击压力" if var == "Ps" else
                         "样品格吕奈森系数" if var == "gammas" else
                         "样品冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "最大参数组合数量（过多会影响速度）", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("开始求解方程组"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var}输入无效，请检查")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # 截断过多的组合
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"参数组合过多（{len(combinations)}），截断为{max_combinations}以提高速度")
            combinations = combinations[:max_combinations]
        
        results = []
        progress_bar = st.progress(0)
        total = len(combinations)
        count = 0
        
        for combo in combinations:
            count += 1
            # 每10次更新进度条
            if count % 10 == 0 or count == total:
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
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*(sym_vars['w'] - sym_vars['uf']), 0),
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            try:
                # 检查样品和基板是否为同一材料
                cond = all([
                    current_subs.get(sym_vars['rh0s'], sym_vars['rh0s']) == current_subs.get(sym_vars['rh0b'], sym_vars['rh0b']),
                    current_subs.get(sym_vars['C0b'], sym_vars['C0b']) == current_subs.get(sym_vars['C0s'], sym_vars['C0s']),
                    current_subs.get(sym_vars['Sb'], sym_vars['Sb']) == current_subs.get(sym_vars['Ss'], sym_vars['Ss']),
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
                    # 样品质量守恒
                    Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    # 基板-样品界面动量守恒
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    # 样品动量守恒
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    # 样品能量守恒
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # 样品Hugoniot关系
                    Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                    # 基板-样品界面Hugoniot关系
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Eq(sym_vars['ub'] - sym_vars['us'], 0)   # 速度连续性
                ]
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # 构建初始猜测值
                initial_guess = {}
                for var in remaining_vars:
                    var_str = str(var)
                    if var_str.startswith(('rh0', 'rh')):
                        initial_guess[var] = 8.0
                    elif var_str.startswith(('D', 'C0', 'u', 'w')):
                        initial_guess[var] = 5.0
                    elif var_str.startswith(('P', 'E')):
                        initial_guess[var] = 100.0
                    elif var_str.startswith('gamma'):
                        initial_guess[var] = 2.0
                    elif var_str.startswith('T'):
                        initial_guess[var] = 3000.0
                    else:
                        initial_guess[var] = 1.0
                
                # 数值求解
                solution = solve_numerically(substituted_eqs, {v:v for v in remaining_vars}, initial_guess)
                
                if solution:
                    record = solution.copy()
                    for k, v in current_subs.items():
                        try:
                            record[str(k)] = float(v)
                        except:
                            pass
                    results.append(record)
            except Exception as e:
                st.warning(f"求解错误: {str(e)}（可能由于参数范围超出模型适用条件）")
        
        if results:
            st.success(f"求解完成，找到{len(results)}个解")
            
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
            fig = plot_results_streamlit(results)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="下载图表",
                    data=buf2,
                    file_name="analysis_with_temp_error.png",
                    mime="image/png"
                )
            
            if st.button("保存计算结果到数据库"):
                save_results_to_db(results, material_name)
        else:
            st.warning("未找到有效解（请检查参数是否符合物理规律，如冲击波速度>粒子速度）")
    
    if st.button("返回首页"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

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
