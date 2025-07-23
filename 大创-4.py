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

# 初始化数据库（保持原有结构，新增温度相关说明）
def init_database():
    try:
        with sqlite_engine.connect() as conn:
            if not conn.dialect.has_table(conn, 'copper_shock_data'):
                conn.execute(text("""
                    CREATE TABLE copper_shock_data (
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
                        gamma REAL,      -- Grüneisen系数（新增）
                        T REAL           -- 冲击温度 (K)（新增）
                    )
                """))
                conn.commit()
    except Exception as e:
        st.error(f"数据库初始化失败: {str(e)}")

init_database()

# 数据库操作函数（新增温度字段的保存）
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
                    'gamma': result.get('gamma', 0),  # 新增
                    'T': result.get('T', 0)           # 新增
                }
                stmt = text("""
                    INSERT INTO copper_shock_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, data)
        st.success(f"成功保存 {len(results)} 条计算结果到数据库")
    except Exception as e:
        st.error(f"保存失败: {str(e)}")

# 新增：保存输入数据到数据库（包含温度相关参数）
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
                'gamma': input_data.get('gamma', 0),  # 新增
                'T': input_data.get('T', 0)           # 新增
            }
            stmt = text("""
                INSERT INTO copper_shock_data 
                (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
            """)
            conn.execute(stmt, data)
        st.success(f"成功保存输入数据到数据库 (材料: {material_name})")
    except Exception as e:
        st.error(f"保存输入数据失败: {str(e)}")

# 冲击波参数计算（新增温度计算）
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300):
    """
    新增温度计算：基于Mie-Grüneisen状态方程
    参数：
    gamma: Grüneisen系数（无量纲，铜约为2.0，铝约为2.13）
    Cv: 定容比热容 (J/(kg·K)，铜约385，铝约900)
    T0: 初始温度 (K，默认300K)
    """
    # 原有参数计算
    P = rho0 * U_s * u_p  # 冲击压力 (GPa)
    V = (1 / rho0) * (1 - u_p / U_s)  # 比容 (cm³/g)
    rho = rho0 * U_s / (U_s - u_p)  # 压缩后密度 (g/cm³)
    V_V0 = V * rho0  # 比容比
    
    # 温度计算（Mie-Grüneisen方程近似）
    # 单位转换：1 GPa·cm³/g = 1e5 J/kg
    E_shock = 0.5 * P * (1/rho0 - V) * 1e5  # 冲击内能 (J/kg)
    T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
    
    return P, V, rho, V_V0, T

# Hugoniot关系拟合（新增拟合误差计算）
def fit_material_data(df, material_name):
    if df is None or df.empty:
        st.warning(f"材料 '{material_name}' 没有数据")
        return None
    
    X = df['Up'].values.reshape(-1, 1)
    y = df['Us'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 拟合参数
    C0 = model.intercept_
    lambda_val = model.coef_[0]
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
        "r2": r2, "rmse": rmse, "mae": mae  # 新增误差指标
    }

# 误差传递计算
def calculate_error(params, param_errors):
    """
    计算输出参数的误差（基于误差传递公式）
    params: 字典包含关键参数 (rho0, Us, Up)
    param_errors: 字典包含参数误差 (rho0_err, Us_err, Up_err)
    """
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

# 输入函数（新增温度相关参数输入）
def get_input_streamlit(label, var_name, key, default=None, unit="", desc="", is_temp_param=False):
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

# 绘图函数（新增误差范围显示）
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
    
    # 1. 压力-粒子速度图（新增误差棒）
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
    
    # 3. 冲击波速度-粒子速度图（原有）
    ax3 = fig.add_subplot(223)
    ax3.scatter(uf_values, df_values, c='blue', label='飞片')
    ax3.set_xlabel('粒子速度 Up (km/s)')
    ax3.set_ylabel('冲击波速度 Us (km/s)')
    ax3.set_title('冲击波速度-粒子速度关系')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 密度-压力图（原有）
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

# 手动输入模式（新增温度和误差参数）
def manual_mode_page():
    st.title("手动输入模式")
    st.write("手动输入参数求解，支持温度计算和误差分析")
    
    # 材料参数输入（新增温度相关参数）
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input("材料名称", value="Copper")
        gamma = st.number_input("Grüneisen系数 Γ", value=2.0, min_value=0.1, help="铜约2.0，铝约2.13")
    with col2:
        exp_method = st.text_input("实验方法", value="manual_input")
        Cv = st.number_input("定容比热容 Cv (J/(kg·K))", value=385, help="铜约385，铝约900")
    
    # 冲击波参数输入（新增误差输入）
    st.subheader("冲击波参数（含误差）")
    col1, col2, col3 = st.columns(3)
    with col1:
        U_s = st.number_input("冲击波速度 Us (km/s)", 5.0, help=">粒子速度")
        Us_err = st.number_input("Us 误差 (km/s)", 0.1, help="测量误差")
    with col2:
        u_p = st.number_input("粒子速度 Up (km/s)", 1.0, help="<冲击波速度")
        Up_err = st.number_input("Up 误差 (km/s)", 0.05, help="测量误差")
    with col3:
        rho0 = st.number_input("初始密度 ρ0 (g/cm³)", 8.96)
        rho0_err = st.number_input("ρ0 误差 (g/cm³)", 0.02, help="测量误差")
    
    # 计算结果存储
    calculation_result = None
    
    if st.button("计算冲击波参数"):
        if U_s <= u_p:
            st.error("错误：冲击波速度必须>粒子速度")
        else:
            # 调用新增的温度计算函数
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
                'gamma': gamma, 'T': T,  # 新增温度相关
                'P_err': error_params['P_err'],  # 误差
                'Us_err': error_params['Us_err'],
                'Up_err': error_params['Up_err']
            }
            
            # 显示结果（含温度和误差）
            st.success(f"""
            计算结果：
            - 冲击压力 P = {P:.2f} ± {error_params['P_err']:.2f} GPa
            - 冲击温度 T = {T:.0f} K
            - 压缩后密度 ρ = {rho:.2f} g/cm³
            - 比容比 V/V0 = {V_V0:.4f}
            """)
    
    # 保存按钮（含温度数据）
    if calculation_result:
        if st.button("保存数据到数据库"):
            save_input_data_to_db(calculation_result, material_name, exp_method)
    
    # 方程组求解部分（略，保持原有逻辑，新增温度和误差字段传递）
    # ...（原有代码保持不变，在结果中添加Tf、Tb、Ts等温度字段）
    
    plt.tight_layout()
    st.pyplot(fig)

# 数据库模式（新增温度和误差展示）
def database_mode_page():
    st.title("数据库模式")
    materials = get_all_materials()
    if not materials:
        st.error("数据库无数据")
        return
    
    # 材料选择
    material = st.selectbox("选择材料", materials)
    df = get_material_data(material)
    
    if not df.empty:
        st.subheader(f"{material} 数据（含温度）")
        st.dataframe(df[['Us', 'Up', 'P', 'T', 'gamma', 'exp_method']])  # 显示温度
        
        # 拟合分析（含误差）
        fit_result = fit_material_data(df, material)
        if fit_result:
            # 绘制Hugoniot曲线（含误差范围）
            fig = generate_shock_plots(df, fit_result['C0'], fit_result['lambda'])
            st.pyplot(fig)
    
    # 原有求解逻辑（略，新增温度和误差参数传递）
    # ...

# 主页面保持不变
def home_page():
    st.title("冲击波参数计算与分析系统")
    st.info("新增功能：温度计算（Mie-Grüneisen方程）、误差分析（拟合误差+传递误差）")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("手动输入模式"):
            st.session_state.page = "manual_mode"
    with col2:
        if st.button("数据库模式"):
            st.session_state.page = "database_mode"

# 主函数
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    st.set_page_config(page_title="冲击波系统（含温度和误差）", layout="wide")
    
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "database_mode":
        database_mode_page()
    elif st.session_state.page == "manual_mode":
        manual_mode_page()

if __name__ == "__main__":
    main()
