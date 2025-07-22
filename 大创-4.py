import streamlit as st
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from sympy import symbols, Eq, solve
from io import BytesIO
from PIL import Image
import itertools
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建SQLite引擎（与导出脚本完全一致）
sqlite_path = os.path.abspath('shock_wave_data.db')  # 获取绝对路径
sqlite_engine = create_engine(f'sqlite:///{sqlite_path}')  # 

# SQLite性能优化（可选）
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # 写前日志
    cursor.execute('PRAGMA synchronous=NORMAL;')  # 同步模式
    cursor.execute('PRAGMA temp_store=MEMORY;')   # 临时存储
    cursor.close()

# 初始化数据库（使用SQLAlchemy替代sqlite3）
def init_database():
    try:
        # 检查表是否存在，不存在则创建
        with sqlite_engine.connect() as conn:
            if not conn.dialect.has_table(conn, 'copper_shock_data'):
                conn.execute(text("""
                    CREATE TABLE copper_shock_data (
                        id INTEGER PRIMARY KEY,
                        material TEXT,
                        rho0 REAL,
                        Us REAL,
                        Up REAL,
                        P REAL,
                        V REAL,
                        rho REAL,
                        V_V0 REAL,
                        exp_method TEXT
                    )
                """))
                conn.commit()
    except Exception as e:
        st.error(f"数据库初始化失败: {str(e)}")

# 初始化数据库
init_database()

# 数据库操作函数（使用SQLAlchemy引擎）
def get_all_materials():
    try:
        # 使用参数化查询防止SQL注入
        query = text("SELECT DISTINCT material FROM copper_shock_data")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['material'].tolist()
    except Exception as e:
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material_name):
    try:
        # 使用参数化查询防止SQL注入
        query = text("SELECT * FROM copper_shock_data WHERE material = :material")
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
        with sqlite_engine.begin() as conn:  # 自动事务管理
            for result in results:
                # 使用命名参数提高可读性
                data = {
                    'material': material_name,
                    'rho0': result.get('rh0f', 0),
                    'Us': result.get('Df', 0),
                    'Up': result.get('uf', 0),
                    'P': result.get('Pf', 0),
                    'V': result.get('V', 0),
                    'rho': result.get('rhf', 0),
                    'V_V0': result.get('V_V0', 0),
                    'exp_method': 'calculated'
                }
                
                # 使用参数化INSERT语句
                stmt = text("""
                    INSERT INTO copper_shock_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method)
                """)
                conn.execute(stmt, data)
        st.success(f"成功保存 {len(results)} 条计算结果到数据库")
    except Exception as e:
        st.error(f"保存失败: {str(e)}")

# 以下是原有功能代码（保持不变）
# 冲击波参数计算函数
def calculate_shock_parameters(U_s, u_p, rho0):
    """计算冲击波参数"""
    P = rho0 * U_s * u_p
    V = 1 / rho0 * (1 - u_p / U_s)
    rho = rho0 * U_s / (U_s - u_p)
    V_V0 = V * rho0
    return P, V, rho, V_V0

# 冲击波Hugoniot关系拟合
def fit_hugoniot(df):
    """拟合冲击波Hugoniot关系"""
    U_s = df['Us'].values
    u_p = df['Up'].values
    
    # 线性拟合 U_s = C0 + S*u_p
    coeffs = np.polyfit(u_p, U_s, 1)
    C0 = coeffs[1]
    S = coeffs[0]
    
    return C0, S

# 对称冲击波速度计算
def symmetric_impact_velocity(projectile_velocities):
    """计算对称冲击下的粒子速度"""
    u_p = projectile_velocities / 2
    return u_p

# 图形生成函数
def generate_shock_plots(df, C0, S):
    """生成冲击波关系图"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # U_s vs u_p
    axs[0, 0].scatter(df['Up'], df['Us'], label='实验数据')
    u_p_range = np.linspace(0, df['Up'].max()*1.1, 100)
    U_s_fit = C0 + S * u_p_range
    axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'拟合曲线: $U_s = {C0:.2f} + {S:.2f}u_p$')
    axs[0, 0].set_xlabel('粒子速度 $u_p$ (km/s)')
    axs[0, 0].set_ylabel('冲击波速度 $U_s$ (km/s)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # P vs u_p
    axs[0, 1].scatter(df['Up'], df['P'], label='实验数据')
    P_range = df['rho0'].iloc[0] * u_p_range * (C0 + S * u_p_range)
    axs[0, 1].plot(u_p_range, P_range, 'r-', label='理论曲线')
    axs[0, 1].set_xlabel('粒子速度 $u_p$ (km/s)')
    axs[0, 1].set_ylabel('压力 $P$ (GPa)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P vs V/V0
    axs[1, 0].scatter(df['V_V0'], df['P'], label='实验数据')
    V_V0_range = 1 - u_p_range / (C0 + S * u_p_range)
    P_V_range = df['rho0'].iloc[0] * u_p_range * (C0 + S * u_p_range)
    axs[1, 0].plot(V_V0_range, P_V_range, 'r-', label='理论曲线')
    axs[1, 0].set_xlabel('比容比 $V/V_0$')
    axs[1, 0].set_ylabel('压力 $P$ (GPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # rho vs P
    axs[1, 1].scatter(df['P'], df['rho'], label='实验数据')
    rho_range = df['rho0'].iloc[0] / (1 - u_p_range / (C0 + S * u_p_range))
    axs[1, 1].plot(P_range, rho_range, 'r-', label='理论曲线')
    axs[1, 1].set_xlabel('压力 $P$ (GPa)')
    axs[1, 1].set_ylabel('密度 $\\rho$ (g/cm³)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

# 保存图形到BytesIO
def save_plot_to_bytes(fig):
    """将matplotlib图形保存到BytesIO对象"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

# Streamlit应用主函数
def main():
    st.title("冲击波参数计算与分析系统")
    
    # 材料选择
    materials = get_all_materials()
    selected_material = st.sidebar.selectbox("选择材料", materials)
    
    if selected_material:
        material_df = get_material_data(selected_material)
        
        # 显示材料数据
        st.subheader(f"{selected_material} 材料数据")
        st.dataframe(material_df)
        
        # 计算冲击波参数
        st.subheader("冲击波参数计算")
        col1, col2 = st.columns(2)
        with col1:
            U_s = st.number_input("冲击波速度 U_s (km/s)", min_value=0.0, value=5.0)
        with col2:
            u_p = st.number_input("粒子速度 u_p (km/s)", min_value=0.0, value=1.0)
        
        rho0 = material_df['rho0'].iloc[0] if not material_df.empty else 8.96  # 铜的密度
        
        if st.button("计算冲击波参数"):
            P, V, rho, V_V0 = calculate_shock_parameters(U_s, u_p, rho0)
            
            result = {
                'rh0f': rho0,
                'Df': U_s,
                'uf': u_p,
                'Pf': P,
                'V': V,
                'rhf': rho,
                'V_V0': V_V0
            }
            
            st.success(f"""
                计算结果:
                - 压力 P = {P:.2f} GPa
                - 比容 V = {V:.4f} cm³/g
                - 密度 ρ = {rho:.2f} g/cm³
                - 比容比 V/V0 = {V_V0:.4f}
            """)
            
            # 保存结果到数据库
            save_results_to_db([result], selected_material)
        
        # 对称冲击计算
        st.subheader("对称冲击计算")
        projectile_velocity = st.number_input("弹丸速度 (km/s)", min_value=0.0, value=2.0)
        
        if st.button("计算对称冲击"):
            u_p_sym = symmetric_impact_velocity(projectile_velocity)
            st.success(f"对称冲击粒子速度 u_p = {u_p_sym:.2f} km/s")
        
        # 冲击波关系拟合
        if not material_df.empty:
            st.subheader("冲击波Hugoniot关系拟合")
            C0, S = fit_hugoniot(material_df)
            st.success(f"拟合结果: $U_s = {C0:.2f} + {S:.2f}u_p$")
            
            # 生成并显示图形
            fig = generate_shock_plots(material_df, C0, S)
            st.pyplot(fig)
            
            # 提供图形下载
            buf = save_plot_to_bytes(fig)
            st.download_button(
                label="下载图形",
                data=buf,
                file_name=f"{selected_material}_shock_relations.png",
                mime="image/png"
            )
    
    # 显示SQLite文件路径
    st.sidebar.markdown(f"**SQLite文件路径:** `{sqlite_path}`")

if __name__ == "__main__":
    main()
