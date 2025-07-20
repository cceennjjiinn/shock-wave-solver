import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import sympy as sp
from io import BytesIO

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 连接 SQLite 数据库
conn = sqlite3.connect('shock_wave_data.db')

# 初始化数据库
def init_database():
    cursor = conn.cursor()
    
    # 创建数据表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS copper_shock_data (
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
    ''')
    
    # 检查是否需要从 MySQL 迁移数据
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='copper_shock_data'")
    if cursor.fetchone()[0] == 0:
        st.info("检测到新数据库，尝试从 MySQL 迁移数据...")
        migrate_from_mysql()
    
    conn.commit()

# 从 MySQL 迁移数据（仅首次运行）
def migrate_from_mysql():
    try:
        # 尝试连接 MySQL（需要安装 mysql-connector-python）
        import mysql.connector
        from mysql.connector import Error
        
        mysql_conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='14122Lyx.',
            database='shock_wave_db'
        )
        
        if mysql_conn.is_connected():
            st.info("成功连接到 MySQL 数据库，开始迁移数据...")
            cursor = mysql_conn.cursor()
            
            # 检查表是否存在
            cursor.execute("SHOW TABLES LIKE 'copper_shock_data'")
            if cursor.fetchone():
                # 从 MySQL 导出数据
                df = pd.read_sql("SELECT * FROM copper_shock_data", mysql_conn)
                
                # 导入到 SQLite
                if not df.empty:
                    df.to_sql('copper_shock_data', conn, index=False, if_exists='replace')
                    st.success(f"成功从 MySQL 迁移 {len(df)} 条记录到 SQLite")
                else:
                    st.warning("MySQL 表中没有数据")
            else:
                st.warning("MySQL 中找不到 'copper_shock_data' 表")
                
    except ImportError:
        st.warning("未安装 mysql-connector-python，跳过数据迁移。请手动导入数据或安装该库。")
    except Error as e:
        st.error(f"连接 MySQL 失败: {str(e)}")
    except Exception as e:
        st.error(f"数据迁移过程中出错: {str(e)}")

# 初始化数据库
init_database()

# 数据库操作函数
def get_all_materials():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT material FROM copper_shock_data")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material_name):
    try:
        query = f"SELECT * FROM copper_shock_data WHERE material = '{material_name}'"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    if not results:
        st.warning("没有数据可保存")
        return
        
    try:
        cursor = conn.cursor()
        for result in results:
            data = (
                material_name,
                result.get('rh0f', 0),
                result.get('Df', 0),
                result.get('uf', 0),
                result.get('Pf', 0),
                result.get('V', 0),
                result.get('rhf', 0),
                result.get('V_V0', 0),
                'calculated'
            )
            
            insert_query = """
            INSERT INTO copper_shock_data 
            (material, rho0, Us, Up, P, V, rho, V_V0, exp_method) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, data)
        
        conn.commit()
        st.success(f"成功保存 {len(results)} 条数据到数据库")
    except Exception as e:
        st.error(f"保存到数据库失败: {str(e)}")

# 冲击波计算函数
def calculate_shock_wave(rho0, Us, Up_range=None, num_points=50):
    """
    计算冲击波参数
    
    参数:
    - rho0: 初始密度 (g/cm³)
    - Us: 冲击波速度 (km/s)
    - Up_range: 粒子速度范围 (km/s)，若为None则使用Us计算
    - num_points: 计算点数
    
    返回:
    - 计算结果列表
    """
    results = []
    
    # 如果没有提供Up范围，计算单个点
    if Up_range is None:
        # 计算冲击波后的参数
        rh0f = rho0  # 初始流体密度
        Df = Us      # 冲击波速度
        uf = Up_range if Up_range is not None else 0.5 * Us  # 粒子速度
        Pf = rho0 * Us * uf * 10  # 压力 (GPa)，乘以10是单位转换
        V = rho0 / (rho0 + uf * rho0 / Us)  # 比容
        rhf = 1 / V  # 最终密度
        V_V0 = V / (1/rho0)  # 比容比
        
        results.append({
            'rh0f': rh0f,
            'Df': Df,
            'uf': uf,
            'Pf': Pf,
            'V': V,
            'rhf': rhf,
            'V_V0': V_V0
        })
    else:
        # 计算一系列点
        Up_min, Up_max = Up_range
        Up_values = np.linspace(Up_min, Up_max, num_points)
        
        for uf in Up_values:
            # 计算冲击波后的参数
            rh0f = rho0
            Df = Us
            Pf = rho0 * Us * uf * 10  # GPa
            V = rho0 / (rho0 + uf * rho0 / Us)
            rhf = 1 / V
            V_V0 = V / (1/rho0)
            
            results.append({
                'rh0f': rh0f,
                'Df': Df,
                'uf': uf,
                'Pf': Pf,
                'V': V,
                'rhf': rhf,
                'V_V0': V_V0
            })
    
    return results

# 计算Hugoniot曲线
def calculate_hugoniot(rho0, c0, s, P_max=1000, num_points=100):
    """
    计算材料的Hugoniot曲线
    
    参数:
    - rho0: 初始密度 (g/cm³)
    - c0: 声速 (km/s)
    - s: 冲击阻抗系数
    - P_max: 最大压力 (GPa)
    - num_points: 计算点数
    
    返回:
    - Hugoniot曲线数据
    """
    P_values = np.linspace(0, P_max, num_points)
    hugoniot_data = []
    
    for P in P_values:
        if P == 0:
            # 零压力点
            V = 1/rho0
            rho = rho0
            Us = c0
            Up = 0
        else:
            # 使用Rankine-Hugoniot关系计算
            # P = rho0 * Us * Up
            # Us = c0 + s * Up
            # 联立方程解得:
            a = rho0 * s**2
            b = 2 * rho0 * c0 * s
            c = rho0 * c0**2 - P/10  # 转换为km/s单位
            
            # 解二次方程 a*Up^2 + b*Up + c = 0
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                continue
                
            Up = (-b + np.sqrt(discriminant)) / (2*a)
            Us = c0 + s * Up
            V = 1/rho0 - Up*Us/(10*P)  # 比容，单位转换
            rho = 1/V
            
        hugoniot_data.append({
            'P': P,
            'V': V,
            'rho': rho,
            'Us': Us,
            'Up': Up,
            'V_V0': V / (1/rho0)
        })
    
    return hugoniot_data

# 绘制冲击波参数关系图
def plot_shock_parameters(results, material_data=None):
    """
    绘制冲击波参数关系图
    
    参数:
    - results: 计算结果列表
    - material_data: 材料实验数据 (DataFrame)
    """
    if not results:
        st.warning("没有数据可绘制")
        return
    
    # 提取计算数据
    Us_values = [r['Df'] for r in results]
    Up_values = [r['uf'] for r in results]
    P_values = [r['Pf'] for r in results]
    V_V0_values = [r['V_V0'] for r in results]
    
    # 创建两个图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制 Us-Up 关系图
    ax1.plot(Up_values, Us_values, 'b-', label='计算结果')
    ax1.set_xlabel('粒子速度 $U_p$ (km/s)')
    ax1.set_ylabel('冲击波速度 $U_s$ (km/s)')
    ax1.set_title('冲击波速度 vs 粒子速度')
    ax1.grid(True)
    
    # 如果有实验数据，添加到图中
    if material_data is not None and not material_data.empty:
        ax1.scatter(material_data['Up'], material_data['Us'], color='red', label='实验数据')
    
    ax1.legend()
    
    # 绘制 P-V/V0 关系图
    ax2.plot(V_V0_values, P_values, 'g-', label='计算结果')
    ax2.set_xlabel('比容比 $V/V_0$')
    ax2.set_ylabel('压力 $P$ (GPa)')
    ax2.set_title('Hugoniot曲线 (压力 vs 比容比)')
    ax2.grid(True)
    
    # 如果有实验数据，添加到图中
    if material_data is not None and not material_data.empty:
        ax2.scatter(material_data['V_V0'], material_data['P'], color='red', label='实验数据')
    
    ax2.legend()
    
    # 使用工程单位格式化坐标轴
    for ax in [ax1, ax2]:
        formatter = EngFormatter(unit='')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 返回图表数据用于下载
    return fig

# 主应用
def main():
    st.title("多物理场冲击波参数计算器")
    
    # 侧边栏 - 参数设置
    st.sidebar.header("材料参数设置")
    
    # 选择材料或自定义
    materials = get_all_materials()
    material_selection = st.sidebar.selectbox(
        "选择材料",
        ["自定义"] + materials
    )
    
    if material_selection != "自定义":
        # 从数据库加载材料参数
        material_df = get_material_data(material_selection)
        if not material_df.empty:
            # 使用第一条记录作为默认参数
            sample = material_df.iloc[0]
            rho0_default = sample['rho0']
            Us_default = sample['Us']
            st.sidebar.info(f"从数据库加载材料: {material_selection}")
        else:
            rho0_default = 8.96  # 铜的密度 (g/cm³)
            Us_default = 3.94    # 铜的典型冲击波速度 (km/s)
            st.sidebar.warning(f"材料 '{material_selection}' 数据为空，使用默认值")
    else:
        # 自定义参数
        rho0_default = 8.96  # 铜的密度 (g/cm³)
        Us_default = 3.94    # 铜的典型冲击波速度 (km/s)
    
    # 输入参数
    rho0 = st.sidebar.number_input("初始密度 ρ₀ (g/cm³)", min_value=0.1, value=rho0_default, step=0.1)
    Us = st.sidebar.number_input("冲击波速度 Uₛ (km/s)", min_value=0.1, value=Us_default, step=0.1)
    
    # 计算模式
    calc_mode = st.sidebar.radio("计算模式", ["单点计算", "参数扫描"])
    
    if calc_mode == "单点计算":
        Up = st.sidebar.number_input("粒子速度 Uₚ (km/s)", min_value=0.0, value=Us/2, step=0.1)
        num_points = 1
    else:
        Up_min = st.sidebar.number_input("最小粒子速度 Uₚ_min (km/s)", min_value=0.0, value=0.1, step=0.1)
        Up_max = st.sidebar.number_input("最大粒子速度 Uₚ_max (km/s)", min_value=Up_min, value=Us, step=0.1)
        num_points = st.sidebar.slider("计算点数", min_value=5, max_value=200, value=50, step=5)
        Up_range = (Up_min, Up_max)
    
    # 计算按钮
    if st.sidebar.button("计算"):
        # 执行计算
        if calc_mode == "单点计算":
            results = calculate_shock_wave(rho0, Us, Up)
        else:
            results = calculate_shock_wave(rho0, Us, Up_range, num_points)
        
        if results:
            # 显示计算结果
            st.subheader("计算结果")
            
            # 显示单点结果或统计信息
            if len(results) == 1:
                result = results[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("密度 ρ", f"{result['rhf']:.3f} g/cm³")
                    st.metric("压力 P", f"{result['Pf']:.3f} GPa")
                with col2:
                    st.metric("冲击波速度 Uₛ", f"{result['Df']:.3f} km/s")
                    st.metric("粒子速度 Uₚ", f"{result['uf']:.3f} km/s")
                with col3:
                    st.metric("比容 V", f"{result['V']:.6f} cm³/g")
                    st.metric("比容比 V/V₀", f"{result['V_V0']:.6f}")
            else:
                # 转换为DataFrame并显示统计信息
                df = pd.DataFrame(results)
                st.dataframe(df.describe().round(3))
                
                # 保存结果到数据库
                if st.button("保存计算结果到数据库"):
                    save_results_to_db(results, material_selection if material_selection != "自定义" else "Custom")
            
            # 绘制图表
            material_data = get_material_data(material_selection) if material_selection != "自定义" else None
            fig = plot_shock_parameters(results, material_data)
            
            # 下载图表
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="下载图表",
                data=buf,
                file_name=f"shock_wave_{material_selection}.png",
                mime="image/png"
            )
            
            # 下载数据
            if len(results) > 1:
                csv_data = df.to_csv(sep='\t', na_rep='nan')
                st.download_button(
                    label="下载数据",
                    data=csv_data,
                    file_name=f"shock_wave_data_{material_selection}.txt",
                    mime="text/plain"
                )
    
    # 显示数据库中的材料数据
    st.subheader("材料数据库")
    if materials:
        selected_material = st.selectbox("查看材料数据", materials)
        if st.button("加载材料数据"):
            material_df = get_material_data(selected_material)
            if not material_df.empty:
                st.dataframe(material_df)
                
                # 计算并显示Hugoniot曲线
                st.subheader(f"{selected_material} 的 Hugoniot 曲线")
                
                # 从数据中估计Hugoniot参数 (Us = c0 + s*Up)
                if len(material_df) >= 2:
                    # 线性拟合 Us-Up 关系
                    x = material_df['Up'].values
                    y = material_df['Us'].values
                    s, c0 = np.polyfit(x, y, 1)
                    
                    st.info(f"Hugoniot参数估计: c₀ = {c0:.3f} km/s, s = {s:.3f}")
                    
                    # 计算Hugoniot曲线
                    hugoniot_data = calculate_hugoniot(rho0, c0, s)
                    hugoniot_df = pd.DataFrame(hugoniot_data)
                    
                    # 绘制Hugoniot曲线
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Us-Up 关系
                    ax1.plot(hugoniot_df['Up'], hugoniot_df['Us'], 'b-', label='Hugoniot曲线')
                    ax1.scatter(material_df['Up'], material_df['Us'], color='red', label='实验数据')
                    ax1.set_xlabel('粒子速度 $U_p$ (km/s)')
                    ax1.set_ylabel('冲击波速度 $U_s$ (km/s)')
                    ax1.set_title('冲击波速度 vs 粒子速度')
                    ax1.grid(True)
                    ax1.legend()
                    
                    # P-V/V0 关系
                    ax2.plot(hugoniot_df['V_V0'], hugoniot_df['P'], 'g-', label='Hugoniot曲线')
                    ax2.scatter(material_df['V_V0'], material_df['P'], color='red', label='实验数据')
                    ax2.set_xlabel('比容比 $V/V_0$')
                    ax2.set_ylabel('压力 $P$ (GPa)')
                    ax2.set_title('Hugoniot曲线 (压力 vs 比容比)')
                    ax2.grid(True)
                    ax2.legend()
                    
                    # 使用工程单位格式化坐标轴
                    for ax in [ax1, ax2]:
                        formatter = EngFormatter(unit='')
                        ax.xaxis.set_major_formatter(formatter)
                        ax.yaxis.set_major_formatter(formatter)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("数据点不足，无法计算Hugoniot曲线")
            else:
                st.warning(f"材料 '{selected_material}' 没有数据")
    else:
        st.info("数据库中没有材料数据")

if __name__ == "__main__":
    main()