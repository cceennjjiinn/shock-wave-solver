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
import logging

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建SQLite引擎（关键优化：线程安全）
sqlite_path = os.path.abspath('shock_wave_data.db')
sqlite_engine = create_engine(
    f'sqlite:///{sqlite_path}',
    connect_args={"check_same_thread": False}  # 解决多线程访问问题 
)

# SQLite性能优化
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # 写前日志提升并发 
    cursor.execute('PRAGMA synchronous=NORMAL;')  # 平衡性能与数据安全 
    cursor.execute('PRAGMA temp_store=MEMORY;')   # 临时表内存存储 
    cursor.close()

# 初始化数据库
def init_database():
    try:
        with sqlite_engine.connect() as conn:
            # 支持表结构变更（新增列）
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
                logger.info("数据库表创建成功")
            else:
                # 示例：动态添加列（按需扩展）
                try:
                    conn.execute(text("ALTER TABLE shock_wave_all_data ADD COLUMN source_lab TEXT"))
                except:
                    pass  # 忽略已存在列的异常
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        st.error(f"数据库初始化失败: {str(e)}")

init_database()

# 数据库操作函数
def get_all_materials():
    try:
        query = text("SELECT DISTINCT material FROM shock_wave_all_data")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['material'].tolist()
    except Exception as e:
        logger.warning(f"获取材料列表失败: {str(e)}")
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material_name):
    try:
        query = text("SELECT * FROM shock_wave_all_data WHERE material = :material")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'material': material_name})
        return df
    except Exception as e:
        logger.warning(f"获取材料数据失败: {str(e)}")
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    """批量保存数据（关键优化：事务+批量插入）"""
    if not results:
        st.warning("没有数据可保存")
        return
    
    try:
        with sqlite_engine.begin() as conn:  # 自动事务管理 
            # 临时提升写入性能 [[3]][[10]]
            conn.execute(text("PRAGMA synchronous=OFF"))
            
            # 批量插入（性能提升32x+ ）
            conn.execute(
                text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T)
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """),
                [{"material": material_name, **r} for r in results]  # 批量参数化
            )
            
            # 恢复安全设置
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            
        st.success("数据保存成功")
        logger.info(f"保存{len(results)}条数据到材料[{material_name}]")
    except Exception as e:
        logger.error(f"保存失败: {str(e)}")
        st.error(f"保存失败: {str(e)}")

# Streamlit界面优化
def main():
    st.title("冲击波数据分析平台")
    
    # 线程池管理数据库操作 [[9]][[14]]
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=4)
    
    # 材料选择
    materials = get_all_materials()
    selected_material = st.selectbox("选择材料", materials)
    
    if selected_material:
        # 异步加载数据（避免阻塞UI）
        with st.spinner("加载数据中..."):
            future = executor.submit(get_material_data, selected_material)
            df = future.result()
        
        if not df.empty:
            st.dataframe(df.head())
            
            # 数据分析模块
            if st.button("线性回归分析"):
                X = df[["Us"]].values
                y = df["Up"].values
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                # 可视化
                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue')
                ax.plot(X, y_pred, color='red')
                ax.set_xlabel("冲击波速度 (km/s)")
                ax.set_ylabel("粒子速度 (km/s)")
                st.pyplot(fig)
                st.write(f"R² = {r2:.4f}")
    
    # 数据保存模块
    with st.expander("添加新数据"):
        new_data = {
            "rho0": st.number_input("初始密度 (g/cm³)", min_value=0.0),
            "Us": st.number_input("冲击波速度 (km/s)", min_value=0.0),
            "Up": st.number_input("粒子速度 (km/s)", min_value=0.0),
            "P": st.number_input("冲击压力 (GPa)", min_value=0.0),
            "gamma": st.number_input("Grüneisen系数", min_value=0.0)
        }
        if st.button("提交"):
            # 异步保存 
            executor.submit(save_results_to_db, [new_data], selected_material)
            st.experimental_rerun()  # 刷新界面 

if __name__ == "__main__":
    main()
