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
from io import BytesIO, StringIO
from PIL import Image
import itertools
import os
import openpyxl  # 新增：用于生成Excel文件

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

# 修复数据库表结构 - 确保gamma字段存在
def fix_database_schema():
    """修复数据库表结构，添加缺失的gamma字段"""
    try:
        with sqlite_engine.connect() as conn:
            # 检查是否存在gamma字段
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(shock_wave_all_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'gamma' not in columns:
                conn.execute(text("ALTER TABLE shock_wave_all_data ADD COLUMN gamma REAL"))
                conn.commit()
                st.success("数据库表结构已修复，添加了gamma字段")
    except Exception as e:
        st.error(f"修复数据库表结构失败: {str(e)}")

# 先初始化数据库，再修复可能的表结构问题
init_database()
fix_database_schema()

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
    """按需查询字段以减少数据传输，确保包含实验方法字段"""
    try:
        if fields is None:
            fields = '*'  # 默认查询所有字段
        else:
            # 确保包含实验方法字段用于颜色区分
            if 'exp_method' not in fields:
                fields.append('exp_method')
            fields = ', '.join(fields)  # 按需指定字段
        query = text(f"SELECT {fields} FROM shock_wave_all_data WHERE material = :material")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'material': material_name})
        return df
    except Exception as e:
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    """保存多组求解结果到数据库，返回保存的记录数"""
    if not results:
        return 0
        
    try:
        count = 0
        with sqlite_engine.begin() as conn:
            for result in results:
                # 检查必要的参数是否存在
                required_params = ['rh0f', 'Df', 'uf', 'Pf']
                if not all(param in result for param in required_params):
                    continue
                    
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
                    'gamma': result.get('gammaf', 0),
                    'T': result.get('Tf', 0)
                }
                stmt = text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, data)
                count += 1
        return count
    except Exception as e:
        st.error(f"保存失败: {str(e)}")
        return 0

def save_input_parameters(input_params, material_name="Copper", exp_method="manual_input"):
    """保存当前输入的参数到数据库"""
    try:
        # 提取关键参数
        data = {
            'material': material_name,
            'rho0': input_params.get('rh0f') if isinstance(input_params.get('rh0f'), (int, float)) else 0,
            'Us': input_params.get('Df') if isinstance(input_params.get('Df'), (int, float)) else 0,
            'Up': input_params.get('uf') if isinstance(input_params.get('uf'), (int, float)) else 0,
            'P': input_params.get('Pf') if isinstance(input_params.get('Pf'), (int, float)) else 0,
            'V': 0,  # 无法直接从输入参数获取
            'rho': input_params.get('rhf') if isinstance(input_params.get('rhf'), (int, float)) else 0,
            'V_V0': 0,  # 无法直接从输入参数获取
            'exp_method': exp_method,
            'gamma': input_params.get('gammaf') if isinstance(input_params.get('gammaf'), (int, float)) else 0,
            'T': input_params.get('Tf') if isinstance(input_params.get('Tf'), (int, float)) else 0
        }
        
        with sqlite_engine.begin() as conn:
            stmt = text("""
                INSERT INTO shock_wave_all_data 
                (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
            """)
            conn.execute(stmt, data)
        return 1
    except Exception as e:
        st.error(f"保存输入参数失败: {str(e)}")
        return 0

def save_input_data_to_db(input_data, material_name, exp_method="manual_input"):
    """保存计算结果到数据库，返回保存的记录数"""
    try:
        # 验证输入数据的有效性
        required_fields = ['rho0', 'Us', 'Up', 'P']
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                st.error(f"保存失败：缺少必要的参数 {field}")
                return 0
                
            # 确保数值有效
            if not isinstance(input_data[field], (int, float)) or input_data[field] <= 0:
                st.error(f"保存失败：参数 {field} 必须是正数")
                return 0

        with sqlite_engine.begin() as conn:
            data = {
                'material': material_name,
                'rho0': float(input_data['rho0']),
                'Us': float(input_data['Us']),
                'Up': float(input_data['Up']),
                'P': float(input_data['P']),
                'V': float(input_data.get('V', 0)),
                'rho': float(input_data.get('rho', 0)),
                'V_V0': float(input_data.get('V_V0', 0)),
                'exp_method': exp_method,
                'gamma': float(input_data.get('gamma', 0)),
                'T': float(input_data.get('T', 0))
            }
            stmt = text("""
                INSERT INTO shock_wave_all_data 
                (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
            """)
            conn.execute(stmt, data)
        return 1
    except Exception as e:
        st.error(f"保存输入数据失败: {str(e)}")
        return 0

# 批量导入数据到数据库
def bulk_import_data(df, material_name, exp_method="bulk_import"):
    """批量导入数据到数据库，返回成功导入的记录数"""
    if df.empty:
        return 0
        
    required_columns = ['rho0', 'Us', 'Up']  # 至少需要这三个参数
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"导入失败：文件缺少必要的列: {', '.join(missing_cols)}")
        return 0
        
    try:
        count = 0
        with sqlite_engine.begin() as conn:
            for _, row in df.iterrows():
                # 跳过包含空值的行
                if row[required_columns].isnull().any():
                    continue
                    
                data = {
                    'material': material_name,
                    'rho0': row.get('rho0', 0),
                    'Us': row.get('Us', 0),
                    'Up': row.get('Up', 0),
                    'P': row.get('P', 0),
                    'V': row.get('V', 0),
                    'rho': row.get('rho', 0),
                    'V_V0': row.get('V_V0', 0),
                    'exp_method': exp_method,
                    'gamma': row.get('gamma', 0),
                    'T': row.get('T', 0)
                }
                stmt = text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, data)
                count += 1
        return count
    except Exception as e:
        st.error(f"批量导入失败: {str(e)}")
        return 0

# 批量删除选中的记录
def bulk_delete_records(ids):
    """删除指定ID的记录，返回删除的记录数"""
    if not ids or not isinstance(ids, list):
        return 0
        
    try:
        with sqlite_engine.begin() as conn:
            placeholders = ', '.join([':id' + str(i) for i in range(len(ids))])
            params = {'id' + str(i): id for i, id in enumerate(ids)}
            stmt = text(f"DELETE FROM shock_wave_all_data WHERE id IN ({placeholders})")
            result = conn.execute(stmt, params)
            return result.rowcount
    except Exception as e:
        st.error(f"删除失败: {str(e)}")
        return 0

# 清空指定材料的所有数据
def clear_material_data(material_name):
    """清空指定材料的所有数据，返回删除的记录数"""
    if not material_name:
        return 0
        
    try:
        with sqlite_engine.begin() as conn:
            stmt = text("DELETE FROM shock_wave_all_data WHERE material = :material")
            result = conn.execute(stmt, {'material': material_name})
            return result.rowcount
    except Exception as e:
        st.error(f"清空数据失败: {str(e)}")
        return 0

def view_database():
    """显示数据库内容，包含批量添加和删除功能"""
    with st.expander("数据库内容", expanded=True):
        # 批量操作区域
        st.subheader("批量数据操作")
        col1, col2 = st.columns(2)
        
        # 批量导入部分
        with col1:
            st.subheader("批量导入数据")
            new_material = st.text_input("材料名称 (Material Name)", help="输入要导入数据的材料名称，使用英文")
            # 支持Excel和CSV文件上传
            uploaded_file = st.file_uploader("选择Excel或CSV文件", type=["csv", "xlsx", "xls"])
            exp_method = st.text_input("实验方法/数据来源", value="bulk_import")
            
            if st.button("导入数据"):
                if not new_material:
                    st.error("请输入材料名称")
                elif uploaded_file is None:
                    st.error("请选择文件")
                else:
                    # 读取文件，根据文件类型自动处理
                    try:
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        if file_extension in ['xlsx', 'xls']:
                            # 读取Excel文件
                            df = pd.read_excel(uploaded_file)
                            st.success(f"成功读取Excel文件，包含 {len(df)} 条记录")
                        else:
                            # 读取CSV文件
                            df = pd.read_csv(uploaded_file)
                            st.success(f"成功读取CSV文件，包含 {len(df)} 条记录")
                            
                        st.dataframe(df.head())  # 显示前几行预览
                        
                        # 导入数据
                        count = bulk_import_data(df, new_material, exp_method)
                        if count > 0:
                            st.success(f"成功导入 {count} 条记录（跳过包含空值的行）")
                            st.rerun()
                        else:
                            st.warning("没有导入任何记录，请检查数据格式")
                    except Exception as e:
                        st.error(f"读取文件失败: {str(e)}")
        
        # 批量删除部分
        with col2:
            st.subheader("批量删除数据")
            materials = get_all_materials()
            if materials:
                del_material = st.selectbox("选择要操作的材料", materials, key="del_material")
                
                # 显示该材料的数据供选择删除
                df = get_material_data(del_material)
                if not df.empty and 'id' in df.columns:
                    # 添加复选框选择要删除的记录
                    df['选择'] = False
                    edited_df = st.data_editor(
                        df,
                        column_config={
                            "选择": st.column_config.CheckboxColumn(
                                "选择删除",
                                default=False,
                            )
                        },
                        disabled=df.columns.difference(["选择"]),
                        hide_index=True,
                    )
                    
                    # 获取选中的记录ID
                    selected_ids = edited_df[edited_df['选择']]['id'].tolist()
                    
                    col_del1, col_del2 = st.columns(2)
                    with col_del1:
                        if st.button("删除所选记录"):
                            if selected_ids:
                                if st.session_state.get('confirm_delete', False):
                                    deleted = bulk_delete_records(selected_ids)
                                    if deleted > 0:
                                        st.success(f"成功删除 {deleted} 条记录")
                                        st.session_state['confirm_delete'] = False
                                        st.rerun()
                                    else:
                                        st.warning("删除失败或没有记录被删除")
                                else:
                                    st.warning("请确认删除操作")
                                    st.session_state['confirm_delete'] = True
                                    st.rerun()
                            else:
                                st.warning("请先选择要删除的记录")
                    
                    with col_del2:
                        if st.button("清空该材料所有数据"):
                            if st.session_state.get('confirm_clear', False):
                                deleted = clear_material_data(del_material)
                                if deleted > 0:
                                    st.success(f"成功清空 {del_material} 的所有 {deleted} 条记录")
                                    st.session_state['confirm_clear'] = False
                                    st.rerun()
                                else:
                                    st.warning("清空失败或该材料没有数据")
                            else:
                                st.warning("此操作将删除该材料所有数据，请确认")
                                st.session_state['confirm_clear'] = True
                                st.rerun()
                else:
                    st.info(f"材料 {del_material} 暂无数据可删除")
            else:
                st.info("数据库中暂无材料数据")
        
        # 数据查看部分
        st.subheader("数据查看与导出")
        materials = get_all_materials()
        if not materials:
            st.info("数据库中暂无数据")
            return
            
        selected_material = st.selectbox("选择材料查看数据", materials, key="view_material")
        df = get_material_data(selected_material)
        
        if df.empty:
            st.info(f"材料 {selected_material} 暂无数据")
        else:
            st.info(f"材料 {selected_material} 共有 {len(df)} 条记录")
            st.dataframe(df)
            
            # 提供下载选项
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"下载 {selected_material} 数据 (CSV)",
                data=csv,
                file_name=f"{selected_material}_data.csv",
                mime="text/csv",
            )
            
            # 新增Excel下载选项
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label=f"下载 {selected_material} 数据 (Excel)",
                data=excel_buffer,
                file_name=f"{selected_material}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            
            # 提供数据导入模板下载，同时支持CSV和Excel格式
            st.subheader("数据导入模板下载")
            template = pd.DataFrame(columns=[
                'rho0', 'Us', 'Up', 'P', 'V', 'rho', 
                'V_V0', 'gamma', 'T'
            ])
            template.loc[0] = [8.96, 5.0, 1.0, 44.8, 0.089, 11.2, 0.8, 2.0, 3000]
            
            # CSV模板
            csv_template = template.to_csv(index=False)
            st.download_button(
                label="下载CSV模板",
                data=csv_template,
                file_name="shock_wave_data_template.csv",
                mime="text/csv",
                key="csv_template_btn"
            )
            
            # Excel模板
            excel_template_buffer = BytesIO()
            template.to_excel(excel_template_buffer, index=False, engine='openpyxl')
            excel_template_buffer.seek(0)
            st.download_button(
                label="下载Excel模板",
                data=excel_template_buffer,
                file_name="shock_wave_data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_template_btn"
            )

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
        st.warning(f"{material_type} material '{material_name}' has no data")
        return None
    
    # 过滤异常值
    df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
    if len(df) < 2:
        st.warning(f"{material_type} material '{material_name}' has insufficient valid data for fitting")
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
    
    st.info(f"{material_type} material {material_name} fitting results: Us = {C0:.4f} + {S:.4f}*Up")
    st.info(f"Fitting errors: R² = {r2:.4f}, RMSE = {rmse:.4f} km/s, MAE = {mae:.4f} km/s")
    st.info(f"Average parameters: ρ₀ = {df['rho0'].mean():.4f} g/cm³, Average pressure = {df['P'].mean():.4f} GPa")
    
    # 按实验方法统计数据
    if 'exp_method' in df.columns:
        method_counts = df['exp_method'].value_counts()
        st.info(f"Experimental method distribution: {', '.join([f'{k}: {v} records' for k, v in method_counts.items()])}")
    
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
def get_input_streamlit(label, var_name, key, default=None, unit="", desc="", disabled=False):
    st.caption(f"{desc} | Unit: {unit}")
    input_type = st.radio(
        f"{label} Input Type",
        ["Single Value", "Multiple Values (comma separated)", "Range (with step)"],
        key=f"{key}_type",
        horizontal=True,
        disabled=disabled,
        help="Select input method: single value, multiple discrete values, or continuous range"
    )
    
    default_val = str(default) if default is not None else ""
    
    if input_type == "Single Value":
        val = st.text_input(label, default_val, key=f"{key}_single", disabled=disabled)
        if val == "":
            return symbols(var_name)
        try:
            return [float(val)]
        except ValueError:
            st.error("Please enter a valid number (e.g., 3.14)")
            return None
    elif input_type == "Multiple Values (comma separated)":
        val = st.text_input(
            label, 
            default_val, 
            key=f"{key}_multi", 
            disabled=disabled,
            help="Enter multiple values separated by commas (e.g., 1.5, 3.0, 4.5)"
        )
        if val == "":
            return symbols(var_name)
        try:
            # 处理可能的空格并分割
            values = [float(x.strip()) for x in val.split(',') if x.strip()]
            if not values:
                st.error("Please enter at least one value")
                return None
            return values
        except ValueError:
            st.error("Please enter valid comma-separated numbers (e.g., 1.0, 2.5, 3.8)")
            return None
    else:
        st.caption("Range example: start=1.0, end=5.0, step=1.0 → generates [1.0, 2.0, 3.0, 4.0, 5.0]")
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.text_input(
                f"{label} Start Value", 
                default_val, 
                key=f"{key}_start", 
                disabled=disabled,
                help="First value in the range (e.g., 2.0)"
            )
        with col2:
            end = st.text_input(
                f"{label} End Value", 
                "", 
                key=f"{key}_end", 
                disabled=disabled,
                help="Last value in the range (must be greater than start value, e.g., 10.0)"
            )
        with col3:
            step = st.text_input(
                f"{label} Step (optional)", 
                "0.5", 
                key=f"{key}_step", 
                disabled=disabled,
                help="Increment value (e.g., 0.5 or 2.0, default 0.5)"
            )
            
        if not start or not end:
            return symbols(var_name)
            
        try:
            start = float(start)
            end = float(end)
            step = float(step) if step else 0.5
            
            # 验证和修正输入
            if step <= 0:
                step = 0.5
                st.warning("Step must be positive, automatically set to 0.5")
            if start > end:
                start, end = end, start
                st.warning("Start value greater than end value, automatically swapped to ascending order")
            if (end - start) < step:
                st.warning("Step larger than range difference, will return only start value")
                return [start]
                
            # 生成范围值
            values = []
            current = start
            epsilon = 1e-9  # 处理浮点数精度问题
            while current <= end + epsilon:
                values.append(round(current, 6))
                current += step
            return values
        except ValueError:
            st.error("Please enter valid range numbers (e.g., start=1.0, end=5.0, step=1.0)")
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

# 冲击波关系图绘制 - 完全使用英文标签
@st.cache_data(ttl=3600)  # 缓存图像结果
def generate_shock_plots(df, C0, S, material_name, material_type):
    # 数据量大时进行采样
    if len(df) > 1000:
        df = df.sample(1000)
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 定义实验方法的颜色映射 - 确保iml为红色，ssp为蓝色
    method_colors = {
        'iml': 'red',
        'ssp': 'blue',
        'calculated': 'green',
        'manual_input': 'purple',
        'bulk_import': 'orange'  # 批量导入数据的颜色标识
    }
    default_color = 'gray'  # 未定义的实验方法用灰色
    
    # 标题使用英文
    fig.suptitle(f'{material_type} Material: {material_name} - Shock Wave Relationships', fontsize=16)
    
    # 获取所有唯一的实验方法
    if 'exp_method' in df.columns:
        methods = df['exp_method'].unique()
    else:
        methods = ['unknown']
        df['exp_method'] = 'unknown'
    
    # Us vs Up
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[0, 0].scatter(
            method_df['Up'], method_df['Us'], 
            label=f'{method}',
            color=color, alpha=0.7
        )
    
    u_p_range = np.linspace(0, df['Up'].max()*1.1, 100)
    U_s_fit = C0 + S * u_p_range
    
    axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'Fit: Us = {C0:.2f} + {S:.2f}·Up')
    axs[0, 0].set_xlabel('Particle Velocity Up (km/s)')
    axs[0, 0].set_ylabel('Shock Velocity Us (km/s)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # P vs Up
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[0, 1].scatter(
            method_df['Up'], method_df['P'], 
            label=f'{method}' if method == methods[0] else "",  # 只在第一个图显示完整图例
            color=color, alpha=0.7
        )
    
    # 使用数据中的平均密度而非硬编码值
    rho0 = df['rho0'].mean() if not df.empty else 8.96
    P_range = rho0 * U_s_fit * u_p_range  # P = rho0 * Us * Up
    
    axs[0, 1].plot(u_p_range, P_range, 'r-', label='Theoretical: P = ρ0·Us·Up')
    axs[0, 1].set_xlabel('Particle Velocity Up (km/s)')
    axs[0, 1].set_ylabel('Pressure P (GPa)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P vs V/V0
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 0].scatter(
            method_df['V_V0'], method_df['P'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7
        )
    
    V_V0_range = 1 - u_p_range / U_s_fit  # V/V0 = 1 - Up/Us
    
    axs[1, 0].plot(V_V0_range, P_range, 'r-', label='Theoretical Curve')
    axs[1, 0].set_xlabel('Specific Volume Ratio V/V0')
    axs[1, 0].set_ylabel('Pressure P (GPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # rho vs P
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 1].scatter(
            method_df['P'], method_df['rho'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7
        )
    
    rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)  # rho = rho0·Us/(Us-Up)
    
    axs[1, 1].plot(P_range, rho_range, 'r-', label='Theoretical Curve')
    axs[1, 1].set_xlabel('Pressure P (GPa)')
    axs[1, 1].set_ylabel('Density ρ (g/cm³)')
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
def display_material_plots(df, material_name, material_type):
    if not df.empty:
        with st.expander(f"View {material_type} Material {material_name} Shock Wave Plots", expanded=True):
            C0, S = fit_hugoniot(df)
            fig = generate_shock_plots(df, C0, S, material_name, material_type)
            st.pyplot(fig)
            buf = save_plot_to_bytes(fig)
            
            material_type_en = {
                "飞片": "flyer",
                "基板": "substrate",
                "样品": "sample"
            }.get(material_type, material_type.lower())
            download_label = f"Download {material_type} Material {material_name} Shock Wave Plots"
            file_name = f"{material_type_en}_{material_name}_shock_relations.png"
                
            st.download_button(
                label=download_label,
                data=buf,
                file_name=file_name,
                mime="image/png"
            )
    else:
        st.info(f"No available data to generate {material_type} material {material_name} plots")

# 结果绘图函数 - 完全使用英文标签
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
                 fmt='bo', ecolor='r', capsize=5, label='Flyer Data')
    ax1.set_xlabel('Particle Velocity Up (km/s)')
    ax1.set_ylabel('Shock Pressure P (GPa)')
    ax1.set_title('Pressure-Particle Velocity Relationship (with error range)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 温度-压力图
    ax2 = fig.add_subplot(222)
    ax2.scatter(pf_values, tf_values, c='orange', label='Flyer Temperature')
    ax2.set_xlabel('Shock Pressure P (GPa)')
    ax2.set_ylabel('Shock Temperature T (K)')
    ax2.set_title('Temperature-Pressure Relationship')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 冲击波速度-粒子速度图
    ax3 = fig.add_subplot(223)
    ax3.scatter(uf_values, df_values, c='blue', label='Flyer')
    ax3.set_xlabel('Particle Velocity Up (km/s)')
    ax3.set_ylabel('Shock Wave Velocity Us (km/s)')
    ax3.set_title('Shock Velocity-Particle Velocity Relationship')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 密度-压力图
    ax4 = fig.add_subplot(224)
    ax4.scatter(pf_values, rhf_values, c='green', label='Flyer')
    ax4.set_xlabel('Shock Pressure P (GPa)')
    ax4.set_ylabel('Compressed Density (g/cm³)')
    ax4.set_title('Density-Pressure Relationship')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

# 页面函数
def home_page():
    st.title("冲击波参数计算与分析系统")
    st.info("""
    System Core Model Description:
    1. Based on Rankine-Hugoniot conservation equations (mass, momentum, energy conservation)
    2. Assumptions: planar shock wave, steady propagation, negligible initial pressure
    3. Unit system: density (g/cm³), velocity (km/s), pressure (GPa)
    """)
    
    # 查看数据库快捷入口
    if st.button("View Database"):
        st.session_state.page = "view_database"
        st.rerun()
    
    st.write("Select Operation Mode:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use Database Data"):
            st.session_state.page = "database_mode"
            st.rerun()  # 立即刷新页面
    with col2:
        if st.button("Manual Input Parameters"):
            st.session_state.page = "manual_mode"
            st.rerun()  # 立即刷新页面
    
    # 底部返回按钮
    st.markdown("---")
    if st.button("Return to Previous Page"):
        # 如果在首页，返回按钮不执行任何操作或给出提示
        st.info("Currently on the home page")

def database_mode_page():
    st.title("Database Mode")
    st.write("Load material data from database, fit parameters based on Hugoniot relationships and solve")
    
    # 查看数据库快捷入口
    if st.button("View Database"):
        st.session_state.page = "view_database"
        st.rerun()
    
    materials = get_all_materials()
    if not materials:
        st.error("No materials available in database")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.selectbox("Flyer Material", materials, key="flyer_material")
    with col2:
        base_material = st.selectbox("Substrate Material", materials, key="base_material")
    with col3:
        sample_material = st.selectbox("Sample Material", materials, key="sample_material")
    
    # 检测相同材料并提供共享选项 - 修正逻辑：只有材料相同时才显示共享选项
    st.subheader("Material Parameter Sharing Settings")
    material_relations = {}
    
    # 飞片与基板是否相同
    share_flyer_base = False
    if flyer_material == base_material:
        share_flyer_base = st.checkbox(f"Flyer and substrate are both {flyer_material}, share parameters", value=True)
    material_relations['flyer_base'] = share_flyer_base
        
    # 基板与样品是否相同
    share_base_sample = False
    if base_material == sample_material:
        share_base_sample = st.checkbox(f"Substrate and sample are both {base_material}, share parameters", value=True)
    material_relations['base_sample'] = share_base_sample
        
    # 飞片与样品是否相同（仅当三者不都相同时显示）
    share_flyer_sample = False
    all_same = (flyer_material == base_material == sample_material)
    if flyer_material == sample_material and not all_same and not share_flyer_base:
        share_flyer_sample = st.checkbox(f"Flyer and sample are both {flyer_material}, share parameters", value=True)
    material_relations['flyer_sample'] = share_flyer_sample
    
    # 按需查询字段以减少数据传输，确保包含exp_method
    flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method'])
    
    # 根据共享设置决定是否复用数据
    if material_relations['flyer_base']:
        base_df = flyer_df.copy()
    else:
        base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method'])
    
    if material_relations['base_sample']:
        sample_df = base_df.copy()
    elif material_relations['flyer_sample']:
        sample_df = flyer_df.copy()
    else:
        sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method'])
    
    # 为每种材料类型拟合数据并清晰标注
    with st.spinner(f"Fitting flyer material {flyer_material} data..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material, "Flyer")
    
    # 根据共享设置决定是否复用拟合结果
    if material_relations['flyer_base']:
        base_fit = flyer_fit
        st.info(f"Substrate and flyer materials are the same, reusing flyer fitting parameters")
    else:
        with st.spinner(f"Fitting substrate material {base_material} data..."):
            base_fit = fit_material_data(base_df, base_material, "Substrate")
    
    if material_relations['base_sample']:
        sample_fit = base_fit
        st.info(f"Sample and substrate materials are the same, reusing substrate fitting parameters")
    elif material_relations['flyer_sample']:
        sample_fit = flyer_fit
        st.info(f"Sample and flyer materials are the same, reusing flyer fitting parameters")
    else:
        with st.spinner(f"Fitting sample material {sample_material} data..."):
            sample_fit = fit_material_data(sample_df, sample_material, "Sample")
    
    # 冲击波参数分析部分，为每种材料单独绘图
    st.subheader("Shock Wave Parameter Analysis (Hugoniot Relationships)")
    st.caption("""
    Based on linear Hugoniot relationship Us = C0 + S·Up:
    - C0: Bulk sound speed (sound speed at zero pressure, unit: km/s)
    - S: Hugoniot parameter (describes the rate of change of shock velocity with particle velocity, dimensionless)
    - Application note: Deviations may occur under high pressure (e.g., >100 GPa), phase transitions or nonlinear terms should be considered
    - Data point color coding: iml(red), ssp(blue), calculated(green), manual input(purple), bulk import(orange)
    """)
    
    # 为每种材料类型显示单独的图像
    display_material_plots(flyer_df, flyer_material, "Flyer")
    
    if not material_relations['flyer_base']:
        display_material_plots(base_df, base_material, "Substrate")
    else:
        st.info(f"Substrate and flyer materials are the same, reusing flyer shock wave images")
    
    if material_relations['base_sample'] or material_relations['flyer_sample']:
        source = "substrate" if material_relations['base_sample'] else "flyer"
        st.info(f"Sample and {source} materials are the same, reusing {source} shock wave images")
    else:
        display_material_plots(sample_df, sample_material, "Sample")
    
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
    with st.expander(f"{flyer_material} Flyer Parameters", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "Initial density",
            "rhf": "Compressed density",
            "Df": "Shock wave velocity (corresponds to Us)",
            "C0f": "Bulk sound speed (Hugoniot fit)",
            "Sf": "Hugoniot parameter S (dimensionless)",
            "E0f": "Initial internal energy density",
            "Ef": "Post-compression internal energy density",
            "uf": "Particle velocity (corresponds to Up)",
            "w": "Initial impact velocity of flyer",
            "Pf": "Impact pressure",
            "gammaf": "Grüneisen coefficient",
            "Tf": "Shock temperature (K)"
        }
        var_units = {
            "rh0f": "g/cm³",
            "rhf": "g/cm³",
            "Df": "km/s",
            "C0f": "km/s",
            "Sf": "dimensionless",
            "E0f": "GPa·cm³/g",
            "Ef": "GPa·cm³/g",
            "uf": "km/s",
            "w": "km/s",
            "Pf": "GPa",
            "gammaf": "dimensionless",
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
    
    # 基板参数 - 根据共享设置决定是否禁用输入
    disabled_base = material_relations['flyer_base']
    with st.expander(f"{base_material} Substrate Parameters {'(shared with flyer)' if disabled_base else ''}", expanded=not disabled_base):
        if disabled_base:
            st.info(f"Substrate and flyer are both {flyer_material}, will use flyer parameter values")
        
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                default_val = None
                # 如果共享参数，使用飞片的参数作为默认值
                if disabled_base:
                    flyer_var_map = {
                        "rh0b": "rh0f", "rhb": "rhf", "Db": "Df", 
                        "C0b": "C0f", "Sb": "Sf", "E0b": "E0f", 
                        "Eb": "Ef", "ub": "uf", "Pb": "Pf", 
                        "gammab": "gammaf", "Tb": "Tf"
                    }
                    flyer_equivalent = flyer_var_map.get(var)
                    if flyer_equivalent and flyer_equivalent in input_params:
                        default_val = input_params[flyer_equivalent]
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
                         "K" if var == "Tb" else "dimensionless",
                    desc="Substrate initial density" if var == "rh0b" else
                         "Substrate compressed density" if var == "rhb" else
                         "Substrate shock wave velocity" if var == "Db" else
                         "Substrate bulk sound speed" if var == "C0b" else
                         "Substrate Hugoniot parameter" if var == "Sb" else
                         "Substrate initial internal energy density" if var == "E0b" else
                         "Substrate post-compression internal energy density" if var == "Eb" else
                         "Substrate particle velocity" if var == "ub" else
                         "Substrate impact pressure" if var == "Pb" else
                         "Substrate Grüneisen coefficient" if var == "gammab" else
                         "Substrate shock temperature",
                    disabled=disabled_base
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数 - 根据共享设置决定是否禁用输入
    disabled_sample = material_relations['base_sample'] or material_relations['flyer_sample']
    share_source = "substrate" if material_relations['base_sample'] else "flyer"
    
    with st.expander(f"{sample_material} Sample Parameters {'(shared with ' + share_source + ')' if disabled_sample else ''}", expanded=not disabled_sample):
        if disabled_sample:
            st.info(f"Sample and {share_source} are both {sample_material}, will use {share_source} parameter values")
        
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                default_val = None
                # 如果共享参数，使用相应来源的参数作为默认值
                if disabled_sample:
                    source_var_map = {
                        "rh0s": "rh0b" if material_relations['base_sample'] else "rh0f",
                        "rhs": "rhb" if material_relations['base_sample'] else "rhf",
                        "Ds": "Db" if material_relations['base_sample'] else "Df",
                        "C0s": "C0b" if material_relations['base_sample'] else "C0f",
                        "Ss": "Sb" if material_relations['base_sample'] else "Sf",
                        "E0s": "E0b" if material_relations['base_sample'] else "E0f",
                        "Es": "Eb" if material_relations['base_sample'] else "Ef",
                        "us": "ub" if material_relations['base_sample'] else "uf",
                        "Ps": "Pb" if material_relations['base_sample'] else "Pf",
                        "gammas": "gammab" if material_relations['base_sample'] else "gammaf",
                        "Ts": "Tb" if material_relations['base_sample'] else "Tf"
                    }
                    source_equivalent = source_var_map.get(var)
                    if source_equivalent and source_equivalent in input_params:
                        default_val = input_params[source_equivalent]
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
                         "K" if var == "Ts" else "dimensionless",
                    desc="Sample initial density" if var == "rh0s" else
                         "Sample compressed density" if var == "rhs" else
                         "Sample shock wave velocity" if var == "Ds" else
                         "Sample bulk sound speed" if var == "C0s" else
                         "Sample Hugoniot parameter S" if var == "Ss" else
                         "Sample initial internal energy density" if var == "E0s" else
                         "Sample post-compression internal energy density" if var == "Es" else
                         "Sample particle velocity" if var == "us" else
                         "Sample impact pressure" if var == "Ps" else
                         "Sample Grüneisen coefficient" if var == "gammas" else
                         "Sample shock temperature",
                    disabled=disabled_sample
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 固定显示保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("Save Current Parameters to Database"):
            count = save_input_parameters(input_params, sample_material, "database_mode_input")
            if count > 0:
                st.success(f"Saved to {sample_material} dataset, total {count} records")
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "Maximum parameter combinations (too many will affect speed)", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("Start Solving"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} input is invalid, please check")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # 截断过多的组合
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"Too many parameter combinations ({len(combinations)}), truncated to {max_combinations} for speed")
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
                st.warning(f"Solution error: {str(e)} (may be caused by nonlinear effects under high pressure, please check parameter ranges)")
        
        if results:
            st.success(f"Solution completed, found {len(results)} solutions (results based on ideal shock wave assumptions, verification required for practical applications)")
            
            st.subheader("Result Data (units: rho=g/cm³, D=km/s, u=km/s, P=GPa, T=K)")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Result Data",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            st.subheader("Result Visualization")
            fig = plot_results_streamlit(results)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="Download Charts",
                    data=buf2,
                    file_name="analysis_with_temp_error.png",
                    mime="image/png"
                )
            
            if st.button("Save Results to Database"):
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"Saved to {sample_material} dataset, total {count} records")
    
    # 底部返回按钮
    st.markdown("---")
    if st.button("Return to Previous Page"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

def manual_mode_page():
    st.title("Manual Input Mode")
    st.write("Solve by manually inputting parameters, suitable for scenarios without database data")
    
    # 查看数据库快捷入口
    if st.button("View Database"):
        st.session_state.page = "view_database"
        st.rerun()
    
    # 材料参数输入 - 已统一为英文
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.text_input("Flyer Material Name", value="Copper", help="Enter material name, e.g.: Copper, Aluminum")
    with col2:
        base_material = st.text_input("Substrate Material Name", value="Aluminum", help="Enter material name, e.g.: Copper, Aluminum")
    with col3:
        sample_material = st.text_input("Sample Material Name", value="Copper", help="Enter material name, e.g.: Copper, Aluminum")
    
    # 检测相同材料并提供共享选项 - 修正逻辑：只有材料相同时才显示共享选项
    st.subheader("Material Parameter Sharing Settings")
    material_relations = {}
    
    # 飞片与基板是否相同
    share_flyer_base = False
    if flyer_material == base_material:
        share_flyer_base = st.checkbox(f"Flyer and substrate are both {flyer_material}, share parameters", value=True)
    material_relations['flyer_base'] = share_flyer_base
        
    # 基板与样品是否相同
    share_base_sample = False
    if base_material == sample_material:
        share_base_sample = st.checkbox(f"Substrate and sample are both {base_material}, share parameters", value=True)
    material_relations['base_sample'] = share_base_sample
        
    # 飞片与样品是否相同（仅当三者不都相同时显示）
    share_flyer_sample = False
    all_same = (flyer_material == base_material == sample_material)
    if flyer_material == sample_material and not all_same and not share_flyer_base:
        share_flyer_sample = st.checkbox(f"Flyer and sample are both {flyer_material}, share parameters", value=True)
    material_relations['flyer_sample'] = share_flyer_sample
    
    # 公共材料参数
    col1, col2 = st.columns(2)
    with col1:
        # 格吕奈森系数 - 根据共享设置决定是否需要单独设置
        if material_relations['flyer_base'] and material_relations['base_sample']:
            gamma = st.number_input("Grüneisen coefficient Γ (shared by all materials)", value=2.0, min_value=0.1, help="Approximately 2.0 for copper, 2.13 for aluminum")
            gamma_flyer = gamma_base = gamma_sample = gamma
        elif material_relations['flyer_base']:
            gamma_flyer_base = st.number_input(f"Grüneisen coefficient Γ (shared by flyer and substrate, {flyer_material})", value=2.0, min_value=0.1)
            gamma_sample = st.number_input(f"Grüneisen coefficient Γ (sample, {sample_material})", value=2.0, min_value=0.1)
            gamma_flyer = gamma_base = gamma_flyer_base
        elif material_relations['base_sample']:
            gamma_base_sample = st.number_input(f"Grüneisen coefficient Γ (shared by substrate and sample, {base_material})", value=2.0, min_value=0.1)
            gamma_flyer = st.number_input(f"Grüneisen coefficient Γ (flyer, {flyer_material})", value=2.0, min_value=0.1)
            gamma_base = gamma_sample = gamma_base_sample
        elif material_relations['flyer_sample']:
            gamma_flyer_sample = st.number_input(f"Grüneisen coefficient Γ (shared by flyer and sample, {flyer_material})", value=2.0, min_value=0.1)
            gamma_base = st.number_input(f"Grüneisen coefficient Γ (substrate, {base_material})", value=2.0, min_value=0.1)
            gamma_flyer = gamma_sample = gamma_flyer_sample
        else:
            gamma_flyer = st.number_input(f"Grüneisen coefficient Γ (flyer, {flyer_material})", value=2.0, min_value=0.1)
            gamma_base = st.number_input(f"Grüneisen coefficient Γ (substrate, {base_material})", value=2.0, min_value=0.1)
            gamma_sample = st.number_input(f"Grüneisen coefficient Γ (sample, {sample_material})", value=2.0, min_value=0.1)
    
    with col2:
        # 定容比热容 - 根据共享设置决定是否需要单独设置
        if material_relations['flyer_base'] and material_relations['base_sample']:
            Cv = st.number_input("Specific heat capacity at constant volume Cv (J/(kg·K)) (shared by all materials)", value=385, help="Approximately 385 for copper, 900 for aluminum")
            Cv_flyer = Cv_base = Cv_sample = Cv
        elif material_relations['flyer_base']:
            Cv_flyer_base = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (shared by flyer and substrate, {flyer_material})", value=385)
            Cv_sample = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (sample, {sample_material})", value=385)
            Cv_flyer = Cv_base = Cv_flyer_base
        elif material_relations['base_sample']:
            Cv_base_sample = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (shared by substrate and sample, {base_material})", value=385)
            Cv_flyer = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (flyer, {flyer_material})", value=385)
            Cv_base = Cv_sample = Cv_base_sample
        elif material_relations['flyer_sample']:
            Cv_flyer_sample = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (shared by flyer and sample, {flyer_material})", value=385)
            Cv_base = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (substrate, {base_material})", value=385)
            Cv_flyer = Cv_sample = Cv_flyer_sample
        else:
            Cv_flyer = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (flyer, {flyer_material})", value=385)
            Cv_base = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (substrate, {base_material})", value=385)
            Cv_sample = st.number_input(f"Specific heat capacity at constant volume Cv (J/(kg·K)) (sample, {sample_material})", value=385)
    
    exp_method = st.text_input("Experimental method/data source", value="manual_input", help="Record data source, e.g.: iml, ssp, experimental equipment, literature, etc.")
    
    # 参数输入
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片参数
    with st.expander(f"{flyer_material} Flyer Parameters", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["f"]):
            with cols[i % 3]:
                default_val = gamma_flyer if var == "gammaf" else None
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=var,
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Df", "C0f", "uf", "w"] else 
                         "GPa·cm³/g" if var in ["E0f", "Ef"] else
                         "GPa" if var == "Pf" else
                         "K" if var == "Tf" else "dimensionless",
                    desc="Flyer initial density" if var == "rh0f" else
                         "Flyer compressed density" if var == "rhf" else
                         "Flyer shock wave velocity" if var == "Df" else
                         "Flyer bulk sound speed" if var == "C0f" else
                         "Flyer Hugoniot parameter S" if var == "Sf" else
                         "Flyer initial internal energy density" if var == "E0f" else
                         "Flyer post-compression internal energy density" if var == "Ef" else
                         "Flyer particle velocity" if var == "uf" else
                         "Flyer initial impact velocity" if var == "w" else
                         "Flyer impact pressure" if var == "Pf" else
                         "Flyer Grüneisen coefficient" if var == "gammaf" else
                         "Flyer shock temperature"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数输入 - 根据共享设置决定是否禁用
    disabled_base = material_relations['flyer_base']
    with st.expander(f"{base_material} Substrate Parameters {'(shared with flyer)' if disabled_base else ''}", expanded=not disabled_base):
        if disabled_base:
            st.info(f"Substrate and flyer are both {flyer_material}, will use flyer parameter values")
        
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                default_val = None
                # 如果共享参数，使用飞片的参数作为默认值
                if disabled_base:
                    flyer_var_map = {
                        "rh0b": "rh0f", "rhb": "rhf", "Db": "Df", 
                        "C0b": "C0f", "Sb": "Sf", "E0b": "E0f", 
                        "Eb": "Ef", "ub": "uf", "Pb": "Pf", 
                        "gammab": "gammaf", "Tb": "Tf"
                    }
                    flyer_equivalent = flyer_var_map.get(var)
                    if flyer_equivalent and flyer_equivalent in input_params:
                        default_val = input_params[flyer_equivalent]
                else:
                    default_val = gamma_base if var == "gammab" else None
                
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"b_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Db", "C0b", "ub"] else 
                         "GPa·cm³/g" if var in ["E0b", "Eb"] else
                         "GPa" if var == "Pb" else
                         "K" if var == "Tb" else "dimensionless",
                    desc="Substrate initial density" if var == "rh0b" else
                         "Substrate compressed density" if var == "rhb" else
                         "Substrate shock wave velocity" if var == "Db" else
                         "Substrate bulk sound speed" if var == "C0b" else
                         "Substrate Hugoniot parameter" if var == "Sb" else
                         "Substrate initial internal energy density" if var == "E0b" else
                         "Substrate post-compression internal energy density" if var == "Eb" else
                         "Substrate particle velocity" if var == "ub" else
                         "Substrate impact pressure" if var == "Pb" else
                         "Substrate Grüneisen coefficient" if var == "gammab" else
                         "Substrate shock temperature",
                    disabled=disabled_base
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数输入 - 根据共享设置决定是否禁用
    disabled_sample = material_relations['base_sample'] or material_relations['flyer_sample']
    share_source = "substrate" if material_relations['base_sample'] else "flyer"
    
    with st.expander(f"{sample_material} Sample Parameters {'(shared with ' + share_source + ')' if disabled_sample else ''}", expanded=not disabled_sample):
        if disabled_sample:
            st.info(f"Sample and {share_source} are both {sample_material}, will use {share_source} parameter values")
        
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                default_val = None
                # 如果共享参数，使用相应来源的参数作为默认值
                if disabled_sample:
                    source_var_map = {
                        "rh0s": "rh0b" if material_relations['base_sample'] else "rh0f",
                        "rhs": "rhb" if material_relations['base_sample'] else "rhf",
                        "Ds": "Db" if material_relations['base_sample'] else "Df",
                        "C0s": "C0b" if material_relations['base_sample'] else "C0f",
                        "Ss": "Sb" if material_relations['base_sample'] else "Sf",
                        "E0s": "E0b" if material_relations['base_sample'] else "E0f",
                        "Es": "Eb" if material_relations['base_sample'] else "Ef",
                        "us": "ub" if material_relations['base_sample'] else "uf",
                        "Ps": "Pb" if material_relations['base_sample'] else "Pf",
                        "gammas": "gammab" if material_relations['base_sample'] else "gammaf",
                        "Ts": "Tb" if material_relations['base_sample'] else "Tf"
                    }
                    source_equivalent = source_var_map.get(var)
                    if source_equivalent and source_equivalent in input_params:
                        default_val = input_params[source_equivalent]
                else:
                    default_val = gamma_sample if var == "gammas" else None
                
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"s_{var}",
                    default=default_val,
                    unit="g/cm³" if var.startswith("rh") else 
                         "km/s" if var in ["Ds", "C0s", "us"] else 
                         "GPa·cm³/g" if var in ["E0s", "Es"] else
                         "GPa" if var == "Ps" else
                         "K" if var == "Ts" else "dimensionless",
                    desc="Sample initial density" if var == "rh0s" else
                         "Sample compressed density" if var == "rhs" else
                         "Sample shock wave velocity" if var == "Ds" else
                         "Sample bulk sound speed" if var == "C0s" else
                         "Sample Hugoniot parameter S" if var == "Ss" else
                         "Sample initial internal energy density" if var == "E0s" else
                         "Sample post-compression internal energy density" if var == "Es" else
                         "Sample particle velocity" if var == "us" else
                         "Sample impact pressure" if var == "Ps" else
                         "Sample Grüneisen coefficient" if var == "gammas" else
                         "Sample shock temperature",
                    disabled=disabled_sample
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 固定显示保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("Save Current Parameters to Database"):
            count = save_input_parameters(input_params, sample_material, exp_method)
            if count > 0:
                st.success(f"Saved to {sample_material} dataset, total {count} records")
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "Maximum parameter combinations (too many will affect speed)", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("Start Solving Equations"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} input is invalid, please check")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # 截断过多的组合
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"Too many parameter combinations ({len(combinations)}), truncated to {max_combinations} for speed")
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
                    st.warning("Flyer parameter error: Df (shock wave velocity) must be greater than uf (particle velocity)")
                    continue
                if current_subs.get(sym_vars['Db'], 0) <= current_subs.get(sym_vars['ub'], 0):
                    st.warning("Substrate parameter error: Db (shock wave velocity) must be greater than ub (particle velocity)")
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
                st.warning(f"Solution error: {str(e)} (may be due to parameter range exceeding model applicability)")
        
        if results:
            st.success(f"Solution completed, found {len(results)} solutions")
            
            st.subheader("Result Data")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Result Data",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            st.subheader("Result Visualization")
            fig = plot_results_streamlit(results)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="Download Charts",
                    data=buf2,
                    file_name="analysis_with_temp_error.png",
                    mime="image/png"
                )
            
            if st.button("Save Calculation Results to Database"):
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"Saved to {sample_material} dataset, total {count} records")
        else:
            st.warning("No valid solutions found (please check if parameters conform to physical laws, e.g.: shock wave velocity > particle velocity)")
    
    # 底部返回按钮
    st.markdown("---")
    if st.button("Return to Previous Page"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # 初始化会话状态变量
    if 'confirm_delete' not in st.session_state:
        st.session_state['confirm_delete'] = False
    if 'confirm_clear' not in st.session_state:
        st.session_state['confirm_clear'] = False
    if 'calculation_result' not in st.session_state:
        st.session_state.calculation_result = None
    
    st.set_page_config(
        page_title="Shock Wave Parameter Calculation and Analysis System",
        page_icon="✨",
        layout="wide"
    )
    
    # 数据库查看页面
    if st.session_state.page == "view_database":
        st.title("Database View and Management")
        view_database()
        # 底部返回按钮
        st.markdown("---")
        if st.button("Return to Previous Page"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "database_mode":
        database_mode_page()
    elif st.session_state.page == "manual_mode":
        manual_mode_page()

if __name__ == "__main__":
    main()
