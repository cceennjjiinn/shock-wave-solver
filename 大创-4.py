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
        return 1
    except Exception as e:
        st.error(f"保存输入数据失败: {str(e)}")
        return 0

# 新增：批量导入数据到数据库
def bulk_import_data(df, material_name, exp_method="bulk_import"):
    """批量导入数据到数据库，返回成功导入的记录数"""
    if df.empty:
        return 0
        
    required_columns = ['rho0', 'Us', 'Up']  # 至少需要这三个参数
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"导入失败：CSV文件缺少必要的列: {', '.join(missing_cols)}")
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

# 新增：批量删除选中的记录
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

# 新增：清空指定材料的所有数据
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
    """显示数据库内容，新增批量添加和删除功能"""
    with st.expander("数据库内容", expanded=True):
        # 批量操作区域
        st.subheader("批量数据操作")
        col1, col2 = st.columns(2)
        
        # 批量导入部分
        with col1:
            st.subheader("批量导入数据")
            new_material = st.text_input("材料名称", help="输入要导入数据的材料名称")
            uploaded_file = st.file_uploader("选择CSV文件", type="csv")
            exp_method = st.text_input("实验方法/数据来源", value="bulk_import")
            
            if st.button("导入数据"):
                if not new_material:
                    st.error("请输入材料名称")
                elif uploaded_file is None:
                    st.error("请选择CSV文件")
                else:
                    # 读取CSV文件
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"成功读取CSV文件，包含 {len(df)} 条记录")
                        st.dataframe(df.head())  # 显示前几行预览
                        
                        # 导入数据
                        count = bulk_import_data(df, new_material, exp_method)
                        if count > 0:
                            st.success(f"成功导入 {count} 条记录（跳过包含空值的行）")
                            # 刷新数据
                            st.rerun()  # 修改：使用st.rerun()替代st.experimental_rerun()
                        else:
                            st.warning("没有导入任何记录，请检查数据格式")
                    except Exception as e:
                        st.error(f"读取CSV文件失败: {str(e)}")
        
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
                        # 移除：selection_mode参数在新版本中已不支持
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
                                        st.rerun()  # 修改：使用st.rerun()替代st.experimental_rerun()
                                    else:
                                        st.warning("删除失败或没有记录被删除")
                                else:
                                    st.warning("请确认删除操作")
                                    st.session_state['confirm_delete'] = True
                                    st.rerun()  # 修改：使用st.rerun()替代st.experimental_rerun()
                            else:
                                st.warning("请先选择要删除的记录")
                    
                    with col_del2:
                        if st.button("清空该材料所有数据"):
                            if st.session_state.get('confirm_clear', False):
                                deleted = clear_material_data(del_material)
                                if deleted > 0:
                                    st.success(f"成功清空 {del_material} 的所有 {deleted} 条记录")
                                    st.session_state['confirm_clear'] = False
                                    st.rerun()  # 修改：使用st.rerun()替代st.experimental_rerun()
                                else:
                                    st.warning("清空失败或该材料没有数据")
                            else:
                                st.warning("此操作将删除该材料所有数据，请确认")
                                st.session_state['confirm_clear'] = True
                                st.rerun()  # 修改：使用st.rerun()替代st.experimental_rerun()
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
                label=f"下载 {selected_material} 数据",
                data=csv,
                file_name=f"{selected_material}_data.csv",
                mime="text/csv",
            )
            
            # 提供CSV模板下载，方便用户按格式准备数据
            if st.button("下载数据导入模板"):
                template = pd.DataFrame(columns=[
                    'rho0', 'Us', 'Up', 'P', 'V', 'rho', 
                    'V_V0', 'gamma', 'T'
                ])
                template.loc[0] = [8.96, 5.0, 1.0, 44.8, 0.089, 11.2, 0.8, 2.0, 3000]
                csv = template.to_csv(index=False)
                st.download_button(
                    label="下载CSV模板",
                    data=csv,
                    file_name="shock_wave_data_template.csv",
                    mime="text/csv",
                    on_click=lambda: st.success("模板已准备好下载")
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
    
    # 按实验方法统计数据
    if 'exp_method' in df.columns:
        method_counts = df['exp_method'].value_counts()
        st.info(f"实验方法分布: {', '.join([f'{k}: {v}条' for k, v in method_counts.items()])}")
    
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
    st.caption(f"{desc} | 单位: {unit}")
    input_type = st.radio(
        f"{label}输入类型",
        ["单一值", "多个值（逗号分隔）", "范围（可指定步长）"],
        key=f"{key}_type",
        horizontal=True,
        disabled=disabled
    )
    
    default_val = str(default) if default is not None else ""
    
    if input_type == "单一值":
        val = st.text_input(label, default_val, key=f"{key}_single", disabled=disabled)
        if val == "":
            return symbols(var_name)
        try:
            return [float(val)]
        except ValueError:
            st.error("请输入有效的数字")
            return None
    elif input_type == "多个值（逗号分隔）":
        val = st.text_input(label, default_val, key=f"{key}_multi", disabled=disabled)
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
            start = st.text_input(f"{label}起始值", default_val, key=f"{key}_start", disabled=disabled)
        with col2:
            end = st.text_input(f"{label}结束值", "", key=f"{key}_end", disabled=disabled)
        with col3:
            step = st.text_input(f"{label}步长（可选）", "0.5", key=f"{key}_step", disabled=disabled)
            
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

# 冲击波关系图绘制 - 根据实验方法区分颜色
@st.cache_data(ttl=3600)  # 缓存图像结果
def generate_shock_plots(df, C0, S, material_name, material_type, use_english=False):
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
        'bulk_import': 'orange'  # 新增：批量导入数据的颜色标识
    }
    default_color = 'gray'  # 未定义的实验方法用灰色
    
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
            label=f'{method}' if not use_english else f'{method}',
            color=color, alpha=0.7
        )
    
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
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 0].scatter(
            method_df['V_V0'], method_df['P'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7
        )
    
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
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 1].scatter(
            method_df['P'], method_df['rho'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7
        )
    
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
    
    # 查看数据库快捷入口
    if st.button("查看数据库"):
        st.session_state.page = "view_database"
        st.rerun()
    
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
    
    # 查看数据库快捷入口
    if st.button("查看数据库"):
        st.session_state.page = "view_database"
        st.rerun()
    
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
    
    # 检测相同材料并提供共享选项
    st.subheader("材料参数共享设置")
    material_relations = {}
    
    # 飞片与基板是否相同
    if flyer_material == base_material:
        share_flyer_base = st.checkbox(f"飞片与基板均为{flyer_material}，共享参数", value=True)
        material_relations['flyer_base'] = share_flyer_base
    else:
        material_relations['flyer_base'] = False
        
    # 基板与样品是否相同
    if base_material == sample_material:
        share_base_sample = st.checkbox(f"基板与样品均为{base_material}，共享参数", value=True)
        material_relations['base_sample'] = share_base_sample
    else:
        material_relations['base_sample'] = False
        
    # 飞片与样品是否相同
    if flyer_material == sample_material and not material_relations.get('flyer_base', False) or not material_relations.get('base_sample', False):
        share_flyer_sample = st.checkbox(f"飞片与样品均为{flyer_material}，共享参数", value=True)
        material_relations['flyer_sample'] = share_flyer_sample
    else:
        material_relations['flyer_sample'] = False
    
    # 按需查询字段以减少数据传输，确保包含exp_method
    flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method'])
    
    # 根据共享设置决定是否复用数据
    if material_relations['flyer_base']:
        base_df = flyer_df.copy()
    else:
        base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method'])
    
    if material_relations['base_sample'] and not material_relations['flyer_base']:
        sample_df = base_df.copy()
    elif material_relations['flyer_sample']:
        sample_df = flyer_df.copy()
    else:
        sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method'])
    
    # 为每种材料类型拟合数据并清晰标注
    with st.spinner(f"正在拟合飞片材料{flyer_material}的数据..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material, "飞片")
    
    # 根据共享设置决定是否复用拟合结果
    if material_relations['flyer_base']:
        base_fit = flyer_fit
        st.info(f"基板与飞片材料相同，复用飞片的拟合参数")
    else:
        with st.spinner(f"正在拟合基板材料{base_material}的数据..."):
            base_fit = fit_material_data(base_df, base_material, "基板")
    
    if material_relations['base_sample'] and not material_relations['flyer_base']:
        sample_fit = base_fit
        st.info(f"样品与基板材料相同，复用基板的拟合参数")
    elif material_relations['flyer_sample']:
        sample_fit = flyer_fit
        st.info(f"样品与飞片材料相同，复用飞片的拟合参数")
    else:
        with st.spinner(f"正在拟合样品材料{sample_material}的数据..."):
            sample_fit = fit_material_data(sample_df, sample_material, "样品")
    
    # 冲击波参数分析部分，为每种材料单独绘图
    st.subheader("冲击波参数分析（Hugoniot关系）")
    st.caption("""
    基于线性Hugoniot关系Us = C0 + S·Up进行分析，其中：
    - C0：材料的体声速（零压状态下的声速，单位km/s）
    - S：Hugoniot参数（描述冲击波速度随粒子速度的变化率，无量纲）
    - 应用说明：在高压下（如>100 GPa）可能出现偏差，需考虑相变或非线性项
    - 数据点颜色区分：iml(红色)、ssp(蓝色)、计算值(绿色)、手动输入(紫色)、批量导入(橙色)
    """)
    
    # 为每种材料类型显示单独的图像，数据库模式下设置use_english=True
    display_material_plots(flyer_df, flyer_material, "飞片", use_english=True)
    
    if not material_relations['flyer_base']:
        display_material_plots(base_df, base_material, "基板", use_english=True)
    else:
        st.info(f"基板与飞片材料相同，复用飞片的冲击波图像")
    
    if (material_relations['base_sample'] and not material_relations['flyer_base']) or material_relations['flyer_sample']:
        st.info(f"样品与{'基板' if material_relations['base_sample'] else '飞片'}材料相同，复用{'基板' if material_relations['base_sample'] else '飞片'}的冲击波图像")
    else:
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
    
    # 基板参数 - 根据共享设置决定是否禁用输入
    disabled_base = material_relations['flyer_base']
    with st.expander(f"{base_material}基板参数 {'(与飞片共享参数)' if disabled_base else ''}", expanded=not disabled_base):
        if disabled_base:
            st.info(f"基板与飞片均为{flyer_material}，将使用飞片的参数值")
        
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
                
                if not disabled_base:
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
                         "冲击温度",
                    disabled=disabled_base
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数 - 根据共享设置决定是否禁用输入
    disabled_sample = material_relations['base_sample'] and not material_relations['flyer_base'] or material_relations['flyer_sample']
    share_source = "基板" if (material_relations['base_sample'] and not material_relations['flyer_base']) else "飞片"
    
    with st.expander(f"{sample_material}样品参数 {'(与' + share_source + '共享参数)' if disabled_sample else ''}", expanded=not disabled_sample):
        if disabled_sample:
            st.info(f"样品与{share_source}均为{sample_material}，将使用{share_source}的参数值")
        
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
                
                if not disabled_sample:
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
                         "冲击温度",
                    disabled=disabled_sample
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 固定显示保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("保存当前参数到数据库"):
            count = save_input_parameters(input_params, sample_material, "database_mode_input")
            if count > 0:
                st.success(f"已保存到材料 {sample_material} 的数据集，共 {count} 条记录")
    
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
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"已保存到材料 {sample_material} 的数据集，共 {count} 条记录")
    
    if st.button("返回首页"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

def manual_mode_page():
    st.title("手动输入模式")
    st.write("通过手动输入参数进行求解，适用于没有数据库数据的场景")
    
    # 查看数据库快捷入口
    if st.button("查看数据库"):
        st.session_state.page = "view_database"
        st.rerun()
    
    # 材料参数输入
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.text_input("飞片材料名称", value="铜", help="输入材料名称，如：铜、铝等")
    with col2:
        base_material = st.text_input("基板材料名称", value="铝", help="输入材料名称，如：铜、铝等")
    with col3:
        sample_material = st.text_input("样品材料名称", value="铜", help="输入材料名称，如：铜、铝等")
    
    # 检测相同材料并提供共享选项
    st.subheader("材料参数共享设置")
    material_relations = {}
    
    # 飞片与基板是否相同
    if flyer_material == base_material:
        share_flyer_base = st.checkbox(f"飞片与基板均为{flyer_material}，共享参数", value=True)
        material_relations['flyer_base'] = share_flyer_base
    else:
        material_relations['flyer_base'] = False
        
    # 基板与样品是否相同
    if base_material == sample_material:
        share_base_sample = st.checkbox(f"基板与样品均为{base_material}，共享参数", value=True)
        material_relations['base_sample'] = share_base_sample
    else:
        material_relations['base_sample'] = False
        
    # 飞片与样品是否相同
    if flyer_material == sample_material and not (material_relations.get('flyer_base', False) and material_relations.get('base_sample', False)):
        share_flyer_sample = st.checkbox(f"飞片与样品均为{flyer_material}，共享参数", value=True)
        material_relations['flyer_sample'] = share_flyer_sample
    else:
        material_relations['flyer_sample'] = False
    
    # 公共材料参数
    col1, col2 = st.columns(2)
    with col1:
        # 格吕奈森系数 - 根据共享设置决定是否需要单独设置
        if material_relations['flyer_base'] and material_relations['base_sample']:
            gamma = st.number_input("格吕奈森系数Γ（所有材料共用）", value=2.0, min_value=0.1, help="铜约为2.0，铝约为2.13")
            gamma_flyer = gamma_base = gamma_sample = gamma
        elif material_relations['flyer_base']:
            gamma_flyer_base = st.number_input(f"格吕奈森系数Γ（飞片与基板共用，{flyer_material}）", value=2.0, min_value=0.1)
            gamma_sample = st.number_input(f"格吕奈森系数Γ（样品，{sample_material}）", value=2.0, min_value=0.1)
            gamma_flyer = gamma_base = gamma_flyer_base
        elif material_relations['base_sample']:
            gamma_base_sample = st.number_input(f"格吕奈森系数Γ（基板与样品共用，{base_material}）", value=2.0, min_value=0.1)
            gamma_flyer = st.number_input(f"格吕奈森系数Γ（飞片，{flyer_material}）", value=2.0, min_value=0.1)
            gamma_base = gamma_sample = gamma_base_sample
        elif material_relations['flyer_sample']:
            gamma_flyer_sample = st.number_input(f"格吕奈森系数Γ（飞片与样品共用，{flyer_material}）", value=2.0, min_value=0.1)
            gamma_base = st.number_input(f"格吕奈森系数Γ（基板，{base_material}）", value=2.0, min_value=0.1)
            gamma_flyer = gamma_sample = gamma_flyer_sample
        else:
            gamma_flyer = st.number_input(f"格吕奈森系数Γ（飞片，{flyer_material}）", value=2.0, min_value=0.1)
            gamma_base = st.number_input(f"格吕奈森系数Γ（基板，{base_material}）", value=2.0, min_value=0.1)
            gamma_sample = st.number_input(f"格吕奈森系数Γ（样品，{sample_material}）", value=2.0, min_value=0.1)
    
    with col2:
        # 定容比热容 - 根据共享设置决定是否需要单独设置
        if material_relations['flyer_base'] and material_relations['base_sample']:
            Cv = st.number_input("定容比热容Cv (J/(kg·K))（所有材料共用）", value=385, help="铜约为385，铝约为900")
            Cv_flyer = Cv_base = Cv_sample = Cv
        elif material_relations['flyer_base']:
            Cv_flyer_base = st.number_input(f"定容比热容Cv (J/(kg·K))（飞片与基板共用，{flyer_material}）", value=385)
            Cv_sample = st.number_input(f"定容比热容Cv (J/(kg·K))（样品，{sample_material}）", value=385)
            Cv_flyer = Cv_base = Cv_flyer_base
        elif material_relations['base_sample']:
            Cv_base_sample = st.number_input(f"定容比热容Cv (J/(kg·K))（基板与样品共用，{base_material}）", value=385)
            Cv_flyer = st.number_input(f"定容比热容Cv (J/(kg·K))（飞片，{flyer_material}）", value=385)
            Cv_base = Cv_sample = Cv_base_sample
        elif material_relations['flyer_sample']:
            Cv_flyer_sample = st.number_input(f"定容比热容Cv (J/(kg·K))（飞片与样品共用，{flyer_material}）", value=385)
            Cv_base = st.number_input(f"定容比热容Cv (J/(kg·K))（基板，{base_material}）", value=385)
            Cv_flyer = Cv_sample = Cv_flyer_sample
        else:
            Cv_flyer = st.number_input(f"定容比热容Cv (J/(kg·K))（飞片，{flyer_material}）", value=385)
            Cv_base = st.number_input(f"定容比热容Cv (J/(kg·K))（基板，{base_material}）", value=385)
            Cv_sample = st.number_input(f"定容比热容Cv (J/(kg·K))（样品，{sample_material}）", value=385)
    
    exp_method = st.text_input("实验方法/数据来源", value="manual_input", help="记录数据来源，如：iml、ssp、实验设备、文献等")
    
    # 冲击波参数快速计算
    st.subheader("冲击波参数快速计算")
    st.caption("""
    基于Rankine-Hugoniot守恒方程组，适用于理想平面冲击波：
    - 公式：P = ρ0·Us·Up，ρ = ρ0·Us/(Us-Up)，V/V0 = 1 - Up/Us
    - 输入要求：Us > Up（冲击波速度必须大于粒子速度）
    - 单位：ρ0(g/cm³)，Us(km/s)，Up(km/s) → 输出P(GPa)
    """)
    
    # 选择要计算的材料
    calc_material = st.selectbox("选择要计算的材料", [flyer_material, base_material, sample_material])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        U_s = st.number_input("冲击波速度Us (km/s)", min_value=0.01, value=5.0, help="必须大于粒子速度Up")
        Us_err = st.number_input("Us误差 (km/s)", 0.1, help="测量误差")
    with col2:
        u_p = st.number_input("粒子速度Up (km/s)", min_value=0.0, value=1.0, help="必须小于冲击波速度Us")
        Up_err = st.number_input("Up误差 (km/s)", 0.05, help="测量误差")
    with col3:
        # 根据所选材料自动选择密度和相关参数
        if calc_material == flyer_material:
            rho0 = st.number_input(f"初始密度ρ0 (g/cm³) - {flyer_material}", min_value=0.01, value=8.96)
            gamma = gamma_flyer
            Cv = Cv_flyer
        elif calc_material == base_material:
            rho0 = st.number_input(f"初始密度ρ0 (g/cm³) - {base_material}", min_value=0.01, value=2.7)
            gamma = gamma_base
            Cv = Cv_base
        else:
            rho0 = st.number_input(f"初始密度ρ0 (g/cm³) - {sample_material}", min_value=0.01, value=8.96)
            gamma = gamma_sample
            Cv = Cv_sample
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
        if st.button("保存快速计算结果到数据库"):
            count = save_input_data_to_db(calculation_result, calc_material, exp_method)
            if count > 0:
                st.success(f"已保存到材料 {calc_material} 的数据集，共 {count} 条记录")
    
    # 参数输入
    # 参数输入
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
    
    # 基板参数输入 - 根据共享设置决定是否禁用
    disabled_base = material_relations['flyer_base']
    with st.expander(f"{base_material}基板参数 {'(与飞片共享参数)' if disabled_base else ''}", expanded=not disabled_base):
        if disabled_base:
            st.info(f"基板与飞片均为{flyer_material}，将使用飞片的参数值")
        
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
                         "基板冲击温度",
                    disabled=disabled_base
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数输入 - 根据共享设置决定是否禁用
    disabled_sample = material_relations['base_sample'] or material_relations['flyer_sample']
    share_source = "基板" if material_relations['base_sample'] else "飞片"
    
    with st.expander(f"{sample_material}样品参数 {'(与' + share_source + '共享参数)' if disabled_sample else ''}", expanded=not disabled_sample):
        if disabled_sample:
            st.info(f"样品与{share_source}均为{sample_material}，将使用{share_source}的参数值")
        
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
                         "样品冲击温度",
                    disabled=disabled_sample
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 固定显示保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("保存当前参数到数据库"):
            count = save_input_parameters(input_params, sample_material, exp_method)
            if count > 0:
                st.success(f"已保存到材料 {sample_material} 的数据集，共 {count} 条记录")
    
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
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"已保存到材料 {sample_material} 的数据集，共 {count} 条记录")
        else:
            st.warning("未找到有效解（请检查参数是否符合物理规律，如冲击波速度>粒子速度）")
    
    if st.button("返回首页"):
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
    
    st.set_page_config(
        page_title="冲击波参数计算与分析系统",
        page_icon="✨",
        layout="wide"
    )
    
    # 数据库查看页面
    if st.session_state.page == "view_database":
        st.title("数据库查看与管理")
        view_database()
        if st.button("返回首页"):
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
    
