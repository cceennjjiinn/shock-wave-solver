import streamlit as st
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sympy import symbols, Eq, solve, simplify
from scipy.optimize import least_squares
from io import BytesIO, StringIO
import itertools
import os
import tempfile  # 新增：用于处理临时文件

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建SQLite引擎 - 使用临时目录避免权限问题
try:
    # 优先使用临时目录存储数据库，避免权限问题
    temp_dir = tempfile.gettempdir()
    sqlite_path = os.path.join(temp_dir, 'shock_wave_data.db')
    sqlite_engine = create_engine(
        f'sqlite:///{sqlite_path}',
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600
    )
except Exception as e:
    st.error(f"数据库引擎初始化失败: {str(e)}")
    # 作为备选方案，使用内存数据库
    sqlite_engine = create_engine('sqlite:///:memory:')

# SQLite性能优化
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA synchronous=NORMAL;')
        cursor.execute('PRAGMA temp_store=MEMORY;')
        cursor.execute('PRAGMA cache_size=-20000;')
        cursor.close()
    except Exception as e:
        st.warning(f"SQLite优化设置失败: {str(e)}")

# 初始化数据库
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
                        T REAL           -- 冲击温度 (K)
                    )
                """))
                conn.commit()
    except Exception as e:
        st.error(f"数据库初始化失败: {str(e)}")

# 修复数据库表结构
def fix_database_schema():
    try:
        with sqlite_engine.connect() as conn:
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(shock_wave_all_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'gamma' not in columns:
                conn.execute(text("ALTER TABLE shock_wave_all_data ADD COLUMN gamma REAL"))
                conn.commit()
    except Exception as e:
        st.warning(f"修复数据库表结构失败: {str(e)}")

# 初始化数据库
init_database()
fix_database_schema()

# 数据库操作函数
@st.cache_data(ttl=3600)
def get_all_materials():
    try:
        query = text("SELECT DISTINCT material FROM shock_wave_all_data")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['material'].tolist() if not df.empty else []
    except Exception as e:
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material_name, fields=None):
    try:
        if fields is None:
            fields = '*'
        else:
            if 'exp_method' not in fields:
                fields.append('exp_method')
            fields = ', '.join(fields)
            
        query = text(f"SELECT {fields} FROM shock_wave_all_data WHERE material = :material")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'material': material_name})
        return df
    except Exception as e:
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    if not results:
        return 0
        
    try:
        count = 0
        with sqlite_engine.begin() as conn:
            for result in results:
                required_params = ['rh0f', 'Df', 'uf', 'Pf']
                if not all(param in result for param in required_params):
                    continue
                    
                # 确保所有值都是数值类型
                def safe_float(value, default=0.0):
                    try:
                        return float(value)
                    except:
                        return default
                        
                data = {
                    'material': material_name,
                    'rho0': safe_float(result.get('rh0f')),
                    'Us': safe_float(result.get('Df')),
                    'Up': safe_float(result.get('uf')),
                    'P': safe_float(result.get('Pf')),
                    'V': safe_float(result.get('V')),
                    'rho': safe_float(result.get('rhf')),
                    'V_V0': safe_float(result.get('V_V0')),
                    'exp_method': 'calculated',
                    'gamma': safe_float(result.get('gammaf')),
                    'T': safe_float(result.get('Tf'))
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
    try:
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except:
                return default
                
        data = {
            'material': material_name,
            'rho0': safe_float(input_params.get('rh0f')),
            'Us': safe_float(input_params.get('Df')),
            'Up': safe_float(input_params.get('uf')),
            'P': safe_float(input_params.get('Pf')),
            'V': 0,
            'rho': safe_float(input_params.get('rhf')),
            'V_V0': 0,
            'exp_method': exp_method,
            'gamma': safe_float(input_params.get('gammaf')),
            'T': safe_float(input_params.get('Tf'))
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

def bulk_import_data(df, material_name, exp_method="bulk_import"):
    if df.empty:
        return 0
        
    required_columns = ['rho0', 'Us', 'Up']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"导入失败：CSV文件缺少必要的列: {', '.join(missing_cols)}")
        return 0
        
    try:
        count = 0
        with sqlite_engine.begin() as conn:
            for _, row in df.iterrows():
                if row[required_columns].isnull().any():
                    continue
                    
                def safe_float(value, default=0.0):
                    try:
                        return float(value)
                    except:
                        return default
                
                data = {
                    'material': material_name,
                    'rho0': safe_float(row.get('rho0')),
                    'Us': safe_float(row.get('Us')),
                    'Up': safe_float(row.get('Up')),
                    'P': safe_float(row.get('P')),
                    'V': safe_float(row.get('V')),
                    'rho': safe_float(row.get('rho')),
                    'V_V0': safe_float(row.get('V_V0')),
                    'exp_method': exp_method,
                    'gamma': safe_float(row.get('gamma')),
                    'T': safe_float(row.get('T'))
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

def bulk_delete_records(ids):
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

def clear_material_data(material_name):
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
    with st.expander("数据库内容", expanded=True):
        st.subheader("批量数据操作")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("批量导入数据")
            new_material = st.text_input("材料名称 (Material Name)", help="输入要导入数据的材料名称")
            uploaded_file = st.file_uploader("选择CSV文件", type="csv")
            exp_method = st.text_input("实验方法/数据来源", value="bulk_import")
            
            if st.button("导入数据"):
                if not new_material:
                    st.error("请输入材料名称")
                elif uploaded_file is None:
                    st.error("请选择CSV文件")
                else:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"成功读取CSV文件，包含 {len(df)} 条记录")
                        st.dataframe(df.head())
                        
                        count = bulk_import_data(df, new_material, exp_method)
                        if count > 0:
                            st.success(f"成功导入 {count} 条记录")
                            st.rerun()
                        else:
                            st.warning("没有导入任何记录，请检查数据格式")
                    except Exception as e:
                        st.error(f"读取CSV文件失败: {str(e)}")
        
        with col2:
            st.subheader("批量删除数据")
            materials = get_all_materials()
            if materials:
                del_material = st.selectbox("选择要操作的材料", materials, key="del_material")
                
                df = get_material_data(del_material)
                if not df.empty and 'id' in df.columns:
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
            
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"下载 {selected_material} 数据",
                data=csv,
                file_name=f"{selected_material}_data.csv",
                mime="text/csv",
            )
            
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

# 冲击波参数计算
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300, calculate_temp=True):
    try:
        P = rho0 * U_s * u_p
        V = (1 / rho0) * (1 - u_p / U_s)
        rho = rho0 * U_s / (U_s - u_p)
        V_V0 = V * rho0
        
        T = None
        if calculate_temp:
            E_shock = 0.5 * P * (1/rho0 - V) * 1e6
            T = T0 + (E_shock) / (Cv * (1 + gamma/2))
        
        return P, V, rho, V_V0, T
    except Exception as e:
        st.error(f"参数计算错误: {str(e)}")
        return None, None, None, None, None

def fit_hugoniot(df):
    try:
        df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
        if len(df) < 2:
            return 0, 0
            
        U_s = df['Us'].values
        u_p = df['Up'].values
        coeffs = np.polyfit(u_p, U_s, 1)
        return coeffs[1], coeffs[0]
    except Exception as e:
        st.warning(f"Hugoniot拟合失败: {str(e)}")
        return 0, 0

@st.cache_data(ttl=3600)
def fit_material_data(df, material_name, material_type):
    try:
        if df is None or df.empty:
            st.warning(f"{material_type}材料 '{material_name}' 没有数据")
            return None
        
        df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
        if len(df) < 2:
            st.warning(f"{material_type}材料 '{material_name}' 有效数据不足")
            return None
        
        X = df['Up'].values.reshape(-1, 1)
        y = df['Us'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        C0 = model.intercept_
        S = model.coef_[0]
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = np.mean(np.abs(y - y_pred))
        
        st.info(f"{material_type}材料 {material_name} 拟合结果: Us = {C0:.4f} + {S:.4f}*Up")
        st.info(f"拟合误差: R² = {r2:.4f}, RMSE = {rmse:.4f} km/s, MAE = {mae:.4f} km/s")
        
        return {
            "C0": C0, "S": S, "rho0": df['rho0'].mean(),
            "r2": r2, "rmse": rmse, "mae": mae
        }
    except Exception as e:
        st.error(f"材料数据拟合错误: {str(e)}")
        return None

def calculate_error(params, param_errors):
    try:
        rho0, Us, Up = params['rho0'], params['Us'], params['Up']
        rho0_err, Us_err, Up_err = param_errors['rho0'], param_errors['Us'], param_errors['Up']
        
        P_rel_err = (rho0_err/rho0)**2 + (Us_err/Us)** 2 + (Up_err/Up)**2
        P_err = rho0*Us*Up * np.sqrt(P_rel_err)
        Us_err = np.sqrt(Us_err**2 + (0.01*Us)** 2)
        
        return {
            "P_err": P_err,
            "Us_err": Us_err,
            "Up_err": Up_err
        }
    except Exception as e:
        st.warning(f"误差计算失败: {str(e)}")
        return {"P_err": 0, "Us_err": 0, "Up_err": 0}

def get_input_streamlit(label, var_name, key, default=None, unit="", desc="", disabled=False):
    try:
        st.caption(f"{desc} | 单位: {unit}")
        input_type = st.radio(
            f"{label} 输入类型",
            ["单一值", "多个值 (逗号分隔)", "范围 (带步长)"],
            key=f"{key}_type",
            horizontal=True,
            disabled=disabled
        )
        
        default_val = str(default) if default is not None else ""
        
        if input_type == "单一值":
            val = st.text_input(label, default_val, key=f"{key}_single", disabled=disabled)
            if val.strip() == "":
                return symbols(var_name)
            try:
                return [float(val)]
            except ValueError:
                st.error("请输入有效的数字")
                return None
        elif input_type == "多个值 (逗号分隔)":
            val = st.text_input(
                label, 
                default_val, 
                key=f"{key}_multi", 
                disabled=disabled
            )
            if val.strip() == "":
                return symbols(var_name)
            try:
                values = [float(x.strip()) for x in val.split(',') if x.strip()]
                if not values:
                    st.error("请至少输入一个值")
                    return None
                return values
            except ValueError:
                st.error("请输入有效的逗号分隔数字")
                return None
        else:
            st.caption("范围示例: 开始=1.0, 结束=5.0, 步长=1.0")
            col1, col2, col3 = st.columns(3)
            with col1:
                start = st.text_input(
                    f"{label} 起始值", 
                    default_val, 
                    key=f"{key}_start", 
                    disabled=disabled
                )
            with col2:
                end = st.text_input(
                    f"{label} 结束值", 
                    "", 
                    key=f"{key}_end", 
                    disabled=disabled
                )
            with col3:
                step = st.text_input(
                    f"{label} 步长", 
                    "0.5", 
                    key=f"{key}_step", 
                    disabled=disabled
                )
                
            if start.strip() == "" or end.strip() == "":
                return symbols(var_name)
                
            try:
                start = float(start)
                end = float(end)
                step = float(step) if step else 0.5
                
                if step <= 0:
                    step = 0.5
                    st.warning("步长必须为正数, 自动设置为 0.5")
                if start > end:
                    start, end = end, start
                    st.warning("起始值大于结束值, 自动交换")
                if (end - start) < step:
                    st.warning("步长大于范围差值, 将只返回起始值")
                    return [start]
                    
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
    except Exception as e:
        st.error(f"输入处理错误: {str(e)}")
        return None

def solve_numerically(eqs, sym_vars, initial_guess):
    try:
        var_list = list(sym_vars.values())
        
        def residuals(x):
            substitutions = {var_list[i]: x[i] for i in range(len(x))}
            residuals = []
            for eq in eqs:
                substituted = eq.subs(substitutions)
                if substituted == True:
                    residuals.append(0.0)
                elif substituted == False:
                    residuals.append(1e10)
                else:
                    try:
                        simplified = simplify(substituted)
                        residuals.append(float(abs(simplified.evalf())))
                    except:
                        residuals.append(1e10)
            return residuals
        
        n_vars = len(initial_guess)
        lower_bounds = [0.1] * n_vars
        upper_bounds = [30.0] * n_vars
        
        for i, var in enumerate(initial_guess.keys()):
            var_str = str(var)
            if var_str.startswith(('rh0', 'rh')):
                lower_bounds[i] = 0.1
                upper_bounds[i] = 20.0
            elif var_str.startswith(('D', 'C0', 'u', 'w')):
                lower_bounds[i] = 0.1
                upper_bounds[i] = 30.0
            elif var_str.startswith(('P', 'E')):
                lower_bounds[i] = 0.01
                upper_bounds[i] = 5000.0
            elif var_str.startswith('gamma'):
                lower_bounds[i] = 0.1
                upper_bounds[i] = 5.0
            elif var_str.startswith('T'):
                lower_bounds[i] = 100.0
                upper_bounds[i] = 1e5
        
        result = least_squares(
            residuals,
            list(initial_guess.values()),
            bounds=(lower_bounds, upper_bounds),
            ftol=1e-8,
            max_nfev=5000
        )
        
        if result.success:
            return {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
        return None
    except Exception as e:
        st.warning(f"数值求解错误: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def generate_shock_plots(df, C0, S, material_name, material_type):
    try:
        if len(df) > 1000:
            df = df.sample(1000)
            
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        method_colors = {
            'iml': 'red',
            'ssp': 'blue',
            'calculated': 'green',
            'manual_input': 'purple',
            'bulk_import': 'orange'
        }
        default_color = 'gray'
        
        fig.suptitle(f'Material: {material_name} - Shock Wave Relationships', fontsize=16)
        
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
        
        u_p_range = np.linspace(0, df['Up'].max()*1.1, 100) if not df.empty else np.linspace(0, 10, 100)
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
                label=f'{method}' if method == methods[0] else "",
                color=color, alpha=0.7
            )
        
        rho0 = df['rho0'].mean() if not df.empty else 8.96
        P_range = rho0 * U_s_fit * u_p_range
        
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
        
        V_V0_range = 1 - u_p_range / U_s_fit
        
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
        
        rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)
        
        axs[1, 1].plot(P_range, rho_range, 'r-', label='Theoretical Curve')
        axs[1, 1].set_xlabel('Pressure P (GPa)')
        axs[1, 1].set_ylabel('Density ρ (g/cm³)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"绘图错误: {str(e)}")
        return None

def save_plot_to_bytes(fig):
    try:
        if fig is None:
            return None
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"保存图像失败: {str(e)}")
        return None

def display_material_plots(df, material_name, material_type):
    try:
        if not df.empty:
            with st.expander(f"查看 {material_type} 材料 {material_name} 冲击波图像", expanded=True):
                C0, S = fit_hugoniot(df)
                fig = generate_shock_plots(df, C0, S, material_name, material_type)
                if fig:
                    st.pyplot(fig)
                    buf = save_plot_to_bytes(fig)
                    
                    material_type_en = {
                        "飞片": "flyer",
                        "基板": "substrate",
                        "样品": "sample"
                    }.get(material_type, material_type.lower())
                    
                    st.download_button(
                        label=f"下载 {material_type} 材料 {material_name} 图像",
                        data=buf,
                        file_name=f"{material_type_en}_{material_name}_shock_relations.png",
                        mime="image/png"
                    )
        else:
            st.info(f"没有可用数据生成 {material_type} 材料 {material_name} 图像")
    except Exception as e:
        st.error(f"显示材料图像错误: {str(e)}")

@st.cache_data(ttl=3600)
def plot_results_streamlit(results, calculate_temp=True):
    try:
        if not results:
            return None
            
        if len(results) > 1000:
            results = results[:1000]
            
        subplot_count = 4 if calculate_temp else 3
        fig = plt.figure(figsize=(18, 9) if calculate_temp else (18, 7))
        
        pf_values = [r.get('Pf', 0) for r in results]
        uf_values = [r.get('uf', 0) for r in results]
        df_values = [r.get('Df', 0) for r in results]
        rhf_values = [r.get('rhf', 0) for r in results]
        
        # 1. 压力-粒子速度图
        ax1 = fig.add_subplot(221 if calculate_temp else 221)
        ax1.errorbar(uf_values, pf_values, 
                     yerr=[r.get('Pf_err', 0.1) for r in results],
                     xerr=[r.get('uf_err', 0.05) for r in results],
                     fmt='bo', ecolor='r', capsize=5, label='Flyer Data')
        ax1.set_xlabel('Particle Velocity Up (km/s)')
        ax1.set_ylabel('Shock Pressure P (GPa)')
        ax1.set_title('Pressure-Particle Velocity Relationship')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 温度-压力图
        ax2 = None
        if calculate_temp:
            tf_values = [r.get('Tf', 0) for r in results]
            
            ax2 = fig.add_subplot(222)
            ax2.scatter(pf_values, tf_values, c='orange', label='Flyer Temperature')
            ax2.set_xlabel('Shock Pressure P (GPa)')
            ax2.set_ylabel('Shock Temperature T (K)')
            ax2.set_title('Temperature-Pressure Relationship')
            ax2.legend()
            ax2.grid(True)
        
        # 3. 冲击波速度-粒子速度图
        ax3 = fig.add_subplot(223 if calculate_temp else 222)
        ax3.scatter(uf_values, df_values, c='blue', label='Flyer')
        ax3.set_xlabel('Particle Velocity Up (km/s)')
        ax3.set_ylabel('Shock Wave Velocity Us (km/s)')
        ax3.set_title('Shock Velocity-Particle Velocity Relationship')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 密度-压力图
        ax4 = fig.add_subplot(224 if calculate_temp else 223)
        ax4.scatter(pf_values, rhf_values, c='green', label='Flyer')
        ax4.set_xlabel('Shock Pressure P (GPa)')
        ax4.set_ylabel('Compressed Density (g/cm³)')
        ax4.set_title('Density-Pressure Relationship')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"结果绘图错误: {str(e)}")
        return None

def home_page():
    try:
        st.session_state.previous_page = "home"
        st.title("冲击波参数计算与分析系统")
        st.info("""
        系统核心模型说明:
        1. 基于Rankine-Hugoniot守恒方程组
        2. 假设条件：平面冲击波、稳态传播、忽略初始压力
        3. 单位体系：密度(g/cm³)、速度(km/s)、压力(GPa)
        """)
        
        if st.button("查看数据库"):
            st.session_state.page = "view_database"
            st.rerun()
        
        st.write("选择操作模式:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("使用数据库数据"):
                st.session_state.page = "database_mode"
                st.rerun()
        with col2:
            if st.button("手动输入参数"):
                st.session_state.page = "manual_mode"
                st.rerun()
    except Exception as e:
        st.error(f"首页错误: {str(e)}")

def database_mode_page():
    try:
        st.session_state.previous_page = "database_mode"
        st.title("数据库模式")
        st.write("从数据库加载材料数据，基于Hugoniot关系拟合参数并求解")
        
        calculate_temp = st.checkbox("进行温度相关计算", value=True)
        
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
        
        flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
        base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
        sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
        
        with st.spinner(f"正在拟合飞片材料 {flyer_material} 数据..."):
            flyer_fit = fit_material_data(flyer_df, flyer_material, "飞片")
        
        with st.spinner(f"正在拟合基板材料 {base_material} 数据..."):
            base_fit = fit_material_data(base_df, base_material, "基板")
        
        with st.spinner(f"正在拟合样品材料 {sample_material} 数据..."):
            sample_fit = fit_material_data(sample_df, sample_material, "样品")
        
        st.subheader("冲击波参数分析（Hugoniot关系）")
        st.caption("基于线性Hugoniot关系 Us = C0 + S·Up")
        
        display_material_plots(flyer_df, flyer_material, "飞片")
        display_material_plots(base_df, base_material, "基板")
        display_material_plots(sample_df, sample_material, "样品")
        
        default_params = {"f": flyer_fit, "b": base_fit, "s": sample_fit}
        variables = {
            "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
            "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
            "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
        }
        
        input_params = {}
        sym_vars = {}
        
        st.info("飞片冲击关系: 飞片速度 w 与粒子速度 uf 的关系为 w = Df + uf")
        
        Cv_values = {}
        if calculate_temp:
            st.subheader("比热容设置（用于温度计算）")
            col1, col2, col3 = st.columns(3)
            with col1:
                Cv_values['f'] = st.number_input(f"飞片比热容 Cv (J/(kg·K)) ({flyer_material})", 
                                                value=385.0, min_value=1.0)
            with col2:
                Cv_values['b'] = st.number_input(f"基板比热容 Cv (J/(kg·K)) ({base_material})", 
                                                value=385.0, min_value=1.0)
            with col3:
                Cv_values['s'] = st.number_input(f"样品比热容 Cv (J/(kg·K)) ({sample_material})", 
                                                value=385.0, min_value=1.0)
        
        with st.expander(f"{flyer_material} 飞片参数", expanded=True):
            cols = st.columns(3)
            var_descs = {
                "rh0f": "初始密度（必须输入）",
                "rhf": "压缩密度",
                "Df": "冲击波速度 (对应Us)",
                "C0f": "体声速 (Hugoniot拟合)",
                "Sf": "Hugoniot参数S (无量纲)",
                "E0f": "初始内能密度",
                "Ef": "压缩后内能密度",
                "uf": "粒子速度 (对应Up)",
                "w": "飞片初始冲击速度",
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
                if var.startswith('T') and not calculate_temp:
                    continue
                    
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
                        if not flyer_df.empty and 'gamma' in flyer_df.columns:
                            gamma_vals = flyer_df['gamma'].dropna()
                            if len(gamma_vals) > 0:
                                default_val = gamma_vals.mean()
                            else:
                                default_val = 2.0
                        else:
                            default_val = 2.0
                    if var == "rh0f" and default_val is None:
                        default_val = 8.96
                    
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
        
        with st.expander(f"{base_material} 基板参数", expanded=True):
            cols = st.columns(3)
            for i, var in enumerate(variables["b"]):
                if var.startswith('T') and not calculate_temp:
                    continue
                    
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
                        if not base_df.empty and 'gamma' in base_df.columns:
                            gamma_vals = base_df['gamma'].dropna()
                            if len(gamma_vals) > 0:
                                default_val = gamma_vals.mean()
                            else:
                                default_val = 2.0
                        else:
                            default_val = 2.0
                    
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
                        desc="基板初始密度（必须输入）" if var == "rh0b" else
                             "基板压缩密度" if var == "rhb" else
                             "基板冲击波速度" if var == "Db" else
                             "基板体声速" if var == "C0b" else
                             "基板Hugoniot参数" if var == "Sb" else
                             "基板初始内能密度" if var == "E0b" else
                             "基板压缩后内能密度" if var == "Eb" else
                             "基板粒子速度" if var == "ub" else
                             "基板冲击压力" if var == "Pb" else
                             "基板格吕奈森系数" if var == "gammab" else
                             "基板冲击温度"
                    )
                    input_params[var] = val
                    sym_vars[var] = symbols(var)
        
        with st.expander(f"{sample_material} 样品参数", expanded=True):
            cols = st.columns(3)
            for i, var in enumerate(variables["s"]):
                if var.startswith('T') and not calculate_temp:
                    continue
                    
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
                        if not sample_df.empty and 'gamma' in sample_df.columns:
                            gamma_vals = sample_df['gamma'].dropna()
                            if len(gamma_vals) > 0:
                                default_val = gamma_vals.mean()
                            else:
                                default_val = 2.0
                        else:
                            default_val = 2.0
                    
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
                        desc="样品初始密度（必须输入）" if var == "rh0s" else
                             "样品压缩密度" if var == "rhs" else
                             "样品冲击波速度" if var == "Ds" else
                             "样品体声速" if var == "C0s" else
                             "样品Hugoniot参数S" if var == "Ss" else
                             "样品初始内能密度" if var == "E0s" else
                             "样品压缩后内能密度" if var == "Es" else
                             "样品粒子速度" if var == "us" else
                             "样品冲击压力" if var == "Ps" else
                             "样品格吕奈森系数" if var == "gammas" else
                             "样品冲击温度"
                    )
                    input_params[var] = val
                    sym_vars[var] = symbols(var)
        
        col_save, col_other = st.columns([1, 3])
        with col_save:
            if st.button("保存当前参数到数据库"):
                count = save_input_parameters(input_params, sample_material, "database_mode_input")
                if count > 0:
                    st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
        
        range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
        total_combinations = 1
        for v in range_params.values():
            total_combinations *= len(v)
        
        max_combinations = st.slider(
            "最大参数组合数", 
            min_value=10, 
            max_value=1000, 
            value=min(100, total_combinations)
        )
        
        if st.button("开始求解"):
            valid = True
            for var in ['rh0f', 'rh0b', 'rh0s']:
                if isinstance(input_params.get(var), symbols):
                    valid = False
                    st.error(f"{var}（初始密度）为必填参数，请输入值")
            
            for var, val in input_params.items():
                if val is None:
                    valid = False
                    st.error(f"{var} 输入无效，请检查")
            
            if not valid:
                return
                
            combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
            combinations = list(combinations)
            
            if len(combinations) > max_combinations:
                st.warning(f"参数组合过多 ({len(combinations)}), 已截断至 {max_combinations} 组")
                combinations = combinations[:max_combinations]
            
            results = []
            progress_bar = st.progress(0)
            total = len(combinations)
            count = 0
            
            for combo in combinations:
                count += 1
                if count % 10 == 0 or count == total:
                    progress_bar.progress(count / total)
                    
                current_subs = {sym_vars[k]: v for k, v in combo}
                
                eqs = [
                    Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                    Eq(sym_vars['w'] - (sym_vars['Df'] + sym_vars['uf']), 0),
                    Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                    Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf'], 0),
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                    Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                    Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                    Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*sym_vars['uf'], 0),
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                    Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                    Eq(sym_vars['uf'] - sym_vars['ub'], 0)
                ]
                
                if calculate_temp:
                    eqs.append(Eq(sym_vars['Tf'] - 300 - (sym_vars['Ef'] - sym_vars['E0f'])*1e6 / 
                                 (Cv_values['f'] * (1 + sym_vars['gammaf']/2)), 0))
                    eqs.append(Eq(sym_vars['Tb'] - 300 - (sym_vars['Eb'] - sym_vars['E0b'])*1e6 / 
                                 (Cv_values['b'] * (1 + sym_vars['gammab']/2)), 0))
                
                try:
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
                        Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
                    ]
                    
                    if calculate_temp:
                        eqs += [
                            Eq(sym_vars['Tb'] - sym_vars['Ts'], 0),
                            Eq(sym_vars['gammab'] - sym_vars['gammas'], 0)
                        ]
                else:
                    eqs += [
                        Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                        Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                        Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                        Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                        Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                        Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                        Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),
                        Eq(sym_vars['ub'] - sym_vars['us'], 0)
                    ]
                    
                    if calculate_temp:
                        eqs.append(Eq(sym_vars['Ts'] - 300 - (sym_vars['Es'] - sym_vars['E0s'])*1e6 / 
                                     (Cv_values['s'] * (1 + sym_vars['gammas']/2)), 0))
                
                substituted_eqs = [eq.subs(current_subs) for eq in eqs]
                remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
                
                if not remaining_vars:
                    continue
                    
                try:
                    initial_guess = {}
                    known_params = {}
                    for k, v in current_subs.items():
                        try:
                            known_params[str(k)] = float(v)
                        except:
                            pass
                    
                    for var in remaining_vars:
                        var_str = str(var)
                        if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                            initial_guess[var] = known_params['Df'] + known_params['uf']
                        elif var_str == 'Df' and 'w' in known_params and 'uf' in known_params:
                            initial_guess[var] = known_params['w'] - known_params['uf']
                        elif var_str == 'uf' and 'w' in known_params and 'Df' in known_params:
                            initial_guess[var] = known_params['w'] - known_params['Df']
                        elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                            initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf']
                        elif var_str.startswith(('rh0', 'rh')):
                            initial_guess[var] = known_params.get('rh0f', 8.0)
                        elif var_str.startswith(('D', 'C0', 'u')):
                            if 'w' in known_params:
                                initial_guess[var] = known_params['w'] / 2
                            else:
                                initial_guess[var] = 5.0
                        elif var_str == 'w':
                            initial_guess[var] = 10.0
                        elif var_str.startswith('P'):
                            if 'rh0f' in known_params and 'w' in known_params:
                                initial_guess[var] = known_params['rh0f'] * (known_params['w']/2) * (known_params['w']/2)
                            else:
                                initial_guess[var] = 100.0
                        elif var_str.startswith('gamma'):
                            initial_guess[var] = 2.0
                        elif var_str.startswith('T'):
                            initial_guess[var] = 3000.0
                        else:
                            initial_guess[var] = 1.0
                    
                    solution = solve_numerically(substituted_eqs, {v:v for v in remaining_vars}, initial_guess)
                    
                    if solution:
                        record = solution.copy()
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
                    st.warning(f"求解错误: {str(e)}")
            
            if results:
                st.success(f"求解完成，找到 {len(results)} 个解")
                
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
                fig = plot_results_streamlit(results, calculate_temp)
                if fig:
                    st.pyplot(fig)
                    buf2 = BytesIO()
                    fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                    buf2.seek(0)
                    st.download_button(
                        label="下载图表",
                        data=buf2,
                        file_name="analysis_with_temp.png" if calculate_temp else "analysis_results.png",
                        mime="image/png"
                    )
                
                if st.button("保存结果到数据库"):
                    count = save_results_to_db(results, sample_material)
                    if count > 0:
                        st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
        
        if st.button("返回首页"):
            st.session_state.page = "home"
            st.rerun()
    except Exception as e:
        st.error(f"数据库模式错误: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def manual_mode_page():
    try:
        st.session_state.previous_page = "manual_mode"
        st.title("手动输入模式")
        st.write("通过手动输入参数进行求解，适用于没有数据库数据的场景")
        
        calculate_temp = st.checkbox("进行温度相关计算", value=True)
        
        if st.button("查看数据库"):
            st.session_state.page = "view_database"
            st.rerun()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            flyer_material = st.text_input("飞片材料名称", value="Copper")
        with col2:
            base_material = st.text_input("基板材料名称", value="Aluminum")
        with col3:
            sample_material = st.text_input("样品材料名称", value="Copper")
        
        st.info("飞片冲击关系: 飞片速度 w 与粒子速度 uf 的关系为 w = Df + uf")
        
        Cv_values = {}
        if calculate_temp:
            st.subheader("比热容设置（用于温度计算）")
            col1, col2, col3 = st.columns(3)
            with col1:
                Cv_values['f'] = st.number_input(f"飞片比热容 Cv (J/(kg·K)) ({flyer_material})", 
                                                value=385.0, min_value=1.0)
            with col2:
                Cv_values['b'] = st.number_input(f"基板比热容 Cv (J/(kg·K)) ({base_material})", 
                                                value=385.0, min_value=1.0)
            with col3:
                Cv_values['s'] = st.number_input(f"样品比热容 Cv (J/(kg·K)) ({sample_material})", 
                                                value=385.0, min_value=1.0)
        
        exp_method = st.text_input("实验方法/数据来源", value="manual_input")
        
        variables = {
            "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
            "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
            "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
        }
        
        input_params = {}
        sym_vars = {}
        
        with st.expander(f"{flyer_material} 飞片参数", expanded=True):
            cols = st.columns(3)
            for i, var in enumerate(variables["f"]):
                if var.startswith('T') and not calculate_temp:
                    continue
                    
                with cols[i % 3]:
                    default_val = 2.0 if var == "gammaf" else None
                    if var == "rh0f":
                        default_val = 8.96
                    
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
                        desc="飞片初始密度（必须输入）" if var == "rh0f" else
                             "飞片压缩密度" if var == "rhf" else
                             "飞片冲击波速度" if var == "Df" else
                             "飞片体声速" if var == "C0f" else
                             "飞片Hugoniot参数S" if var == "Sf" else
                             "飞片初始内能密度" if var == "E0f" else
                             "飞片压缩后内能密度" if var == "Ef" else
                             "飞片粒子速度" if var == "uf" else
                             "飞片初始冲击速度" if var == "w" else
                             "飞片冲击压力" if var == "Pf" else
                             "飞片格吕奈森系数" if var == "gammaf" else
                             "飞片冲击温度"
                    )
                    input_params[var] = val
                    sym_vars[var] = symbols(var)
        
        with st.expander(f"{base_material} 基板参数", expanded=True):
            cols = st.columns(3)
            for i, var in enumerate(variables["b"]):
                if var.startswith('T') and not calculate_temp:
                    continue
                    
                with cols[i % 3]:
                    default_val = 2.0 if var == "gammab" else None
                    if var == "rh0b":
                        default_val = 2.7
                    
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
                        desc="基板初始密度（必须输入）" if var == "rh0b" else
                             "基板压缩密度" if var == "rhb" else
                             "基板冲击波速度" if var == "Db" else
                             "基板体声速" if var == "C0b" else
                             "基板Hugoniot参数" if var == "Sb" else
                             "基板初始内能密度" if var == "E0b" else
                             "基板压缩后内能密度" if var == "Eb" else
                             "基板粒子速度" if var == "ub" else
                             "基板冲击压力" if var == "Pb" else
                             "基板格吕奈森系数" if var == "gammab" else
                             "基板冲击温度"
                    )
                    input_params[var] = val
                    sym_vars[var] = symbols(var)
        
        with st.expander(f"{sample_material} 样品参数", expanded=True):
            cols = st.columns(3)
            for i, var in enumerate(variables["s"]):
                if var.startswith('T') and not calculate_temp:
                    continue
                    
                with cols[i % 3]:
                    default_val = 2.0 if var == "gammas" else None
                    if var == "rh0s":
                        default_val = 8.96
                    
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
                        desc="样品初始密度（必须输入）" if var == "rh0s" else
                             "样品压缩密度" if var == "rhs" else
                             "样品冲击波速度" if var == "Ds" else
                             "样品体声速" if var == "C0s" else
                             "样品Hugoniot参数S" if var == "Ss" else
                             "样品初始内能密度" if var == "E0s" else
                             "样品压缩后内能密度" if var == "Es" else
                             "样品粒子速度" if var == "us" else
                             "样品冲击压力" if var == "Ps" else
                             "样品格吕奈森系数" if var == "gammas" else
                             "样品冲击温度"
                    )
                    input_params[var] = val
                    sym_vars[var] = symbols(var)
        
        col_save, col_other = st.columns([1, 3])
        with col_save:
            if st.button("保存当前参数到数据库"):
                count = save_input_parameters(input_params, sample_material, exp_method)
                if count > 0:
                    st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
        
        range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
        total_combinations = 1
        for v in range_params.values():
            total_combinations *= len(v)
        
        max_combinations = st.slider(
            "最大参数组合数", 
            min_value=10, 
            max_value=1000, 
            value=min(100, total_combinations)
        )
        
        if st.button("开始求解方程组"):
            valid = True
            for var in ['rh0f', 'rh0b', 'rh0s']:
                if isinstance(input_params.get(var), symbols):
                    valid = False
                    st.error(f"{var}（初始密度）为必填参数，请输入值")
            
            for var, val in input_params.items():
                if val is None:
                    valid = False
                    st.error(f"{var} 输入无效，请检查")
            
            if not valid:
                return
                
            combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
            combinations = list(combinations)
            
            if len(combinations) > max_combinations:
                st.warning(f"参数组合过多 ({len(combinations)}), 已截断至 {max_combinations} 组")
                combinations = combinations[:max_combinations]
            
            results = []
            progress_bar = st.progress(0)
            total = len(combinations)
            count = 0
            
            for combo in combinations:
                count += 1
                if count % 10 == 0 or count == total:
                    progress_bar.progress(count / total)
                    
                current_subs = {sym_vars[k]: v for k, v in combo}
                
                try:
                    if current_subs.get(sym_vars['Df'], 0) <= current_subs.get(sym_vars['uf'], 0):
                        st.warning("飞片参数错误: Df 必须大于 uf")
                        continue
                    if current_subs.get(sym_vars['Db'], 0) <= current_subs.get(sym_vars['ub'], 0):
                        st.warning("基板参数错误: Db 必须大于 ub")
                        continue
                except:
                    pass
                
                eqs = [
                    Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                    Eq(sym_vars['w'] - (sym_vars['Df'] + sym_vars['uf']), 0),
                    Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                    Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf'], 0),
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                    Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                    Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                    Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*sym_vars['uf'], 0),
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                    Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                    Eq(sym_vars['uf'] - sym_vars['ub'], 0)
                ]
                
                if calculate_temp:
                    eqs.append(Eq(sym_vars['Tf'] - 300 - (sym_vars['Ef'] - sym_vars['E0f'])*1e6 / 
                                 (Cv_values['f'] * (1 + sym_vars['gammaf']/2)), 0))
                    eqs.append(Eq(sym_vars['Tb'] - 300 - (sym_vars['Eb'] - sym_vars['E0b'])*1e6 / 
                                 (Cv_values['b'] * (1 + sym_vars['gammab']/2)), 0))
                
                try:
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
                        Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
                    ]
                    
                    if calculate_temp:
                        eqs += [
                            Eq(sym_vars['Tb'] - sym_vars['Ts'], 0),
                            Eq(sym_vars['gammab'] - sym_vars['gammas'], 0)
                        ]
                else:
                    eqs += [
                        Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                        Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                        Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                        Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                        Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                        Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                        Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),
                        Eq(sym_vars['ub'] - sym_vars['us'], 0)
                    ]
                    
                    if calculate_temp:
                        eqs.append(Eq(sym_vars['Ts'] - 300 - (sym_vars['Es'] - sym_vars['E0s'])*1e6 / 
                                     (Cv_values['s'] * (1 + sym_vars['gammas']/2)), 0))
                
                substituted_eqs = [eq.subs(current_subs) for eq in eqs]
                remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
                
                if not remaining_vars:
                    continue
                    
                try:
                    initial_guess = {}
                    known_params = {}
                    for k, v in current_subs.items():
                        try:
                            known_params[str(k)] = float(v)
                        except:
                            pass
                    
                    for var in remaining_vars:
                        var_str = str(var)
                        if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                            initial_guess[var] = known_params['Df'] + known_params['uf']
                        elif var_str == 'Df' and 'w' in known_params and 'uf' in known_params:
                            initial_guess[var] = known_params['w'] - known_params['uf']
                        elif var_str == 'uf' and 'w' in known_params and 'Df' in known_params:
                            initial_guess[var] = known_params['w'] - known_params['Df']
                        elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                            initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf']
                        elif var_str.startswith(('rh0', 'rh')):
                            initial_guess[var] = known_params.get('rh0f', 8.0)
                        elif var_str.startswith(('D', 'C0', 'u')):
                            if 'w' in known_params:
                                initial_guess[var] = known_params['w'] / 2
                            else:
                                initial_guess[var] = 5.0
                        elif var_str == 'w':
                            initial_guess[var] = 10.0
                        elif var_str.startswith('P'):
                            if 'rh0f' in known_params and 'w' in known_params:
                                initial_guess[var] = known_params['rh0f'] * (known_params['w']/2) * (known_params['w']/2)
                            else:
                                initial_guess[var] = 100.0
                        elif var_str.startswith('gamma'):
                            initial_guess[var] = 2.0
                        elif var_str.startswith('T'):
                            initial_guess[var] = 3000.0
                        else:
                            initial_guess[var] = 1.0
                    
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
                    st.warning(f"求解错误: {str(e)}")
            
            if results:
                st.success(f"求解完成，找到 {len(results)} 个解")
                
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
                fig = plot_results_streamlit(results, calculate_temp)
                if fig:
                    st.pyplot(fig)
                    buf2 = BytesIO()
                    fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                    buf2.seek(0)
                    st.download_button(
                        label="下载图表",
                        data=buf2,
                        file_name="analysis_with_temp.png" if calculate_temp else "analysis_results.png",
                        mime="image/png"
                    )
                
                if st.button("保存计算结果到数据库"):
                    count = save_results_to_db(results, sample_material)
                    if count > 0:
                        st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
        
        if st.button("返回首页"):
            st.session_state.page = "home"
            st.rerun()
    except Exception as e:
        st.error(f"手动模式错误: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def main():
    try:
        # 确保所有session_state变量都已初始化
        required_states = {
            'page': "home",
            'previous_page': "home",
            'confirm_delete': False,
            'confirm_clear': False
        }
        
        for key, default_value in required_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

        # 页面路由
        if st.session_state.page == "home":
            home_page()
        elif st.session_state.page == "database_mode":
            database_mode_page()
        elif st.session_state.page == "manual_mode":
            manual_mode_page()
        elif st.session_state.page == "view_database":
            view_database()
            # 返回按钮
            col_back, _ = st.columns([1, 5])
            with col_back:
                if st.button("返回"):
                    if 'previous_page' in st.session_state:
                        st.session_state.page = st.session_state.previous_page
                    else:
                        st.session_state.page = "home"
                    st.rerun()
        else:
            # 处理未知页面
            st.session_state.page = "home"
            st.rerun()
    except Exception as e:
        st.error(f"应用错误: {str(e)}")
        # 显示详细错误信息以便调试
        import traceback
        st.text("错误详情:")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
