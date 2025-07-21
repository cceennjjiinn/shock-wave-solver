import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from sympy import symbols, Eq, solve
from io import BytesIO
from PIL import Image
import itertools

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
    
    conn.commit()

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

def fit_material_data(df, material_name):
    """对数据库中的材料数据进行拟合，返回最优参数"""
    if df is None or df.empty:
        st.warning(f"材料 '{material_name}' 没有数据")
        return None
    
    # 对D-u关系进行线性拟合 (D = C0 + λ*u)
    X = df['Up'].values.reshape(-1, 1)
    y = df['Us'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    C0 = model.intercept_
    lambda_val = model.coef_[0]
    r2 = r2_score(y, model.predict(X))
    
    # 计算平均密度、压力等参数
    avg_rho0 = df['rho0'].mean()
    avg_P = df['P'].mean()
    
    st.info(f"{material_name} 拟合结果: D = {C0:.4f} + {lambda_val:.4f}*u (R² = {r2:.4f})")
    st.info(f"{material_name} 平均参数: ρ₀ = {avg_rho0:.4f} g/cm³, P = {avg_P:.4f} GPa")
    
    return {
        "C0": C0,
        "lambda": lambda_val,
        "rho0": avg_rho0,
        "r2": r2
    }

def get_input_streamlit(label, var_name, key, default=None):
    """增强输入函数，支持默认值"""
    input_type = st.radio(
        f"{label} 输入类型",
        ["单个值", "多个值(逗号分隔)", "范围(带可选步长)"],
        key=f"{key}_type"
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
    else:  # 范围
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

def plot_results_streamlit(results):
    """绘制结果图表"""
    if not results:
        st.warning("没有数据可绘制")
        return
        
    fig = plt.figure(figsize=(18, 6))
    
    # 准备数据
    pf_values = [r.get('Pf', 0) for r in results]
    pb_values = [r.get('Pb', 0) for r in results]
    ps_values = [r.get('Ps', 0) for r in results]
    uf_values = [r.get('uf', 0) for r in results]
    ub_values = [r.get('ub', 0) for r in results]
    us_values = [r.get('us', 0) for r in results]
    df_values = [r.get('Df', 0) for r in results]
    db_values = [r.get('Db', 0) for r in results]
    ds_values = [r.get('Ds', 0) for r in results]
    rhf_values = [r.get('rhf', 0) for r in results]
    rhb_values = [r.get('rhb', 0) for r in results]
    rhs_values = [r.get('rhs', 0) for r in results]
    
    # 通用拟合函数
    def add_fit_curve(ax, x_data, y_data, color, label, deg=2):
        try:
            clean_x = [xi for xi, yi in zip(x_data, y_data) if isinstance(xi, (int, float)) and isinstance(yi, (int, float))]
            clean_y = [yi for xi, yi in zip(x_data, y_data) if isinstance(xi, (int, float)) and isinstance(yi, (int, float))]
            
            if len(clean_x) < deg + 1:
                return
                
            coeffs = np.polyfit(clean_x, clean_y, deg)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(clean_x), max(clean_x), 100)
            y_fit = poly(x_fit)
            
            ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'{label} 拟合')
            
        except Exception as e:
            st.warning(f"拟合失败 ({label}): {str(e)}")

    # 1. P-V 图
    ax1 = fig.add_subplot(131)
    ax1.scatter(rhf_values, pf_values, c='blue', label='飞片', alpha=0.6)
    ax1.scatter(rhb_values, pb_values, c='red', label='基板', alpha=0.6)
    ax1.scatter(rhs_values, ps_values, c='green', label='样品', alpha=0.6)
    add_fit_curve(ax1, rhf_values, pf_values, 'blue', '飞片')
    add_fit_curve(ax1, rhb_values, pb_values, 'red', '基板')
    add_fit_curve(ax1, rhs_values, ps_values, 'green', '样品')
    ax1.set(xlabel='密度 (rh)', ylabel='压力 (P)', title='P-rh 图')
    ax1.legend()

    # 2. P-u 图
    ax2 = fig.add_subplot(132)
    ax2.scatter(uf_values, pf_values, c='blue', label='飞片', alpha=0.6)
    ax2.scatter(ub_values, pb_values, c='red', label='基板', alpha=0.6)
    ax2.scatter(us_values, ps_values, c='green', label='样品', alpha=0.6)
    add_fit_curve(ax2, uf_values, pf_values, 'blue', '飞片')
    add_fit_curve(ax2, ub_values, pb_values, 'red', '基板')
    add_fit_curve(ax2, us_values, ps_values, 'green', '样品')
    ax2.set(xlabel='速度 (u)', ylabel='压力 (P)', title='P-u 图')
    ax2.legend()

    # 3. D-u 图
    ax3 = fig.add_subplot(133)
    ax3.scatter(uf_values, df_values, c='blue', label='飞片', alpha=0.6)
    ax3.scatter(ub_values, db_values, c='red', label='基板', alpha=0.6)
    ax3.scatter(us_values, ds_values, c='green', label='样品', alpha=0.6)
    add_fit_curve(ax3, uf_values, df_values, 'blue', '飞片')
    add_fit_curve(ax3, ub_values, db_values, 'red', '基板')
    add_fit_curve(ax3, us_values, ds_values, 'green', '样品')
    ax3.set(xlabel='速度 (u)', ylabel='位移 (D)', title='D-u 图')
    ax3.legend()

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    
    st.image(img, caption="分析结果图表")
    
    # 提供下载
    buf2 = BytesIO()
    plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
    buf2.seek(0)
    st.download_button(
        label="下载图表",
        data=buf2,
        file_name="analysis_with_fit.png",
        mime="image/png"
    )
    
    return fig

def home_page():
    """主页 - 选择模式"""
    st.title("多物理场求解器")
    st.write("选择你要使用的模式：")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("使用数据库数据"):
            st.session_state.page = "database_mode"
    
    with col2:
        if st.button("手动输入参数"):
            st.session_state.page = "manual_mode"

def database_mode_page():
    """数据库模式页面"""
    st.title("数据库模式")
    st.write("从数据库加载材料的冲击波数据，拟合后选择最优参数")
    
    # 获取所有可用材料
    materials = get_all_materials()
    if not materials:
        st.error("数据库中没有可用的材料数据")
        return
    
    # 为飞片、基板和样品选择材料
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.selectbox("选择飞片材料", materials, key="flyer_material")
    with col2:
        base_material = st.selectbox("选择基板材料", materials, key="base_material")
    with col3:
        sample_material = st.selectbox("选择样品材料", materials, key="sample_material")
    
    # 加载所选材料的数据
    flyer_df = get_material_data(flyer_material)
    base_df = get_material_data(base_material)
    sample_df = get_material_data(sample_material)
    
    # 拟合材料数据
    with st.spinner(f"正在拟合 {flyer_material} 数据..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material)
    
    with st.spinner(f"正在拟合 {base_material} 数据..."):
        base_fit = fit_material_data(base_df, base_material)
    
    with st.spinner(f"正在拟合 {sample_material} 数据..."):
        sample_fit = fit_material_data(sample_df, sample_material)
    
    # 为输入字段提供拟合的参数作为默认值
    default_params = {
        "f": flyer_fit,
        "b": base_fit,
        "s": sample_fit
    }
    
    # 参数输入
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "nubdaf", "E0f", "Ef", "uf", "w", "Pf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "nubdab", "E0b", "Eb", "ub", "Pb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "nubdas", "E0s", "Es", "us", "Ps"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片参数
    with st.expander(f"{flyer_material} 飞片相关参数", expanded=True):
        cols = st.columns(3)
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
                val = get_input_streamlit(f"{var}", var, f"f_{var}", default=default_val)
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数
    with st.expander(f"{base_material} 基板相关参数", expanded=True):
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
                val = get_input_streamlit(f"{var}", var, f"b_{var}", default=default_val)
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数
    with st.expander(f"{sample_material} 样品相关参数", expanded=True):
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
                val = get_input_streamlit(f"{var}", var, f"s_{var}", default=default_val)
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 求解按钮
    if st.button("开始求解"):
        # 检查输入
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} 输入无效，请检查")
        
        if not valid:
            return
            
        # 准备计算
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
            
            # 方程定义
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
            
            # 动态条件分支
            try:
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
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
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
                        # 记录材料信息
                        record['flyer_material'] = flyer_material
                        record['base_material'] = base_material
                        record['sample_material'] = sample_material
                        results.append(record)
            except Exception as e:
                st.warning(f"求解错误: {str(e)}")
        
        if results:
            st.success(f"求解完成，共找到 {len(results)} 个解")
            
            # 显示结果表格
            st.subheader("结果数据")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # 提供数据下载
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载结果数据",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            # 绘制图表
            st.subheader("结果可视化")
            plot_results_streamlit(results)
            
            # 保存到数据库选项
            st.subheader("保存结果")
            if st.button("保存结果到数据库"):
                # 使用主要材料名称（样品材料）作为保存的材料名称
                save_results_to_db(results, sample_material)
        else:
            st.warning("未找到有效解")
    
    # 返回主页按钮
    if st.button("返回主页"):
        st.session_state.page = "home"

def manual_mode_page():
    """手动输入模式页面"""
    st.title("手动输入模式")
    st.write("所有参数由用户输入，计算后可选择保存结果到数据库")
    
    # 参数输入（无默认值）
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "nubdaf", "E0f", "Ef", "uf", "w", "Pf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "nubdab", "E0b", "Eb", "ub", "Pb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "nubdas", "E0s", "Es", "us", "Ps"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片参数
    with st.expander("飞片相关参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["f"]):
            with cols[i % 3]:
                val = get_input_streamlit(f"{var}", var, var)
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数
    with st.expander("基板相关参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                val = get_input_streamlit(f"{var}", var, var)
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 样品参数
    with st.expander("样品相关参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                val = get_input_streamlit(f"{var}", var, var)
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 求解按钮
    if st.button("开始求解"):
        # 检查输入
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} 输入无效，请检查")
        
        if not valid:
            return
            
        # 准备计算
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
            
            # 方程定义
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
            
            # 动态条件分支
            try:
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
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
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
                st.warning(f"求解错误: {str(e)}")
        
        if results:
            st.success(f"求解完成，共找到 {len(results)} 个解")
            
            # 显示结果表格
            st.subheader("结果数据")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # 提供数据下载
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载结果数据",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            # 绘制图表
            st.subheader("结果可视化")
            plot_results_streamlit(results)
            
            # 保存到数据库选项
            st.subheader("保存结果")
            if st.button("保存结果到数据库"):
                save_results_to_db(results, "Custom Material")
        else:
            st.warning("未找到有效解")
    
    # 返回主页按钮
    if st.button("返回主页"):
        st.session_state.page = "home"

def main():
    """主应用函数"""
    # 初始化会话状态
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # 设置页面配置
    st.set_page_config(
        page_title="多物理场求解器",
        page_icon="✨",
        layout="wide"
    )
    
    # 页面导航
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "database_mode":
        database_mode_page()
    elif st.session_state.page == "manual_mode":
        manual_mode_page()

if __name__ == "__main__":
    main()
