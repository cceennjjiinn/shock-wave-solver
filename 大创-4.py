import math
import numpy as np
import pandas as pd
import streamlit as st
from sympy import symbols, Eq, simplify
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from io import BytesIO
import itertools
import sqlite3
from sqlalchemy import create_engine
import os

# 数据库配置
DB_TYPE = "sqlite"
DB_NAME = "shock_wave_data.db"
db_engine = None

# 数据库连接函数
def create_db_engine(db_type):
    """创建数据库引擎"""
    if db_type == "sqlite":
        return create_engine(f"sqlite:///{DB_NAME}")
    # 可以扩展其他数据库类型
    return None

def get_all_materials():
    """获取所有材料名称"""
    try:
        query = "SELECT DISTINCT material FROM shock_wave_all_data"
        df = pd.read_sql(query, db_engine)
        return df['material'].tolist()
    except Exception as e:
        st.warning(f"获取材料列表失败: {str(e)}")
        return []

def get_material_data(material, fields=None):
    """获取指定材料的数据"""
    try:
        if fields:
            columns = ", ".join(fields)
            query = f"SELECT {columns} FROM shock_wave_data WHERE material = ?"
        else:
            query = "SELECT * FROM shock_wave_data WHERE material = ?"
            
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql(query, conn, params=(material,))
        conn.close()
        return df
    except Exception as e:
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_input_parameters(params, material, source):
    """保存输入参数到数据库"""
    try:
        df = pd.DataFrame([params])
        df['material'] = material
        df['source'] = source
        df['timestamp'] = pd.Timestamp.now()
        
        # 只保存有值的参数
        df = df.dropna(axis=1, how='all')
        
        df.to_sql('shock_wave_data', db_engine, if_exists='append', index=False)
        return len(df)
    except Exception as e:
        st.error(f"保存参数失败: {str(e)}")
        return 0

def save_results_to_db(results, material):
    """保存计算结果到数据库"""
    try:
        df = pd.DataFrame(results)
        df['material'] = material
        df['source'] = 'calculated'
        df['timestamp'] = pd.Timestamp.now()
        
        # 只保存有值的参数
        df = df.dropna(axis=1, how='all')
        
        df.to_sql('shock_wave_data', db_engine, if_exists='append', index=False)
        return len(df)
    except Exception as e:
        st.error(f"保存结果失败: {str(e)}")
        return 0

def fit_material_data(df, material_name, material_type):
    """拟合材料的Hugoniot关系"""
    if df.empty:
        st.warning(f"{material_type}材料 {material_name} 没有可用数据，使用默认参数")
        return {
            'rho0': 8.96,  # 默认铜的密度
            'C0': 3.94,    # 默认铜的体声速
            'S': 1.48      # 默认铜的Hugoniot参数
        }
    
    # 计算平均初始密度
    rho0 = df['rho0'].mean() if 'rho0' in df.columns and not df['rho0'].isna().all() else 8.96
    
    # 拟合Hugoniot关系 Us = C0 + S*Up
    if 'Us' in df.columns and 'Up' in df.columns:
        valid_data = df.dropna(subset=['Us', 'Up'])
        valid_data = valid_data[(valid_data['Us'] > valid_data['Up']) & (valid_data['Us'] > 0) & (valid_data['Up'] >= 0)]
        
        if len(valid_data) >= 2:
            C0, S = fit_hugoniot(valid_data)
            st.success(f"{material_type}材料 {material_name} 拟合完成: Us = {C0:.2f} + {S:.2f}·Up, 初始密度 = {rho0:.2f} g/cm³")
            return {'rho0': rho0, 'C0': C0, 'S': S}
        else:
            st.warning(f"{material_type}材料 {material_name} 有效数据不足，使用默认参数")
    
    # 使用默认参数
    default_params = {
        '铜': {'rho0': 8.96, 'C0': 3.94, 'S': 1.48},
        '铝': {'rho0': 2.70, 'C0': 5.32, 'S': 1.37},
        '钢': {'rho0': 7.85, 'C0': 4.57, 'S': 1.49},
        '塑料': {'rho0': 1.15, 'C0': 2.50, 'S': 1.50},
        '陶瓷': {'rho0': 3.80, 'C0': 6.00, 'S': 1.60}
    }.get(material_name, {'rho0': 8.96, 'C0': 3.94, 'S': 1.48})
    
    st.info(f"{material_type}材料 {material_name} 使用默认参数: Us = {default_params['C0']:.2f} + {default_params['S']:.2f}·Up, 初始密度 = {default_params['rho0']:.2f} g/cm³")
    return default_params

def get_input_streamlit(label, var_name, key, default=None, unit="", desc="", disabled=False):
    """创建Streamlit输入框，支持范围输入"""
    col1, col2 = st.columns([3, 1])
    with col1:
        val = st.text_input(
            f"{label} ({unit})", 
            value=str(default) if default is not None else "",
            key=key,
            help=desc,
            disabled=disabled
        )
    
    # 解析输入值
    if not val.strip():
        return symbols(var_name)  # 返回符号表示未知数
    try:
        # 检查是否是范围输入
        if '-' in val:
            parts = val.split('-')
            if len(parts) == 2:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
                if start < end:
                    return np.linspace(start, end, 5).tolist()  # 生成5个点的范围
        # 单个值
        return float(val)
    except:
        st.error(f"{label} 输入格式错误，请输入数字或范围（如: 1-5）")
        return None

def view_database():
    """查看数据库内容"""
    st.title("数据库查看与管理")
    
    # 显示所有材料
    materials = get_all_materials()
    if not materials:
        st.info("数据库为空")
        return
    
    # 选择材料
    selected_material = st.selectbox("选择材料", materials)
    
    # 获取并显示数据
    df = get_material_data(selected_material)
    if not df.empty:
        st.subheader(f"{selected_material} 的数据记录 ({len(df)})")
        
        # 显示数据
        st.dataframe(df)
        
        # 数据下载
        csv = df.to_csv(index=False)
        st.download_button(
            label="下载数据",
            data=csv,
            file_name=f"{selected_material}_data.csv",
            mime="text/csv",
        )
        
        # 数据删除功能
        if st.button("删除所有数据", type="primary"):
            st.session_state.confirm_delete = True
            
        if st.session_state.confirm_delete:
            st.warning("确定要删除所有数据吗？此操作不可恢复！")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("确认删除"):
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM shock_wave_data WHERE material = ?", (selected_material,))
                        conn.commit()
                        conn.close()
                        st.success(f"已删除 {selected_material} 的所有数据")
                        st.session_state.confirm_delete = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {str(e)}")
            with col2:
                if st.button("取消"):
                    st.session_state.confirm_delete = False
    else:
        st.info(f"没有 {selected_material} 的数据")

# 冲击波参数计算（包含温度计算）
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300, calculate_temp=True):
    """根据Rankine-Hugoniot守恒关系计算冲击波参数，包含物理约束检查"""
    # 物理约束检查
    tolerance = 1e-3
    if U_s <= u_p - tolerance:
        raise ValueError(f"冲击波速度 (Us={U_s}) 必须大于粒子速度 (Up={u_p})")
    if rho0 <= tolerance:
        raise ValueError(f"初始密度 (rho0={rho0}) 必须为正数")
    if U_s <= tolerance or u_p < -tolerance:
        raise ValueError(f"速度参数必须非负，且冲击波速度必须为正数")
    
    # 动量守恒: P = rho0 * U_s * u_p
    P = rho0 * U_s * u_p
    
    # 质量守恒推导比体积: V = (1/rho0) * (1 - u_p/U_s)
    V = (1 / rho0) * (1 - u_p / U_s)
    
    # 压缩密度: rho = rho0 * U_s/(U_s - u_p)
    rho = rho0 * U_s / (U_s - u_p)
    
    # 比体积比: V/V0 = 1 - u_p/U_s
    V_V0 = V * rho0  # 由于V0 = 1/rho0，V/V0 = V * rho0
    
    # 检查计算结果的物理合理性
    if rho <= rho0 - tolerance:
        raise ValueError(f"计算的压缩密度 (rho={rho}) 必须大于初始密度 (rho0={rho0})")
    if V_V0 >= 1 + tolerance:
        raise ValueError(f"计算的比体积比 (V/V0={V_V0}) 必须小于1")
    if P <= -tolerance:
        raise ValueError(f"计算的压力 (P={P}) 必须为正数")
    
    T = None
    if calculate_temp:
        # 温度计算（Mie-Grüneisen方程近似）
        E_shock = 0.5 * P * (1/rho0 - V) * 1e6  # 冲击内能 (J/kg)
        T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
        
        if T < 100:
            raise ValueError(f"计算的冲击温度 (T={T}) 异常低")
    
    return P, V, rho, V_V0, T

# Hugoniot关系拟合
def fit_hugoniot(df):
    """拟合Hugoniot关系 Us = C0 + S*Up"""
    # 过滤物理上无效的数据
    tolerance = 1e-3
    df = df[(df['Us'] > df['Up'] - tolerance) & (df['Us'] > tolerance) & (df['Up'] >= -tolerance)]
    if len(df) < 2:
        return 0, 0  # 数据不足时返回默认值
        
    U_s = df['Us'].values
    u_p = df['Up'].values
    coeffs = np.polyfit(u_p, U_s, 1)
    S = coeffs[0]    # 斜率参数
    C0 = coeffs[1]   # 截距（零压声速）
    
    # 物理约束调整
    tolerance = 1e-3
    if C0 <= -tolerance:
        C0 = max(1.0, abs(C0))  # 确保体声速为正数且合理
        
    return C0, S

# 冲击波求解器类
class ShockWaveSolver:
    def __init__(self):
        # 初始化所有参数（None表示未知）
        self.params = {
            # 飞片参数
            'w': None, 'u_f': None, 'D_f': None, 'p_f': None,
            'rho0f': None, 'rho_f': None, 'C0f': None, 'S_f': None,
            # 样品参数
            'u_s': None, 'D_s': None, 'p_s': None,
            'rho0s': None, 'rho_s': None, 'C0s': None, 'S_s': None,
            # 基板参数
            'u_b': None, 'D_b': None, 'p_b': None,
            'rho0b': None, 'rho_b': None, 'C0b': None, 'S_b': None
        }
        
        # 定义方程列表 - 调整顺序优先处理关键变量对
        self.equations = [
            self.solve_uf_Df_pair,  # 联立求解u_f和D_f
            self.solve_us_Ds_pair,  # 联立求解u_s和D_s
            self.solve_ub_Db_pair,  # 联立求解u_b和D_b
            self.eq1,  # u_f = w - D_f (飞片速度关系)
            self.eq2,  # D_f = C0f + S_f * u_f (双向)
            self.eq3,  # p_f = rho0f * D_f * u_f (双向)
            self.eq4,  # rho_f = (rho0f * D_f) / (D_f - u_f)
            self.eq5,  # p_s = p_f (压力连续性)
            self.eq6,  # 样品二次方程求u_s (双向)
            self.eq7,  # D_s = C0s + S_s * u_s (双向)
            self.eq8,  # rho_s = (rho0s * D_s) / (D_s - u_s)
            self.eq9,  # p_b = p_s (压力连续性)
            self.eq10, # 基板二次方程求u_b (双向)
            self.eq11, # D_b = C0b + S_b * u_b (双向)
            self.eq12  # rho_b = (rho0b * D_b) / (D_b - u_b)
        ]
        
        # 参数依赖关系图，用于提示缺失参数
        self.dependency_graph = {
            'u_f': ['w', 'D_f', 'C0f', 'S_f', 'D_f', 'rho0f', 'p_f'],
            'D_f': ['w', 'u_f', 'C0f', 'S_f', 'u_f', 'rho0f', 'u_f', 'p_f'],
            'p_f': ['rho0f', 'D_f', 'u_f', 'p_s'],
            'w': ['u_f', 'D_f'],
            'rho_f': ['rho0f', 'D_f', 'u_f'],
            
            'u_s': ['rho0s', 'S_s', 'C0s', 'p_s', 'C0s', 'S_s', 'D_s'],
            'D_s': ['C0s', 'S_s', 'u_s', 'rho0s', 'u_s', 'rho_s'],
            'p_s': ['p_f', 'p_b', 'rho0s', 'S_s', 'C0s', 'u_s'],
            'rho_s': ['rho0s', 'D_s', 'u_s'],
            
            'u_b': ['rho0b', 'S_b', 'C0b', 'p_b', 'C0b', 'S_b', 'D_b'],
            'D_b': ['C0b', 'S_b', 'u_b', 'rho0b', 'u_b', 'rho_b'],
            'p_b': ['p_s', 'rho0b', 'S_b', 'C0b', 'u_b'],
            'rho_b': ['rho0b', 'D_b', 'u_b']
        }
        
        # 关键变量对及其联立方程信息
        self.variable_pairs = {
            ('u_f', 'D_f'): {
                'required_params': ['w', 'C0f', 'S_f'],
                'equation': lambda: self.solve_uf_Df_pair()
            },
            ('u_s', 'D_s'): {
                'required_params': ['p_s', 'rho0s', 'C0s', 'S_s'],
                'equation': lambda: self.solve_us_Ds_pair()
            },
            ('u_b', 'D_b'): {
                'required_params': ['p_b', 'rho0b', 'C0b', 'S_b'],
                'equation': lambda: self.solve_ub_Db_pair()
            }
        }
    
    def set_known_params(self, known_params):
        """设置已知参数（如{'w': 1000, 'C0f': 5000}）"""
        for key, value in known_params.items():
            if key in self.params:
                self.params[key] = value
            else:
                print(f"警告: 未知参数 {key} 被忽略")
    
    def suggest_missing_params(self):
        """分析并建议可能缺少的关键参数，包括联立求解提示"""
        unknown_params = [k for k, v in self.params.items() if v is None]
        suggestions = {}
        
        for param in unknown_params:
            if param in self.dependency_graph:
                possible_deps = [d for d in self.dependency_graph[param] if self.params[d] is None]
                if possible_deps:
                    suggestions[param] = possible_deps[:3]  # 取前3个可能的依赖
        
        # 添加联立求解提示
        for (var1, var2), info in self.variable_pairs.items():
            if self.params[var1] is None and self.params[var2] is None:
                missing_in_pair = [p for p in info['required_params'] if self.params[p] is None]
                if not missing_in_pair and all(self.params[p] is not None for p in info['required_params']):
                    suggestions[f"{var1} 和 {var2} (联立求解)"] = "这两个参数可通过已知参数联立求解"
                elif missing_in_pair:
                    suggestions[f"{var1} 和 {var2} (需补充参数)"] = f"需补充: {missing_in_pair} 才能联立求解"
        
        return suggestions
    
    def solve_uf_Df_pair(self):
        """联立求解u_f和D_f: u_f = w - D_f 和 D_f = C0f + S_f * u_f"""
        # 检查是否有足够的已知参数
        if (self.params['w'] is not None and 
            self.params['C0f'] is not None and 
            self.params['S_f'] is not None and 
            self.params['u_f'] is None and 
            self.params['D_f'] is None):
            
            # 联立方程求解
            denominator = 1 + self.params['S_f']
            if abs(denominator) < 1e-6:
                return False  # 避免除以零
            
            D_f_val = (self.params['w'] + self.params['C0f'] * self.params['S_f']) / denominator
            u_f_val = (self.params['w'] - self.params['C0f']) / denominator
            
            # 检查物理合理性
            if D_f_val > u_f_val - 1e-3 and D_f_val > 0 and u_f_val >= -1e-3:
                self.params['D_f'] = D_f_val
                self.params['u_f'] = u_f_val
                return True
        return False
    
    def solve_us_Ds_pair(self):
        """联立求解u_s和D_s: 动量守恒方程和Hugoniot关系"""
        if (self.params['p_s'] is not None and
            self.params['rho0s'] is not None and
            self.params['C0s'] is not None and
            self.params['S_s'] is not None and
            self.params['u_s'] is None and
            self.params['D_s'] is None):
            
            # 联立方程:
            # p_s = rho0s * D_s * u_s
            # D_s = C0s + S_s * u_s
            # 代入得: p_s = rho0s * (C0s + S_s * u_s) * u_s
            
            a = self.params['rho0s'] * self.params['S_s']
            b = self.params['rho0s'] * self.params['C0s']
            c = -self.params['p_s']
            delta = b**2 - 4*a*c
            
            if delta < 0:
                return False  # 无实根
                
            sqrt_delta = math.sqrt(delta)
            u1 = (-b + sqrt_delta) / (2*a)
            u2 = (-b - sqrt_delta) / (2*a)
            
            # 选择物理合理的正根
            u_s_val = u1 if u1 > 0 else u2
            if u_s_val <= 0:  # 确保粒子速度为正
                return False
                
            # 计算D_s并验证
            D_s_val = self.params['C0s'] + self.params['S_s'] * u_s_val
            if D_s_val <= u_s_val - 1e-3:  # 冲击波速度必须大于粒子速度
                return False
                
            self.params['u_s'] = u_s_val
            self.params['D_s'] = D_s_val
            return True
        return False
    
    def solve_ub_Db_pair(self):
        """联立求解u_b和D_b: 动量守恒方程和Hugoniot关系"""
        if (self.params['p_b'] is not None and
            self.params['rho0b'] is not None and
            self.params['C0b'] is not None and
            self.params['S_b'] is not None and
            self.params['u_b'] is None and
            self.params['D_b'] is None):
            
            # 联立方程:
            # p_b = rho0b * D_b * u_b
            # D_b = C0b + S_b * u_b
            # 代入得: p_b = rho0b * (C0b + S_b * u_b) * u_b
            
            a = self.params['rho0b'] * self.params['S_b']
            b = self.params['rho0b'] * self.params['C0b']
            c = -self.params['p_b']
            delta = b**2 - 4*a*c
            
            if delta < 0:
                return False  # 无实根
                
            sqrt_delta = math.sqrt(delta)
            u1 = (-b + sqrt_delta) / (2*a)
            u2 = (-b - sqrt_delta) / (2*a)
            
            # 选择物理合理的正根
            u_b_val = u1 if u1 > 0 else u2
            if u_b_val <= 0:  # 确保粒子速度为正
                return False
                
            # 计算D_b并验证
            D_b_val = self.params['C0b'] + self.params['S_b'] * u_b_val
            if D_b_val <= u_b_val - 1e-3:  # 冲击波速度必须大于粒子速度
                return False
                
            self.params['u_b'] = u_b_val
            self.params['D_b'] = D_b_val
            return True
        return False
    
    def eq1(self):
        # 飞片速度关系: u_f = w - D_f (双向)
        if self.params['w'] is not None and self.params['D_f'] is not None and self.params['u_f'] is None:
            self.params['u_f'] = self.params['w'] - self.params['D_f']
            return True
        if self.params['u_f'] is not None and self.params['D_f'] is not None and self.params['w'] is None:
            self.params['w'] = self.params['u_f'] + self.params['D_f']
            return True
        if self.params['w'] is not None and self.params['u_f'] is not None and self.params['D_f'] is None:
            self.params['D_f'] = self.params['w'] - self.params['u_f']
            return True
        return False
    
    def eq2(self):
        # D_f = C0f + S_f * u_f (Hugoniot关系) - 双向
        if self.params['C0f'] is not None and self.params['S_f'] is not None \
           and self.params['u_f'] is not None and self.params['D_f'] is None:
            self.params['D_f'] = self.params['C0f'] + self.params['S_f'] * self.params['u_f']
            return True
        # 已知D_f, C0f, u_f，求S_f
        if self.params['D_f'] is not None and self.params['C0f'] is not None \
           and self.params['u_f'] is not None and self.params['S_f'] is None \
           and abs(self.params['u_f']) > 1e-6:  # 避免除以零
            self.params['S_f'] = (self.params['D_f'] - self.params['C0f']) / self.params['u_f']
            return True
        # 已知D_f, S_f, u_f，求C0f
        if self.params['D_f'] is not None and self.params['S_f'] is not None \
           and self.params['u_f'] is not None and self.params['C0f'] is None:
            self.params['C0f'] = self.params['D_f'] - self.params['S_f'] * self.params['u_f']
            return True
        return False
    
    def eq3(self):
        # p_f = rho0f * D_f * u_f (动量守恒) - 双向
        # 已知rho0f, D_f, u_f，求p_f
        if self.params['rho0f'] is not None and self.params['D_f'] is not None \
           and self.params['u_f'] is not None and self.params['p_f'] is None:
            self.params['p_f'] = self.params['rho0f'] * self.params['D_f'] * self.params['u_f']
            return True
        # 已知p_f, rho0f, D_f，求u_f
        if self.params['p_f'] is not None and self.params['rho0f'] is not None \
           and self.params['D_f'] is not None and self.params['u_f'] is None \
           and abs(self.params['rho0f'] * self.params['D_f']) > 1e-6:  # 避免除以零
            self.params['u_f'] = self.params['p_f'] / (self.params['rho0f'] * self.params['D_f'])
            return True
        # 已知p_f, rho0f, u_f，求D_f
        if self.params['p_f'] is not None and self.params['rho0f'] is not None \
           and self.params['u_f'] is not None and self.params['D_f'] is None \
           and abs(self.params['rho0f'] * self.params['u_f']) > 1e-6:  # 避免除以零
            self.params['D_f'] = self.params['p_f'] / (self.params['rho0f'] * self.params['u_f'])
            return True
        return False
    
    def eq4(self):
        # rho_f = (rho0f * D_f) / (D_f - u_f) (质量守恒)
        if self.params['rho0f'] is not None and self.params['D_f'] is not None \
           and self.params['u_f'] is not None and self.params['rho_f'] is None \
           and abs(self.params['D_f'] - self.params['u_f']) > 1e-6:  # 避免除以零
            self.params['rho_f'] = (self.params['rho0f'] * self.params['D_f']) / (self.params['D_f'] - self.params['u_f'])
            return True
        return False
    
    def eq5(self):
        # p_s = p_f (压力连续性)
        if self.params['p_f'] is not None and self.params['p_s'] is None:
            self.params['p_s'] = self.params['p_f']
            return True
        if self.params['p_s'] is not None and self.params['p_f'] is None:
            self.params['p_f'] = self.params['p_s']
            return True
        return False
    
    def eq6(self):
        # 样品二次方程：rho0s*S_s*u_s² + rho0s*C0s*u_s - p_s = 0 (动量守恒)
        # 已知rho0s, S_s, C0s, p_s，求u_s
        if self.params['rho0s'] is not None and self.params['S_s'] is not None \
           and self.params['C0s'] is not None and self.params['p_s'] is not None \
           and self.params['u_s'] is None:
            a = self.params['rho0s'] * self.params['S_s']
            b = self.params['rho0s'] * self.params['C0s']
            c = -self.params['p_s']
            delta = b**2 - 4*a*c
            if delta < 0:
                return False  # 无实根
            sqrt_delta = math.sqrt(delta)
            u1 = (-b + sqrt_delta) / (2*a)
            u2 = (-b - sqrt_delta) / (2*a)
            # 取正根（物理意义合理）
            self.params['u_s'] = u1 if u1 > 0 else u2
            return True
        
        # 已知u_s, rho0s, C0s, p_s，求S_s
        if self.params['u_s'] is not None and self.params['rho0s'] is not None \
           and self.params['C0s'] is not None and self.params['p_s'] is not None \
           and self.params['S_s'] is None and abs(self.params['rho0s'] * self.params['u_s']**2) > 1e-6:
            self.params['S_s'] = (self.params['p_s'] - self.params['rho0s'] * self.params['C0s'] * self.params['u_s']) / \
                               (self.params['rho0s'] * self.params['u_s']**2)
            return True
        return False
    
    def eq7(self):
        # D_s = C0s + S_s * u_s (Hugoniot关系) - 双向
        # 已知C0s, S_s, u_s，求D_s
        if self.params['C0s'] is not None and self.params['S_s'] is not None \
           and self.params['u_s'] is not None and self.params['D_s'] is None:
            self.params['D_s'] = self.params['C0s'] + self.params['S_s'] * self.params['u_s']
            return True
        # 已知D_s, C0s, u_s，求S_s
        if self.params['D_s'] is not None and self.params['C0s'] is not None \
           and self.params['u_s'] is not None and self.params['S_s'] is None \
           and abs(self.params['u_s']) > 1e-6:  # 避免除以零
            self.params['S_s'] = (self.params['D_s'] - self.params['C0s']) / self.params['u_s']
            return True
        # 已知D_s, S_s, u_s，求C0s
        if self.params['D_s'] is not None and self.params['S_s'] is not None \
           and self.params['u_s'] is not None and self.params['C0s'] is None:
            self.params['C0s'] = self.params['D_s'] - self.params['S_s'] * self.params['u_s']
            return True
        return False
    
    def eq8(self):
        # rho_s = (rho0s * D_s) / (D_s - u_s) (质量守恒)
        if self.params['rho0s'] is not None and self.params['D_s'] is not None \
           and self.params['u_s'] is not None and self.params['rho_s'] is None \
           and abs(self.params['D_s'] - self.params['u_s']) > 1e-6:  # 避免除以零
            self.params['rho_s'] = (self.params['rho0s'] * self.params['D_s']) / (self.params['D_s'] - self.params['u_s'])
            return True
        return False
    
    def eq9(self):
        # p_b = p_s (压力连续性)
        if self.params['p_s'] is not None and self.params['p_b'] is None:
            self.params['p_b'] = self.params['p_s']
            return True
        if self.params['p_b'] is not None and self.params['p_s'] is None:
            self.params['p_s'] = self.params['p_b']
            return True
        return False
    
    def eq10(self):
        # 基板二次方程：rho0b*S_b*u_b² + rho0b*C0b*u_b - p_b = 0 (动量守恒)
        # 已知rho0b, S_b, C0b, p_b，求u_b
        if self.params['rho0b'] is not None and self.params['S_b'] is not None \
           and self.params['C0b'] is not None and self.params['p_b'] is not None \
           and self.params['u_b'] is None:
            a = self.params['rho0b'] * self.params['S_b']
            b = self.params['rho0b'] * self.params['C0b']
            c = -self.params['p_b']
            delta = b**2 - 4*a*c
            if delta < 0:
                return False  # 无实根
            sqrt_delta = math.sqrt(delta)
            u1 = (-b + sqrt_delta) / (2*a)
            u2 = (-b - sqrt_delta) / (2*a)
            # 取正根（物理意义合理）
            self.params['u_b'] = u1 if u1 > 0 else u2
            return True
        
        # 已知u_b, rho0b, C0b, p_b，求S_b
        if self.params['u_b'] is not None and self.params['rho0b'] is not None \
           and self.params['C0b'] is not None and self.params['p_b'] is not None \
           and self.params['S_b'] is None and abs(self.params['rho0b'] * self.params['u_b']**2) > 1e-6:
            self.params['S_b'] = (self.params['p_b'] - self.params['rho0b'] * self.params['C0b'] * self.params['u_b']) / \
                               (self.params['rho0b'] * self.params['u_b']**2)
            return True
        return False
    
    def eq11(self):
        # D_b = C0b + S_b * u_b (Hugoniot关系) - 双向
        # 已知C0b, S_b, u_b，求D_b
        if self.params['C0b'] is not None and self.params['S_b'] is not None \
           and self.params['u_b'] is not None and self.params['D_b'] is None:
            self.params['D_b'] = self.params['C0b'] + self.params['S_b'] * self.params['u_b']
            return True
        # 已知D_b, C0b, u_b，求S_b
        if self.params['D_b'] is not None and self.params['C0b'] is not None \
           and self.params['u_b'] is not None and self.params['S_b'] is None \
           and abs(self.params['u_b']) > 1e-6:  # 避免除以零
            self.params['S_b'] = (self.params['D_b'] - self.params['C0b']) / self.params['u_b']
            return True
        # 已知D_b, S_b, u_b，求C0b
        if self.params['D_b'] is not None and self.params['S_b'] is not None \
           and self.params['u_b'] is not None and self.params['C0b'] is None:
            self.params['C0b'] = self.params['D_b'] - self.params['S_b'] * self.params['u_b']
            return True
        return False
    
    def eq12(self):
        # rho_b = (rho0b * D_b) / (D_b - u_b) (质量守恒)
        if self.params['rho0b'] is not None and self.params['D_b'] is not None \
           and self.params['u_b'] is not None and self.params['rho_b'] is None \
           and abs(self.params['D_b'] - self.params['u_b']) > 1e-6:  # 避免除以零
            self.params['rho_b'] = (self.params['rho0b'] * self.params['D_b']) / (self.params['D_b'] - self.params['u_b'])
            return True
        return False
    
    def solve(self):
        """迭代求解所有可解参数，支持联立方程求解"""
        changed = True
        iteration = 0
        max_iterations = 200
        tolerance = 1e-3
        
        # 跟踪连续迭代无变化的次数
        no_change_count = 0
        
        while changed and iteration < max_iterations and no_change_count < 10:
            changed = False
            iteration += 1
            prev_params = self.params.copy()  # 保存当前状态用于比较
            
            # 优先处理联立方程求解
            for pair in self.variable_pairs.values():
                if pair['equation']():
                    changed = True
                    no_change_count = 0
            
            # 处理其他方程
            for eq in self.equations[3:]:  # 跳过前3个，已经处理过联立求解
                if eq():
                    changed = True
                    no_change_count = 0
                    
                    # 检查物理合理性
                    if self.params['D_f'] is not None and self.params['u_f'] is not None and self.params['D_f'] <= self.params['u_f'] - tolerance:
                        print(f"物理矛盾: D_f ({self.params['D_f']}) <= u_f ({self.params['u_f']})")
                        return None
                    if self.params['D_s'] is not None and self.params['u_s'] is not None and self.params['D_s'] <= self.params['u_s'] - tolerance:
                        print(f"物理矛盾: D_s ({self.params['D_s']}) <= u_s ({self.params['u_s']})")
                        return None
                    if self.params['D_b'] is not None and self.params['u_b'] is not None and self.params['D_b'] <= self.params['u_b'] - tolerance:
                        print(f"物理矛盾: D_b ({self.params['D_b']}) <= u_b ({self.params['u_b']})")
                        return None
            
            # 检查参数是否有实质变化
            if not changed:
                no_change_count += 1
        
        # 检查是否达到最大迭代次数
        if iteration >= max_iterations:
            print(f"警告: 达到最大迭代次数 ({max_iterations})，可能未完全求解")
            
        return self.params

# 数值求解器
def solve_numerically(eqs, sym_vars, initial_guess):
    """使用数值方法求解方程组，包含物理约束检查"""
    var_list = list(sym_vars.values())
    
    def residuals(x):
        """计算残差：方程组的误差"""
        substitutions = {var_list[i]: x[i] for i in range(len(x))}
        residuals = []
        for eq in eqs:
            substituted = eq.subs(substitutions)
            if substituted == True:
                residuals.append(0.0)  # 等式成立，残差为0
            elif substituted == False:
                residuals.append(1e10)  # 等式不成立，给予大残差
            else:
                try:
                    simplified = simplify(substituted)
                    residuals.append(float(abs(simplified.evalf())))
                except:
                    residuals.append(1e10)  # 计算失败时给予大残差
        return residuals
    
    # 设置边界
    n_vars = len(initial_guess)
    lower_bounds = [-1.0] * n_vars
    upper_bounds = [100.0] * n_vars
    
    # 根据变量类型调整边界
    for i, var in enumerate(initial_guess.keys()):
        var_str = str(var)
        if var_str.startswith(('rh0', 'rh')):  # 密度
            lower_bounds[i] = 0.01
            upper_bounds[i] = 50.0
        elif var_str.startswith(('D', 'C0', 'u', 'w')):  # 速度
            lower_bounds[i] = 0.01
            upper_bounds[i] = 100.0
        elif var_str.startswith(('P', 'E')):  # 压力/能量
            lower_bounds[i] = 0.001
            upper_bounds[i] = 10000.0
        elif var_str.startswith('gamma'):  # 格吕奈森系数
            lower_bounds[i] = 0.1
            upper_bounds[i] = 20.0
        elif var_str.startswith('T'):  # 温度
            lower_bounds[i] = 100.0
            upper_bounds[i] = 1e6
    
    # 执行最小二乘优化
    result = least_squares(
        residuals,
        list(initial_guess.values()),
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-6,
        gtol=1e-6,
        xtol=1e-6,
        max_nfev=10000,
        loss='soft_l1',
        f_scale=0.1
    )
    
    if result.success:
        solution = {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
        
        # 验证解的物理合理性
        tolerance = 1e-3
        valid = True
        
        # 检查冲击波速度大于粒子速度
        if 'Df' in solution and 'uf' in solution and solution['Df'] <= solution['uf'] - tolerance:
            valid = False
        if 'Db' in solution and 'ub' in solution and solution['Db'] <= solution['ub'] - tolerance:
            valid = False
        if 'Ds' in solution and 'us' in solution and solution['Ds'] <= solution['us'] - tolerance:
            valid = False
            
        # 检查压缩密度大于初始密度
        if 'rh0f' in solution and 'rhf' in solution and solution['rhf'] <= solution['rh0f'] - tolerance:
            valid = False
        if 'rh0b' in solution and 'rhb' in solution and solution['rhb'] <= solution['rh0b'] - tolerance:
            valid = False
        if 'rh0s' in solution and 'rhs' in solution and solution['rhs'] <= solution['rh0s'] - tolerance:
            valid = False
            
        # 检查压力为正数
        for p_var in ['Pf', 'Pb', 'Ps']:
            if p_var in solution and solution[p_var] <= -tolerance:
                valid = False
                break
                
        if not valid:
            # 尝试调整解使其满足物理约束
            adjusted = False
            if 'Df' in solution and 'uf' in solution and solution['Df'] <= solution['uf']:
                solution['Df'] = solution['uf'] + tolerance
                adjusted = True
            if 'Db' in solution and 'ub' in solution and solution['Db'] <= solution['ub']:
                solution['Db'] = solution['ub'] + tolerance
                adjusted = True
            if 'Ds' in solution and 'us' in solution and solution['Ds'] <= solution['us']:
                solution['Ds'] = solution['us'] + tolerance
                adjusted = True
                
            if adjusted:
                return solution
            return None
            
        return solution
    return None

def test_with_partial_params(known_params):
    """
    测试在部分参数已知的情况下求解器的表现
    known_params: 字典，包含已知参数及其值
    """
    print("\n=== 测试部分参数已知的情况 ===")
    print("已知参数:")
    for key, value in known_params.items():
        print(f"  {key}: {value}")
    
    try:
        solver = ShockWaveSolver()
        solver.set_known_params(known_params)
        result = solver.solve()
        
        if not result:
            print("求解失败，未得到任何结果")
            return
        
        # 分类参数：已知、求解成功、仍未知
        known = [k for k in known_params.keys() if k in result]
        solved = [k for k, v in result.items() if v is not None and k not in known]
        unknown = [k for k, v in result.items() if v is None]
        
        print("\n求解结果分析:")
        print(f"  已知参数: {len(known)}个")
        print(f"  成功求解: {len(solved)}个")
        print(f"  仍未知: {len(unknown)}个")
        
        if solved:
            print("\n求解得到的参数:")
            for key in solved:
                print(f"  {key}: {result[key]:.4f}")
        
        if unknown:
            print("\n未能求解的参数:")
            print(f"  {', '.join(unknown)}")
            
            # 提供参数建议
            suggestions = solver.suggest_missing_params()
            if suggestions:
                print("\n可能的缺失参数建议:")
                for param, possible in suggestions.items():
                    print(f"  {param}: {possible}")
        
        # 检查物理合理性
        print("\n物理合理性检查:")
        valid = True
        tolerance = 1e-3
        
        # 检查冲击波速度 > 粒子速度
        for param_pair in [('D_f', 'u_f'), ('D_s', 'u_s'), ('D_b', 'u_b')]:
            D, u = param_pair
            if D in solved and u in solved:
                if result[D] <= result[u] - tolerance:
                    print(f"  不合理: {D} ({result[D]:.4f}) <= {u} ({result[u]:.4f})")
                    valid = False
                else:
                    print(f"  合理: {D} ({result[D]:.4f}) > {u} ({result[u]:.4f})")
        
        # 检查压力为正
        for p in ['p_f', 'p_s', 'p_b']:
            if p in solved:
                if result[p] <= -tolerance:
                    print(f"  不合理: {p} ({result[p]:.4f}) 为非正数")
                    valid = False
                else:
                    print(f"  合理: {p} 为正数")
        
        if valid:
            print("\n所有已求解参数满足物理合理性约束")
    
    except Exception as e:
        print(f"测试失败: {str(e)}")

def test_core_functions():
    print("测试冲击波参数计算...")
    try:
        # 测试正常情况
        P, V, rho, V_V0, T = calculate_shock_parameters(U_s=5.0, u_p=1.0, rho0=8.96)
        print(f"计算结果 - P: {P:.2f} GPa, rho: {rho:.2f} g/cm³, V/V0: {V_V0:.2f}")
        
        # 测试物理约束违反情况
        try:
            calculate_shock_parameters(U_s=1.0, u_p=2.0, rho0=8.96)  # Us < Up
            print("错误: 未检测到冲击波速度小于粒子速度的情况")
        except ValueError as e:
            print(f"正确检测到错误: {str(e)}")
            
        try:
            calculate_shock_parameters(U_s=5.0, u_p=1.0, rho0=-0.1)  # 密度为负
            print("错误: 未检测到密度为负的情况")
        except ValueError as e:
            print(f"正确检测到错误: {str(e)}")
            
    except Exception as e:
        print(f"冲击波参数计算测试失败: {str(e)}")

# 冲击波关系图绘制 - 使用英文标签，根据实验方法区分颜色
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
    fig.suptitle(f'Material: {material_name} - Shock Wave Relationships', fontsize=16)
    
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
    
    # 确保Hugoniot参数合理
    if C0 <= 0:
        C0 = 3.0  # 默认合理值
    if S < 1.0 or S > 3.0:
        S = 1.5  # 默认合理值
        
    u_p_range = np.linspace(0, min(20, df['Up'].max()*1.1), 100)  # 限制在物理合理范围内
    U_s_fit = C0 + S * u_p_range  # Hugoniot关系
    
    # 移除Us = Up线的绘制
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
    P_range = rho0 * U_s_fit * u_p_range  # 动量守恒关系 P = ρ0·Us·Up
    
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
    
    # 添加物理约束线：V/V0 = 1（比体积比必须小于1）
    axs[1, 0].axvline(x=1.0, color='k', linestyle='--', label='V/V0 = 1 (physical boundary)')
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
    
    # 添加初始密度参考线
    axs[1, 1].axhline(y=rho0, color='k', linestyle='--', label=f'Initial density: {rho0:.2f} g/cm³')
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
        with st.expander(f"查看 {material_type} 材料 {material_name} 的冲击波关系图", expanded=True):
            C0, S = fit_hugoniot(df)
            fig = generate_shock_plots(df, C0, S, material_name, material_type)
            st.pyplot(fig)
            buf = save_plot_to_bytes(fig)
            
            material_type_en = {
                "飞片": "flyer",
                "基板": "substrate",
                "样品": "sample"
            }.get(material_type, material_type.lower())
            download_label = f"下载 {material_type} 材料 {material_name} 的冲击波关系图"
            file_name = f"{material_type_en}_{material_name}_shock_relations.png"
                
            st.download_button(
                label=download_label,
                data=buf,
                file_name=file_name,
                mime="image/png"
            )
    else:
        st.info(f"没有可用数据生成 {material_type} 材料 {material_name} 的图表")

# 结果绘图函数 - 使用英文标签
@st.cache_data(ttl=3600)  # 缓存图像结果
def plot_results_streamlit(results, calculate_temp=True):
    if not results:
        return None
        
    # 数据量大时进行采样
    if len(results) > 1000:
        results = results[:1000]
        
    # 确定子图数量
    subplot_count = 4 if calculate_temp else 3
    fig = plt.figure(figsize=(18, 9) if calculate_temp else (18, 7))
    
    # 原始数据
    pf_values = [r.get('Pf', 0) for r in results]
    uf_values = [r.get('uf', 0) for r in results]
    df_values = [r.get('Df', 0) for r in results]
    rhf_values = [r.get('rhf', 0) for r in results]
    
    # 1. 压力-粒子速度图（带误差棒）
    ax1 = fig.add_subplot(221 if calculate_temp else 221)
    ax1.errorbar(uf_values, pf_values, 
                 yerr=[r.get('Pf_err', 0.1) for r in results],
                 xerr=[r.get('uf_err', 0.05) for r in results],
                 fmt='bo', ecolor='r', capsize=5, label='Flyer data')
    ax1.set_xlabel('Particle Velocity Up (km/s)')
    ax1.set_ylabel('Shock Pressure P (GPa)')
    ax1.set_title('Pressure-Particle Velocity Relationship (with error range)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 温度-压力图（仅当计算温度时显示）
    ax2 = None
    if calculate_temp:
        # 温度相关数据
        tf_values = [r.get('Tf', 0) for r in results]
        tb_values = [r.get('Tb', 0) for r in results]
        ts_values = [r.get('Ts', 0) for r in results]
        
        ax2 = fig.add_subplot(222)
        ax2.scatter(pf_values, tf_values, c='orange', label='Flyer temperature')
        # 添加室温参考线
        ax2.axhline(y=300, color='k', linestyle='--', label='Room temperature (300 K)')
        ax2.set_xlabel('Shock Pressure P (GPa)')
        ax2.set_ylabel('Shock Temperature T (K)')
        ax2.set_title('Temperature-Pressure Relationship')
        ax2.legend()
        ax2.grid(True)
    
    # 3. 冲击波速度-粒子速度图
    ax3 = fig.add_subplot(223 if calculate_temp else 222)
    ax3.scatter(uf_values, df_values, c='blue', label='Flyer')
    # 移除Us = Up线的绘制
    ax3.set_xlabel('Particle Velocity Up (km/s)')
    ax3.set_ylabel('Shock Wave Velocity Us (km/s)')
    ax3.set_title('Shock Wave Velocity-Particle Velocity Relationship')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 密度-压力图
    ax4 = fig.add_subplot(224 if calculate_temp else 223)
    ax4.scatter(pf_values, rhf_values, c='green', label='Flyer')
    # 添加初始密度参考线（如果有）
    if results and 'rh0f' in results[0]:
        avg_rh0 = np.mean([r.get('rh0f', 0) for r in results])
        ax4.axhline(y=avg_rh0, color='k', linestyle='--', label=f'Avg initial density: {avg_rh0:.2f} g/cm³')
    ax4.set_xlabel('Shock Pressure P (GPa)')
    ax4.set_ylabel('Compressed Density (g/cm³)')
    ax4.set_title('Density-Pressure Relationship')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

# 页面函数
def home_page():
    # 记录当前页面，用于返回功能
    st.session_state.previous_page = "home"
    st.title("冲击波参数计算与分析系统")
    st.info("""
    系统核心模型说明：
    1. 基于Rankine-Hugoniot守恒方程组（质量、动量、能量守恒）
    2. 假设条件：平面冲击波、稳定传播、初始压力可忽略
    3. 单位系统：密度(g/cm³)、速度(km/s)、压力(GPa)
    4. 关键物理约束：
       - 冲击波速度(Us) > 粒子速度(Up)
       - 压缩密度(ρ) > 初始密度(ρ₀)
       - 比体积比(V/V₀) < 1
       - 冲击压力(P) = ρ₀·Us·Up（动量守恒）
       - Hugoniot关系：Us = C₀ + S·Up（C₀为体声速，S通常在1.3-2.0之间）
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
    # 记录当前页面，用于返回功能
    st.session_state.previous_page = "database_mode"
    st.title("数据库模式")
    st.write("从数据库加载材料数据，基于Hugoniot关系拟合参数并求解")
    st.success("提示：将参数留空将由系统根据物理规律自动求解")
    
    # 添加温度计算选项
    calculate_temp = st.checkbox("进行温度相关计算", value=True, 
                                 help="勾选以计算冲击温度，需要格吕奈森系数和比热容参数")
    
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
    
    # 按需查询字段以减少数据传输，确保包含exp_method
    flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
    base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
    sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
    
    # 为每种材料类型拟合数据并清晰标注
    with st.spinner(f"正在拟合飞片材料 {flyer_material} 数据..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material, "飞片")
    
    with st.spinner(f"正在拟合基板材料 {base_material} 数据..."):
        base_fit = fit_material_data(base_df, base_material, "基板")
    
    with st.spinner(f"正在拟合样品材料 {sample_material} 数据..."):
        sample_fit = fit_material_data(sample_df, sample_material, "样品")
    
    # 冲击波参数分析部分，为每种材料单独绘图
    st.subheader("冲击波参数分析（Hugoniot关系）")
    st.caption("""
    基于线性Hugoniot关系 Us = C0 + S·Up：
    - C0：体声速（零压下的声速，单位：km/s）
    - S：Hugoniot参数（描述冲击波速度随粒子速度的变化，无量纲）
    - 物理约束：Us > Up，S通常在1.3-2.0之间
    - 数据点颜色编码：iml(红色)，ssp(蓝色)，计算值(绿色)，手动输入(紫色)，批量导入(橙色)
    """)
    
    # 为每种材料类型显示单独的图像
    display_material_plots(flyer_df, flyer_material, "飞片")
    display_material_plots(base_df, base_material, "基板")
    display_material_plots(sample_df, sample_material, "样品")
    
    default_params = {"f": flyer_fit, "b": base_fit, "s": sample_fit}
    # 参数定义
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片与基板界面速度关系说明 - 更新为正确的物理关系
    st.info("""
    飞片冲击关系：飞片速度w与粒子速度uf的关系为w = Df + uf
    这是从实验室坐标系中的运动学关系和质量守恒方程推导得出的
    """)
    
    # 比热容设置（仅当计算温度时显示）
    Cv_values = {}
    if calculate_temp:
        st.subheader("比热容设置（用于温度计算）")
        col1, col2, col3 = st.columns(3)
        with col1:
            Cv_values['f'] = st.number_input(f"飞片比热容 Cv (J/(kg·K)) ({flyer_material})", 
                                            value=385.0, min_value=1.0, help="铜约为385，铝约为900")
        with col2:
            Cv_values['b'] = st.number_input(f"基板比热容 Cv (J/(kg·K)) ({base_material})", 
                                            value=385.0, min_value=1.0)
        with col3:
            Cv_values['s'] = st.number_input(f"样品比热容 Cv (J/(kg·K)) ({sample_material})", 
                                            value=385.0, min_value=1.0)
    
    # 飞片参数
    with st.expander(f"{flyer_material} 飞片参数", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "初始密度（必须输入）",
            "rhf": "压缩密度",
            "Df": "冲击波速度（对应Us）",
            "C0f": "体声速（Hugoniot拟合）",
            "Sf": "Hugoniot参数S（无量纲）",
            "E0f": "初始内能密度",
            "Ef": "压缩后内能密度",
            "uf": "粒子速度（对应Up）",
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
            # 温度参数仅在计算温度时显示
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
                    # 尝试从数据中获取平均格吕奈森系数
                    if not flyer_df.empty and 'gamma' in flyer_df.columns:
                        gamma_vals = flyer_df['gamma'].dropna()
                        if len(gamma_vals) > 0:
                            default_val = gamma_vals.mean()
                        else:
                            default_val = 2.0
                    else:
                        default_val = 2.0  # 默认格吕奈森系数
                # 为初始密度设置默认值和更强的提示
                if var == "rh0f" and default_val is None:
                    default_val = 8.96  # 铜的默认密度
                
                val = get_input_streamlit(
                    label=var,
                    var_name=var,
                    key=f"f_{var}",
                    default=default_val,
                    unit=var_units[var],
                    desc=var_descs[var],
                    # 初始密度不允许留空，强制要求输入
                    disabled=False
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数 - 所有参数都需要独立输入
    with st.expander(f"{base_material} 基板参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            # 温度参数仅在计算温度时显示
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
                    # 尝试从数据中获取平均格吕奈森系数
                    if not base_df.empty and 'gamma' in base_df.columns:
                        gamma_vals = base_df['gamma'].dropna()
                        if len(gamma_vals) > 0:
                            default_val = gamma_vals.mean()
                        else:
                            default_val = 2.0
                    else:
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
    
    # 样品参数 - 所有参数都需要独立输入
    with st.expander(f"{sample_material} 样品参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            # 温度参数仅在计算温度时显示
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
                    # 尝试从数据中获取平均格吕奈森系数
                    if not sample_df.empty and 'gamma' in sample_df.columns:
                        gamma_vals = sample_df['gamma'].dropna()
                        if len(gamma_vals) > 0:
                            default_val = gamma_vals.mean()
                        else:
                            default_val = 2.0
                    else:
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
    
    # 固定显示保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("保存当前参数到数据库"):
            count = save_input_parameters(input_params, sample_material, "database_mode_input")
            if count > 0:
                st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "最大参数组合数（过多会影响速度）", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("开始求解"):
        valid = True
        # 检查关键参数是否已输入 - 使用Symbol类进行类型检查
        for var in ['rh0f', 'rh0b', 'rh0s']:
            if isinstance(input_params.get(var), symbols(var).__class__):
                valid = False
                st.error(f"{var}（初始密度）是必填参数，请输入值")
        
        # 检查其他参数输入有效性
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} 输入无效，请检查")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # 截断过多的组合
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"参数组合过多（{len(combinations)}），为提高速度已截断至 {max_combinations} 个")
            combinations = combinations[:max_combinations]
        
        results = []
        progress_bar = st.progress(0)
        total = len(combinations)
        count = 0
        invalid_solutions = 0
        
        for combo in combinations:
            count += 1
            # 每10次更新一次进度条以减少UI开销
            if count % 10 == 0 or count == total:
                progress_bar.progress(count / total)
                
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # 方程组 - 使用修正后的飞片速度方程
            eqs = [
                # 飞片质量守恒: rho0f·Df = rhf·(Df - uf)
                Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # 修正：飞片速度与粒子速度关系 (实验室坐标系): w = Df + uf
                Eq(sym_vars['w'] - (sym_vars['Df'] + sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·uf  (修正：使用标准动量守恒公式)
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf'], 0),
                # 基板动量守恒: Pb = rho0b·Db·ub  (修正：使用标准动量守恒公式)
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # 飞片能量守恒: Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # 基板能量守恒: Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # 飞片Hugoniot关系: Df = C0f + Sf·uf  (修正：使用标准Hugoniot关系)
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*sym_vars['uf'], 0),
                # 基板Hugoniot关系: Db = C0b + Sb·ub  (修正：使用标准Hugoniot关系)
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                # 界面压力连续性: Pf = Pb
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # 界面粒子速度连续性: uf = ub
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            # 温度相关方程（仅当计算温度时添加）
            if calculate_temp:
                # 飞片温度方程 (Mie-Grüneisen)
                eqs.append(Eq(sym_vars['Tf'] - 300 - (sym_vars['Ef'] - sym_vars['E0f'])*1e6 / 
                             (Cv_values['f'] * (1 + sym_vars['gammaf']/2)), 0))
                # 基板温度方程
                eqs.append(Eq(sym_vars['Tb'] - 300 - (sym_vars['Eb'] - sym_vars['E0b'])*1e6 / 
                             (Cv_values['b'] * (1 + sym_vars['gammab']/2)), 0))
            
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
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
                ]
                
                # 温度相关方程（仅当计算温度时添加）
                if calculate_temp:
                    eqs += [
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
                
                # 温度相关方程（仅当计算温度时添加）
                if calculate_temp:
                    eqs.append(Eq(sym_vars['Ts'] - 300 - (sym_vars['Es'] - sym_vars['E0s'])*1e6 / 
                                 (Cv_values['s'] * (1 + sym_vars['gammas']/2)), 0))
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # 构建初始猜测值（基于物理合理范围和已知参数）
                initial_guess = {}
                # 提取已知参数值用于更智能的初始猜测
                known_params = {}
                for k, v in current_subs.items():
                    try:
                        known_params[str(k)] = float(v)
                    except:
                        pass
                
                for var in remaining_vars:
                    var_str = str(var)
                    # 基于已知参数动态设置初始猜测值
                    if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['Df'] + known_params['uf']
                    elif var_str == 'Df' and 'w' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['w'] - known_params['uf']
                    elif var_str == 'uf' and 'w' in known_params and 'Df' in known_params:
                        initial_guess[var] = known_params['w'] - known_params['Df']
                    elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf']
                    elif var_str.startswith(('rh0', 'rh')):  # 密度
                        initial_guess[var] = known_params.get('rh0f', 8.0)  # 使用已知密度作为参考
                    elif var_str.startswith(('D', 'C0', 'u')):  # 速度
                        if 'w' in known_params:
                            initial_guess[var] = known_params['w'] / 2  # 基于飞片速度估算
                        else:
                            initial_guess[var] = 5.0
                    elif var_str == 'w':  # 飞片速度
                        initial_guess[var] = 10.0
                    elif var_str.startswith('P'):  # 压力
                        if 'rh0f' in known_params and 'w' in known_params:
                            # 基于飞片速度估算压力
                            initial_guess[var] = known_params['rh0f'] * (known_params['w']/2) * (known_params['w']/2)
                        else:
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
                else:
                    invalid_solutions += 1
            except Exception as e:
                st.warning(f"求解错误: {str(e)}（可能由高压下的非线性效应引起，请检查参数范围）")
                invalid_solutions += 1
        
        if results:
            st.success(f"求解完成，找到 {len(results)} 个符合物理规律的解（已过滤 {invalid_solutions} 个不合理解）")
            
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
            fig = plot_results_streamlit(results, calculate_temp)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="下载图表",
                    data=buf2,
                    file_name="analysis_with_temp_error.png" if calculate_temp else "analysis_results.png",
                    mime="image/png"
                )
            
            if st.button("保存结果到数据库"):
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
        else:
            st.warning(f"未找到有效解，尝试了 {total} 组参数，均不符合物理规律或求解失败")
            # 显示更多调试信息
            st.info("尝试以下解决方案：\n1. 检查输入参数是否在合理范围内\n2. 减少未知数数量，输入更多已知参数\n3. 放宽物理约束条件\n4. 调整参数范围，避免极端值")
    
    if st.button("返回主页"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

def manual_mode_page():
    # 记录当前页面，用于返回功能
    st.session_state.previous_page = "manual_mode"
    st.title("手动输入模式")
    st.write("通过手动输入参数进行求解，适用于没有数据库数据的场景")
    st.success("提示：将参数留空将由系统根据物理规律自动求解")
    
    # 添加温度计算选项
    calculate_temp = st.checkbox("进行温度相关计算", value=True, 
                                 help="勾选以计算冲击温度，需要格吕奈森系数和比热容参数")
    
    # 查看数据库快捷入口
    if st.button("查看数据库"):
        st.session_state.page = "view_database"
        st.rerun()
    
    # 材料参数输入 - 已统一为中文
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.text_input("飞片材料名称", value="铜", help="输入材料名称，例如：铜、铝")
    with col2:
        base_material = st.text_input("基板材料名称", value="铝", help="输入材料名称，例如：铜、铝")
    with col3:
        sample_material = st.text_input("样品材料名称", value="塑料", help="输入材料名称，例如：塑料、陶瓷")

    # 比热容设置（仅当计算温度时显示）
    Cv_values = {}
    if calculate_temp:
        st.subheader("比热容设置（用于温度计算）")
        col_cv1, col_cv2, col_cv3 = st.columns(3)
        with col_cv1:
            Cv_values['f'] = st.number_input(
                f"{flyer_material} 比热容 Cv (J/(kg·K))", 
                value=385.0, min_value=1.0, 
                help="铜约为385，铝约为900，塑料约为1000"
            )
        with col_cv2:
            Cv_values['b'] = st.number_input(
                f"{base_material} 比热容 Cv (J/(kg·K))", 
                value=900.0, min_value=1.0
            )
        with col_cv3:
            Cv_values['s'] = st.number_input(
                f"{sample_material} 比热容 Cv (J/(kg·K))", 
                value=1000.0, min_value=1.0
            )

    # 参数定义
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }

    input_params = {}
    sym_vars = {}

    # 飞片与基板界面速度关系说明
    st.info("""
    飞片冲击关系：飞片速度w与粒子速度uf的关系为w = Df + uf
    这是从实验室坐标系中的运动学关系和质量守恒方程推导得出的
    """)

    # 飞片参数
    with st.expander(f"{flyer_material} 飞片参数", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "初始密度（必须输入）",
            "rhf": "压缩密度",
            "Df": "冲击波速度（对应Us）",
            "C0f": "体声速（典型值：铜~3.9，铝~5.3）",
            "Sf": "Hugoniot参数S（无量纲，典型值1.3-2.0）",
            "E0f": "初始内能密度（可留空，默认0）",
            "Ef": "压缩后内能密度",
            "uf": "粒子速度（对应Up）",
            "w": "飞片初始冲击速度",
            "Pf": "冲击压力",
            "gammaf": "格吕奈森系数（典型值1.0-3.0）",
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
        # 典型材料参数预设
        typical_params = {
            "铜": {"rh0f": 8.96, "C0f": 3.94, "Sf": 1.48, "gammaf": 2.0},
            "铝": {"rh0f": 2.70, "C0f": 5.32, "Sf": 1.37, "gammaf": 2.1},
            "钢": {"rh0f": 7.85, "C0f": 4.57, "Sf": 1.49, "gammaf": 1.9},
            "塑料": {"rh0f": 1.15, "C0f": 2.50, "Sf": 1.50, "gammaf": 1.5},
            "陶瓷": {"rh0f": 3.80, "C0f": 6.00, "Sf": 1.60, "gammaf": 1.7}
        }
        # 获取当前材料的典型参数
        flyer_typical = typical_params.get(flyer_material, {})

        for i, var in enumerate(variables["f"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                # 设置典型值
                default_val = flyer_typical.get(var, None)
                # 初始内能默认0
                if var == "E0f" and default_val is None:
                    default_val = 0.0
                
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
        # 获取当前材料的典型参数
        base_typical = typical_params.get(base_material, {})

        for i, var in enumerate(variables["b"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                # 转换变量名以匹配典型参数键名
                param_key = var.replace('b', 'f')  # rh0b -> rh0f
                default_val = base_typical.get(param_key, None)
                # 初始内能默认0
                if var == "E0b" and default_val is None:
                    default_val = 0.0
                
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

    # 样品参数
    with st.expander(f"{sample_material} 样品参数", expanded=True):
        cols = st.columns(3)
        # 获取当前材料的典型参数
        sample_typical = typical_params.get(sample_material, {})

        for i, var in enumerate(variables["s"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                # 转换变量名以匹配典型参数键名
                param_key = var.replace('s', 'f')  # rh0s -> rh0f
                default_val = sample_typical.get(param_key, None)
                # 初始内能默认0
                if var == "E0s" and default_val is None:
                    default_val = 0.0
                
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
    
    # 固定显示保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("保存当前参数到数据库"):
            # 整合材料信息到参数中
            params_with_materials = input_params.copy()
            params_with_materials['flyer_material'] = flyer_material
            params_with_materials['base_material'] = base_material
            params_with_materials['sample_material'] = sample_material
            
            count = save_input_parameters(params_with_materials, sample_material, "manual_mode_input")
            if count > 0:
                st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "最大参数组合数（过多会影响速度）", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("开始求解"):
        valid = True
        # 检查关键参数是否已输入
        for var in ['rh0f', 'rh0b', 'rh0s']:
            if isinstance(input_params.get(var), symbols(var).__class__):
                valid = False
                st.error(f"{var}（初始密度）是必填参数，请输入值")
        
        # 检查其他参数输入有效性
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"{var} 输入无效，请检查")
        
        if not valid:
            return
            
        # 生成参数组合
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # 截断过多的组合
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"参数组合过多（{len(combinations)}），为提高速度已截断至 {max_combinations} 个")
            combinations = combinations[:max_combinations]
        
        results = []
        progress_bar = st.progress(0)
        total = len(combinations)
        count = 0
        invalid_solutions = 0
        
        for combo in combinations:
            count += 1
            # 每10次更新一次进度条以减少UI开销
            if count % 10 == 0 or count == total:
                progress_bar.progress(count / total)
                
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # 方程组 - 使用修正后的飞片速度方程
            eqs = [
                # 飞片质量守恒: rho0f·Df = rhf·(Df - uf)
                Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # 修正：飞片速度与粒子速度关系 (实验室坐标系): w = Df + uf
                Eq(sym_vars['w'] - (sym_vars['Df'] + sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·uf
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf'], 0),
                # 基板动量守恒: Pb = rho0b·Db·ub
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # 飞片能量守恒: Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # 基板能量守恒: Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # 飞片Hugoniot关系: Df = C0f + Sf·uf
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*sym_vars['uf'], 0),
                # 基板Hugoniot关系: Db = C0b + Sb·ub
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                # 界面压力连续性: Pf = Pb
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # 界面粒子速度连续性: uf = ub
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            # 温度相关方程（仅当计算温度时添加）
            if calculate_temp:
                # 飞片温度方程 (Mie-Grüneisen)
                eqs.append(Eq(sym_vars['Tf'] - 300 - (sym_vars['Ef'] - sym_vars['E0f'])*1e6 / 
                             (Cv_values['f'] * (1 + sym_vars['gammaf']/2)), 0))
                # 基板温度方程
                eqs.append(Eq(sym_vars['Tb'] - 300 - (sym_vars['Eb'] - sym_vars['E0b'])*1e6 / 
                             (Cv_values['b'] * (1 + sym_vars['gammab']/2)), 0))
            
            try:
                # 检查样品和基板是否为同一材料
                cond = (flyer_material == base_material == sample_material)
            except:
                cond = False
                
            if cond:
                # 样品与基板为同一材料：参数与基板一致
                eqs += [
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Eq(sym_vars['ub'] - sym_vars['us'], 0),  # 速度连续性
                    Eq(sym_vars['rhb'] - sym_vars['rhs'], 0), # 密度连续性
                    Eq(sym_vars['Db'] - sym_vars['Ds'], 0),  # 冲击波速度连续性
                    # 样品能量守恒
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
                ]
                
                # 温度相关方程（仅当计算温度时添加）
                if calculate_temp:
                    eqs += [
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
                
                # 温度相关方程（仅当计算温度时添加）
                if calculate_temp:
                    eqs.append(Eq(sym_vars['Ts'] - 300 - (sym_vars['Es'] - sym_vars['E0s'])*1e6 / 
                                 (Cv_values['s'] * (1 + sym_vars['gammas']/2)), 0))
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # 构建初始猜测值
                initial_guess = {}
                # 提取已知参数值
                known_params = {}
                for k, v in current_subs.items():
                    try:
                        known_params[str(k)] = float(v)
                    except:
                        pass
                
                for var in remaining_vars:
                    var_str = str(var)
                    # 基于已知参数动态设置初始猜测值
                    if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['Df'] + known_params['uf']
                    elif var_str == 'Df' and 'w' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['w'] - known_params['uf']
                    elif var_str == 'uf' and 'w' in known_params and 'Df' in known_params:
                        initial_guess[var] = known_params['w'] - known_params['Df']
                    elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf']
                    elif var_str.startswith(('rh0', 'rh')):  # 密度
                        initial_guess[var] = known_params.get('rh0f', 8.0)
                    elif var_str.startswith(('D', 'C0', 'u')):  # 速度
                        if 'w' in known_params:
                            initial_guess[var] = known_params['w'] / 2
                        else:
                            initial_guess[var] = 5.0
                    elif var_str == 'w':  # 飞片速度
                        initial_guess[var] = 10.0
                    elif var_str.startswith('P'):  # 压力
                        if 'rh0f' in known_params and 'w' in known_params:
                            initial_guess[var] = known_params['rh0f'] * (known_params['w']/2) * (known_params['w']/2)
                        else:
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
                else:
                    invalid_solutions += 1
            except Exception as e:
                st.warning(f"求解错误: {str(e)}（可能由高压下的非线性效应引起，请检查参数范围）")
                invalid_solutions += 1
        
        if results:
            st.success(f"求解完成，找到 {len(results)} 个符合物理规律的解（已过滤 {invalid_solutions} 个不合理解）")
            
            st.subheader("结果数据（单位：rho=g/cm³, D=km/s, u=km/s, P=GPa, T=K）")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载结果数据",
                data=csv,
                file_name="manual_mode_solver_results.csv",
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
                    file_name="manual_analysis_with_temp.png" if calculate_temp else "manual_analysis_results.png",
                    mime="image/png"
                )
            
            if st.button("保存结果到数据库"):
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"已保存到 {sample_material} 数据集，共 {count} 条记录")
        else:
            st.warning(f"未找到有效解，尝试了 {total} 组参数，均不符合物理规律或求解失败")
            st.info("尝试以下解决方案：\n1. 检查输入参数是否在合理范围内\n2. 减少未知数数量，输入更多已知参数\n3. 放宽物理约束条件\n4. 调整参数范围，避免极端值")
    
    if st.button("返回主页"):
        st.session_state.page = "home"
        st.rerun()

# 主函数
def main():
    # 初始化会话状态
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False
    if 'previous_page' not in st.session_state:
        st.session_state.previous_page = None
    
    # 初始化数据库引擎
    global db_engine
    db_engine = create_db_engine(DB_TYPE)
    
    # 设置页面配置
    st.set_page_config(
        page_title="冲击波参数计算与分析系统",
        page_icon="⚡",
        layout="wide"
    )
    
    # 页面导航
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
                if st.session_state.previous_page:
                    st.session_state.page = st.session_state.previous_page
                else:
                    st.session_state.page = "home"
                st.rerun()

if __name__ == "__main__":
    # 运行测试
    # test_core_functions()
    # test_with_partial_params({'w': 5.0, 'C0f': 3.94, 'Sf': 1.48, 'rh0f': 8.96})
    main()
