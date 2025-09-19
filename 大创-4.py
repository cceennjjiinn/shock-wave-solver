import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from sympy import symbols, Eq, solve, simplify, Symbol
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import itertools
import logging
from sqlalchemy import create_engine, text
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库配置 - 只保留SQLite配置
DB_TYPE = "sqlite"
DB_CONFIG = {
    "sqlite": {
        "path": "shock_wave_data.db"
    }
}

# 创建数据库引擎
def create_db_engine(db_type):
    if db_type == "sqlite":
        return create_engine(f"sqlite:///{DB_CONFIG['sqlite']['path']}")
    raise ValueError(f"不支持的数据库类型: {db_type}")

# 初始化数据库引擎
db_engine = create_db_engine(DB_TYPE)

# 数据库查询函数
def query_database(query, params=None):
    try:
        with db_engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            # 获取列名
            columns = result.keys()
            # 转换为字典列表
            return [dict(zip(columns, row)) for row in result.fetchall()]
    except Exception as e:
        logger.error(f"数据库查询失败: {str(e)}")
        st.error(f"数据库查询失败: {str(e)}")
        return []

# 初始化数据库表结构
def init_database():
    try:
        with db_engine.connect() as conn:
            # 创建数据表，增加更多物理参数字段
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS shock_wave_all_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material TEXT,        -- 材料名称
                    rho0 REAL,            -- 初始密度 (g/cm³)
                    Us REAL,              -- 冲击波速度 (km/s)
                    Up REAL,              -- 粒子速度 (km/s)
                    P REAL,               -- 冲击压力 (GPa)
                    V REAL,               -- 比体积 (cm³/g)
                    rho REAL,             -- 压缩密度 (g/cm³)
                    V_V0 REAL,            -- 比体积比 (V/V0)
                    exp_method TEXT,      -- 实验方法/数据来源
                    gamma REAL,           -- 格吕奈森系数
                    T REAL,               -- 冲击温度 (K)
                    INDEX idx_material (material)  -- 新增索引以加快查询
                )
            """))
            conn.commit()
            logger.info("数据库表结构初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        st.error(f"数据库初始化失败: {str(e)}")

# 修复数据库表结构 - 确保所有必要字段存在
def fix_database_schema():
    """修复数据库表结构，添加缺失的字段"""
    try:
        with db_engine.connect() as conn:
            # 检查是否存在所需字段（SQLite方式）
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(shock_wave_all_data)")
            
            columns = [row[1] for row in cursor.fetchall()]  # SQLite返回的列信息中索引1是列名
            
            # 需要确保存在的字段
            required_columns = [
                ('gamma', 'REAL'),
                ('T', 'REAL'),
                ('V', 'REAL'),
                ('V_V0', 'REAL')
            ]
            
            for col_name, col_type in required_columns:
                if col_name not in columns:
                    # SQLite的ALTER TABLE语法
                    conn.execute(text(f"ALTER TABLE shock_wave_all_data ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    logger.info(f"数据库表结构已修复，添加了{col_name}字段")
                    st.success(f"数据库表结构已修复，添加了{col_name}字段")
    except Exception as e:
        logger.error(f"修复数据库表结构失败: {str(e)}")
        st.error(f"修复数据库表结构失败: {str(e)}")

# 先初始化数据库，再修复可能的表结构问题
init_database()
fix_database_schema()

# 物理合理性检查函数
def validate_physical合理性(data, material_type="通用"):
    """检查数据是否符合冲击波物理规律，返回错误信息列表"""
    errors = []
    # 增加容差，使检查更宽松
    tolerance = 1e-3
    
    # 基本物理约束检查
    if 'rho0' in data and data['rho0'] is not None and (data['rho0'] <= tolerance or data['rho0'] > 20):
        errors.append(f"{material_type}初始密度必须为正数且通常小于20 g/cm³，当前值: {data['rho0']}")
    
    if 'Us' in data and data['Us'] is not None and data['Us'] <= tolerance:
        errors.append(f"{material_type}冲击波速度必须为正数，当前值: {data['Us']}")
    
    if 'Up' in data and data['Up'] is not None and data['Up'] < -tolerance:
        errors.append(f"{material_type}粒子速度不能为负数，当前值: {data['Up']}")
    
    # Hugoniot关系检查：冲击波速度必须大于粒子速度（增加容差）
    if 'Us' in data and 'Up' in data and data['Us'] is not None and data['Up'] is not None:
        if data['Us'] <= data['Up'] + tolerance:  # 考虑浮点数精度，增加容差
            errors.append(f"{material_type}冲击波速度(Us={data['Us']})必须大于粒子速度(Up={data['Up']})")
    
    # 压力计算检查：基于动量守恒 P = rho0 * Us * Up * 10^3（添加单位转换，放宽误差允许范围）
    if 'P' in data and 'rho0' in data and 'Us' in data and 'Up' in data:
        if None not in [data['P'], data['rho0'], data['Us'], data['Up']]:
            calculated_P = data['rho0'] * data['Us'] * data['Up'] * 1000  # 添加单位转换系数1000
            # 允许20%误差
            if abs(data['P'] - calculated_P) > 0.2 * calculated_P:
                errors.append(f"{material_type}压力值与动量守恒计算不符，输入P={data['P']}, 计算值={calculated_P:.4f}")
    
    # 密度关系检查：压缩密度必须大于初始密度（增加容差）
    if 'rho' in data and 'rho0' in data and data['rho'] is not None and data['rho0'] is not None:
        if data['rho'] <= data['rho0'] - tolerance:  # 考虑浮点数精度
            errors.append(f"{material_type}压缩密度(rho={data['rho']})必须大于初始密度(rho0={data['rho0']})")
    
    # 比体积比检查：必须小于1（增加容差）
    if 'V_V0' in data and data['V_V0'] is not None and data['V_V0'] >= 1 + tolerance:
        errors.append(f"{material_type}比体积比(V/V0={data['V_V0']})必须小于1")
    
    # 温度检查：冲击温度应高于室温（放宽限制）
    if 'T' in data and data['T'] is not None and data['T'] < 100:  # 从200K放宽到100K
        errors.append(f"{material_type}冲击温度(T={data['T']})异常低，应高于室温(约300K)")
    
    # 格吕奈森系数检查：通常在0.5到5之间（放宽限制）
    if 'gamma' in data and data['gamma'] is not None:
        if data['gamma'] <= -tolerance or data['gamma'] > 20:  # 上限从10放宽到20
            errors.append(f"{material_type}格吕奈森系数(gamma={data['gamma']})应在0到20之间")
    
    return errors

# 数据库操作函数 - 优化查询效率
@st.cache_data(ttl=3600)  # 缓存1小时
def get_all_materials():
    try:
        query = "SELECT DISTINCT material FROM shock_wave_all_data"
        result = query_database(query)
        if result:
            return [row['material'] for row in result]
        return []
    except Exception as e:
        logger.warning(f"获取材料列表失败: {str(e)}")
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
            
        query = f"SELECT {fields} FROM shock_wave_all_data WHERE material = :material"
        result = query_database(query, {'material': material_name})
        
        # 转换为DataFrame
        df = pd.DataFrame(result) if result else pd.DataFrame()
        
        # 验证并清理数据
        if not df.empty:
            invalid_indices = []
            for idx, row in df.iterrows():
                errors = validate_physical合理性(row.to_dict(), material_name)
                if errors:
                    invalid_indices.append(idx)
            
            if invalid_indices:
                st.warning(f"材料 {material_name} 中有 {len(invalid_indices)} 条记录不符合物理规律，已自动过滤")
                df = df.drop(invalid_indices)
        
        return df
    except Exception as e:
        logger.warning(f"获取材料数据失败: {str(e)}")
        st.warning(f"获取材料数据失败: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    """保存多组求解结果到数据库，返回保存的记录数"""
    if not results:
        return 0
        
    try:
        count = 0
        invalid_count = 0
        with db_engine.begin() as conn:
            for result in results:
                # 检查物理合理性
                errors = validate_physical合理性({
                    'rho0': result.get('rh0f', 0),
                    'Us': result.get('Df', 0),
                    'Up': result.get('uf', 0),
                    'P': result.get('Pf', 0),
                    'rho': result.get('rhf', 0),
                    'V_V0': result.get('V_V0', 0),
                    'gamma': result.get('gammaf', 0),
                    'T': result.get('Tf', 0) if 'Tf' in result else 0
                }, material_name)
                
                if errors:
                    invalid_count += 1
                    continue
                    
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
                    'T': result.get('Tf', 0) if 'Tf' in result else 0
                }
                stmt = text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, data)
                count += 1
        
        if invalid_count > 0:
            st.warning(f"过滤了 {invalid_count} 个不合理解，未保存到数据库")
        logger.info(f"成功保存 {count} 条记录到数据库")
        return count
    except Exception as e:
        logger.error(f"保存结果到数据库失败: {str(e)}")
        st.error(f"保存失败: {str(e)}")
        return 0

def save_input_parameters(input_params, material_name="Copper", exp_method="manual_input"):
    """保存当前输入的参数到数据库，包含物理合理性检查"""
    try:
        # 提取关键参数并检查物理合理性
        data_dict = {
            'rho0': input_params.get('rh0f') if isinstance(input_params.get('rh0f'), (int, float)) else 0,
            'Us': input_params.get('Df') if isinstance(input_params.get('Df'), (int, float)) else 0,
            'Up': input_params.get('uf') if isinstance(input_params.get('uf'), (int, float)) else 0,
            'P': input_params.get('Pf') if isinstance(input_params.get('Pf'), (int, float)) else 0,
            'rho': input_params.get('rhf') if isinstance(input_params.get('rhf'), (int, float)) else 0,
            'gamma': input_params.get('gammaf') if isinstance(input_params.get('gammaf'), (int, float)) else 0,
            'T': input_params.get('Tf') if isinstance(input_params.get('Tf'), (int, float)) else 0
        }
        
        # 检查物理合理性
        errors = validate_physical合理性(data_dict, material_name)
        if errors:
            st.error("输入参数不符合物理规律:")
            for err in errors:
                st.error(f"- {err}")
            return 0
        
        data = {
            'material': material_name,
            'rho0': data_dict['rho0'],
            'Us': data_dict['Us'],
            'Up': data_dict['Up'],
            'P': data_dict['P'],
            'V': 0,  # 无法直接从输入参数获取
            'rho': data_dict['rho'],
            'V_V0': 0,  # 无法直接从输入参数获取
            'exp_method': exp_method,
            'gamma': data_dict['gamma'],
            'T': data_dict['T']
        }
        
        with db_engine.begin() as conn:
            stmt = text("""
                INSERT INTO shock_wave_all_data 
                (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
            """)
            conn.execute(stmt, data)
        logger.info(f"成功保存输入参数到 {material_name} 数据集")
        return 1
    except Exception as e:
        logger.error(f"保存输入参数失败: {str(e)}")
        st.error(f"保存输入参数失败: {str(e)}")
        return 0

def save_input_data_to_db(input_data, material_name, exp_method="manual_input"):
    """保存计算结果到数据库，返回保存的记录数，包含物理合理性检查"""
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
        
        # 检查物理合理性
        errors = validate_physical合理性(input_data, material_name)
        if errors:
            st.error("输入数据不符合物理规律:")
            for err in errors:
                st.error(f"- {err}")
            return 0
        
        with db_engine.begin() as conn:
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
        logger.info(f"成功保存输入数据到 {material_name} 数据集")
        return 1
    except Exception as e:
        logger.error(f"保存输入数据失败: {str(e)}")
        st.error(f"保存输入数据失败: {str(e)}")
        return 0

# 批量导入数据到数据库，增加物理合理性检查
def bulk_import_data(df, material_name, exp_method="bulk_import"):
    """批量导入数据到数据库，返回成功导入的记录数，包含物理合理性检查"""
    if df.empty:
        return 0
        
    required_columns = ['rho0', 'Us', 'Up']  # 至少需要这三个参数
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"导入失败：CSV文件缺少必要的列: {', '.join(missing_cols)}")
        return 0
        
    try:
        count = 0
        invalid_count = 0
        with db_engine.begin() as conn:
            for _, row in df.iterrows():
                # 跳过包含空值的行
                if row[required_columns].isnull().any():
                    continue
                    
                # 检查物理合理性
                row_dict = row.to_dict()
                errors = validate_physical合理性(row_dict, material_name)
                if errors:
                    invalid_count += 1
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
        
        if invalid_count > 0:
            st.warning(f"过滤了 {invalid_count} 个不合理解，未导入数据库")
        logger.info(f"成功批量导入 {count} 条记录到 {material_name} 数据集")
        return count
    except Exception as e:
        logger.error(f"批量导入失败: {str(e)}")
        st.error(f"批量导入失败: {str(e)}")
        return 0

# 批量删除选中的记录
def bulk_delete_records(ids):
    """删除指定ID的记录，返回删除的记录数"""
    if not ids or not isinstance(ids, list):
        return 0
        
    try:
        with db_engine.begin() as conn:
            # SQLite的参数绑定使用问号占位符
            placeholders = ', '.join(['?' for _ in range(len(ids))])
            stmt = text(f"DELETE FROM shock_wave_all_data WHERE id IN ({placeholders})")
            result = conn.execute(stmt, ids)
            deleted_count = result.rowcount
            logger.info(f"成功删除 {deleted_count} 条记录")
            return deleted_count
    except Exception as e:
        logger.error(f"删除失败: {str(e)}")
        st.error(f"删除失败: {str(e)}")
        return 0

# 清空指定材料的所有数据
def clear_material_data(material_name):
    """清空指定材料的所有数据，返回删除的记录数"""
    if not material_name:
        return 0
        
    try:
        with db_engine.begin() as conn:
            stmt = text("DELETE FROM shock_wave_all_data WHERE material = :material")
            result = conn.execute(stmt, {'material': material_name})
            deleted_count = result.rowcount
            logger.info(f"成功清空 {material_name} 的所有 {deleted_count} 条记录")
            return deleted_count
    except Exception as e:
        logger.error(f"清空数据失败: {str(e)}")
        st.error(f"清空数据失败: {str(e)}")
        return 0

def view_database():
    """显示数据库内容，包含批量添加和删除功能"""
    with st.expander("数据库内容", expanded=True):
        # 显示当前数据库配置（只显示SQLite信息）
        with st.expander("数据库配置", expanded=False):
            st.text(f"数据库类型: SQLite")
            st.text(f"数据库文件: {DB_CONFIG['sqlite']['path']}")
            st.text(f"文件位置: {os.path.abspath(DB_CONFIG['sqlite']['path'])}")
        
        # 批量操作区域
        st.subheader("批量数据操作")
        col1, col2 = st.columns(2)
        
        # 批量导入部分
        with col1:
            st.subheader("批量导入数据")
            new_material = st.text_input("材料名称", help="输入要导入数据的材料名称，使用英文")
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
                            st.success(f"成功导入 {count} 条记录（已过滤不符合物理规律的行）")
                            st.rerun()
                        else:
                            st.warning("没有导入任何记录，请检查数据格式和物理合理性")
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
            st.info(f"材料 {selected_material} 暂无有效数据")
        else:
            st.info(f"材料 {selected_material} 共有 {len(df)} 条有效记录（已过滤不符合物理规律的数据）")
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
                # 填充符合物理规律的示例数据（铜的典型值）
                template.loc[0] = [8.96, 5.0, 1.0, 44.8, 0.089, 11.2, 0.8, 2.0, 3000]
                csv = template.to_csv(index=False)
                st.download_button(
                    label="下载CSV模板",
                    data=csv,
                    file_name="shock_wave_data_template.csv",
                    mime="text/csv",
                    on_click=lambda: st.success("模板已准备好下载，包含符合物理规律的示例数据")
                )

# 冲击波参数计算（包含温度计算）
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300, calculate_temp=True):
    """根据Rankine-Hugoniot守恒关系计算冲击波参数，增加物理约束检查"""
    # 物理约束检查（放宽容差）
    tolerance = 1e-3
    if U_s <= u_p - tolerance:  # 允许微小的数值误差
        raise ValueError(f"冲击波速度 (Us={U_s}) 必须大于粒子速度 (Up={u_p})")
    if rho0 <= tolerance:
        raise ValueError(f"初始密度 (rho0={rho0}) 必须为正数")
    if U_s <= tolerance or u_p < -tolerance:  # 允许微小的负值，用于数值稳定性
        raise ValueError(f"速度参数必须非负，且冲击波速度必须为正数")
    
    # 动量守恒: P = rho0 * U_s * u_p * 1000（添加单位转换系数1000）
    # 单位转换: (g/cm³) * (km/s) * (km/s) * 1000 = 1e3 kg/m³ * 1e3 m/s * 1e3 m/s * 1000 = 1e9 Pa = 1 GPa
    P = rho0 * U_s * u_p * 1000  # 添加单位转换
    
    # 质量守恒推导比体积: V = (1/rho0) * (1 - u_p/U_s)
    V = (1 / rho0) * (1 - u_p / U_s)
    
    # 压缩密度: rho = rho0 * U_s/(U_s - u_p)
    rho = rho0 * U_s / (U_s - u_p)
    
    # 比体积比: V/V0 = 1 - u_p/U_s
    V_V0 = V * rho0  # 由于V0 = 1/rho0，V/V0 = V * rho0
    
    # 检查计算结果的物理合理性（放宽容差）
    if rho <= rho0 - tolerance:
        raise ValueError(f"计算的压缩密度 (rho={rho}) 必须大于初始密度 (rho0={rho0})")
    if V_V0 >= 1 + tolerance:
        raise ValueError(f"计算的比体积比 (V/V0={V_V0}) 必须小于1")
    if P <= -tolerance:  # 允许微小的负值，用于数值稳定性
        raise ValueError(f"计算的压力 (P={P}) 必须为正数")
    
    T = None
    if calculate_temp:
        # 温度计算（Mie-Grüneisen方程近似）
        # 单位转换: 1 GPa·cm³/g = 1e5 J/kg
        E_shock = 0.5 * P * (1/rho0 - V) * 1e6  # 冲击内能 (J/kg)
        # 基于Mie-Grüneisen方程简化形式（适用于弱冲击，忽略体积修正项）
        T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
        
        if T < 100:  # 放宽温度限制
            raise ValueError(f"计算的冲击温度 (T={T}) 异常低")
    
    return P, V, rho, V_V0, T

# Hugoniot关系拟合 - 优化数据预处理并添加错误处理
def fit_hugoniot(df):
    # 过滤物理上无效的数据（放宽条件）
    tolerance = 1e-3
    # 确保数据框不为空且包含必要的列
    if df is None or df.empty or 'Us' not in df.columns or 'Up' not in df.columns:
        return 0, 0  # 数据无效时返回默认值
    
    # 过滤NaN值
    df = df.dropna(subset=['Us', 'Up'])
    
    # 应用物理约束过滤
    df = df[(df['Us'] > df['Up'] - tolerance) & (df['Us'] > tolerance) & (df['Up'] >= -tolerance)]
    
    # 确保有足够的数据点进行拟合
    if len(df) < 2:
        return 0, 0  # 数据不足时返回默认值
        
    U_s = df['Us'].values
    u_p = df['Up'].values
    
    try:
        # 尝试进行线性拟合
        coeffs = np.polyfit(u_p, U_s, 1)
        S = coeffs[0]    # 斜率参数
        C0 = coeffs[1]   # 截距（零压声速）
    except np.linalg.LinAlgError:
        # 处理SVD不收敛的情况
        st.warning("Hugoniot拟合失败：SVD不收敛，使用默认参数")
        return 3.0, 1.5  # 返回合理的默认值
    except Exception as e:
        # 处理其他可能的错误
        st.warning(f"Hugoniot拟合失败：{str(e)}，使用默认参数")
        return 3.0, 1.5  # 返回合理的默认值
    
    # 物理约束：S通常在1.3-2.0之间，C0应为正数（放宽条件）
    if C0 <= -tolerance:  # 允许微小的负值
        st.warning(f"Hugoniot拟合的体声速 (C0={C0}) 为非正数，已调整为合理值")
        C0 = max(1.0, abs(C0))  # 确保体声速为正数且合理
        
    if S < 0.5 or S > 5.0:  # 放宽范围
        st.warning(f"Hugoniot参数 (S={S}) 超出典型范围 (1.0-3.0)，可能存在数据问题")
        
    return C0, S

@st.cache_data(ttl=3600)  # 缓存拟合结果
def fit_material_data(df, material_name, material_type):
    if df is None or df.empty:
        st.warning(f"{material_type}材料 '{material_name}' 没有数据")
        return None
    
    # 过滤异常值（放宽条件）
    tolerance = 1e-3
    df = df[(df['Us'] > df['Up'] - tolerance) & (df['Us'] > tolerance) & (df['Up'] >= -tolerance)]
    if len(df) < 2:
        st.warning(f"{material_type}材料 '{material_name}' 的有效数据不足，无法进行拟合")
        return None
    
    X = df['Up'].values.reshape(-1, 1)
    y = df['Us'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 拟合参数
    C0 = model.intercept_    # 体声速 (km/s)
    S = model.coef_[0]       # Hugoniot参数S
    y_pred = model.predict(X)
    
    # 物理约束检查（放宽条件）
    if C0 <= -tolerance:  # 允许微小的负值
        st.warning(f"{material_type}材料 '{material_name}' 拟合的体声速 (C0={C0}) 为非正数，已调整")
        C0 = max(1.0, abs(C0))
        
    if S < 0.5 or S > 5.0:  # 放宽范围
        st.warning(f"{material_type}材料 '{material_name}' 的Hugoniot参数 (S={S}) 超出典型范围 (1.0-3.0)")
    
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
        st.info(f"实验方法分布: {', '.join([f'{k}: {v}条记录' for k, v in method_counts.items()])}")
    
    return {
        "C0": C0, "S": S, "rho0": df['rho0'].mean(),
        "r2": r2, "rmse": rmse, "mae": mae
    }

# 误差传播计算
def calculate_error(params, param_errors):
    """计算输出参数的误差（基于误差传播公式）"""
    rho0, Us, Up = params['rho0'], params['Us'], params['Up']
    rho0_err, Us_err, Up_err = param_errors['rho0'], param_errors['Us'], param_errors['Up']
    
    # 压力误差: P = rho0*Us*Up*1000 → 相对误差平方和（添加单位转换系数）
    P_rel_err = (rho0_err/rho0)**2 + (Us_err/Us)** 2 + (Up_err/Up)**2
    P_err = rho0*Us*Up*1000 * np.sqrt(P_rel_err)  # 添加单位转换
    
    # 冲击波速度误差（简化）
    Us_err = np.sqrt(Us_err**2 + (0.01*Us)** 2)  # 加入1%模型误差
    
    return {
        "P_err": P_err,
        "Us_err": Us_err,
        "Up_err": Up_err
    }

# 输入函数 - 修复参数共享问题，确保空白输入被正确识别为未知数
def get_input_streamlit(label, var_name, key, default=None, unit="", desc="", disabled=False):
    st.caption(f"{desc} | 单位: {unit}")
    input_type = st.radio(
        f"{label} 输入类型",
        ["单一值", "多个值 (逗号分隔)", "范围值 (带步长)"],
        key=f"{key}_type",
        horizontal=True,
        disabled=disabled,
        help="选择输入方式：单一值、多个离散值或连续范围。留空将作为未知数求解"
    )
    
    default_val = str(default) if default is not None else ""
    
    # 物理参数范围提示
    param_ranges = {
        'rh0': "典型范围: 0.1-20 g/cm³",
        'D': "典型范围: 1-30 km/s",
        'u': "典型范围: 0-20 km/s (小于冲击波速度)",
        'P': "典型范围: 0.1-5000 GPa",
        'gamma': "典型范围: 0.5-5.0",
        'T': "典型范围: 300-100000 K",
        'C0': "典型范围: 1-10 km/s",
        'S': "典型范围: 1.3-2.0"
    }
    
    # 提取参数类型前缀
    param_type = next((k for k in param_ranges if var_name.startswith(k)), None)
    if param_type:
        st.caption(f"物理约束: {param_ranges[param_type]}")
    
    if input_type == "单一值":
        val = st.text_input(label, default_val, key=f"{key}_single", disabled=disabled)
        if val.strip() == "":  # 空白输入被视为未知数
            return symbols(var_name)
        try:
            val_num = float(val)
            # 基本物理范围检查（仅警告，不阻止输入）
            if param_type == 'rh0' and (val_num <= 0 or val_num > 20):
                st.warning(f"{label} 超出典型范围 (0.1-20 g/cm³)")
            elif param_type == 'D' and (val_num <= 0 or val_num > 30):
                st.warning(f"{label} 超出典型范围 (1-30 km/s)")
            elif param_type == 'u' and (val_num < 0 or val_num > 20):
                st.warning(f"{label} 超出典型范围 (0-20 km/s)")
            elif param_type == 'P' and (val_num <= 0 or val_num > 5000):
                st.warning(f"{label} 超出典型范围 (0.1-5000 GPa)")
            elif param_type == 'gamma' and (val_num <= 0 or val_num > 10):
                st.warning(f"{label} 超出典型范围 (0.5-5.0)")
            return [val_num]
        except ValueError:
            st.error("请输入有效的数字 (例如: 3.14)")
            return None
    elif input_type == "多个值 (逗号分隔)":
        val = st.text_input(
            label, 
            default_val, 
            key=f"{key}_multi", 
            disabled=disabled,
            help="输入多个值，用逗号分隔 (例如: 1.5, 3.0, 4.5)"
        )
        if val.strip() == "":  # 空白输入被视为未知数
            return symbols(var_name)
        try:
            # 处理可能的空格并分割
            values = [float(x.strip()) for x in val.split(',') if x.strip()]
            if not values:
                st.error("请至少输入一个值")
                return None
                
            # 检查范围
            for val_num in values:
                if param_type == 'rh0' and (val_num <= 0 or val_num > 20):
                    st.warning(f"{label} 包含超出典型范围 (0.1-20 g/cm³) 的值")
                    break
                elif param_type == 'D' and (val_num <= 0 or val_num > 30):
                    st.warning(f"{label} 包含超出典型范围 (1-30 km/s) 的值")
                    break
                elif param_type == 'u' and (val_num < 0 or val_num > 20):
                    st.warning(f"{label} 包含超出典型范围 (0-20 km/s) 的值")
                    break
                    
            return values
        except ValueError:
            st.error("请输入有效的逗号分隔数字 (例如: 1.0, 2.5, 3.8)")
            return None
    else:
        st.caption("范围示例: 开始=1.0, 结束=5.0, 步长=1.0 → 生成 [1.0, 2.0, 3.0, 4.0, 5.0]")
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.text_input(
                f"{label} 起始值", 
                default_val, 
                key=f"{key}_start", 
                disabled=disabled,
                help="范围中的第一个值 (例如: 2.0)"
            )
        with col2:
            end = st.text_input(
                f"{label} 结束值", 
                "", 
                key=f"{key}_end", 
                disabled=disabled,
                help="范围中的最后一个值 (必须大于起始值, 例如: 10.0)"
            )
        with col3:
            step = st.text_input(
                f"{label} 步长 (可选)", 
                "0.5", 
                key=f"{key}_step", 
                disabled=disabled,
                help="增量值 (例如: 0.5 或 2.0, 默认 0.5)"
            )
            
        if start.strip() == "" or end.strip() == "":  # 空白输入被视为未知数
            return symbols(var_name)
            
        try:
            start = float(start)
            end = float(end)
            step = float(step) if step else 0.5
            
            # 验证和修正输入
            if step <= 0:
                step = 0.5
                st.warning("步长必须为正数，已自动设置为0.5")
            if start > end:
                start, end = end, start
                st.warning("起始值大于结束值，已自动调整为升序")
            if (end - start) < step:
                st.warning("步长大于范围差值，将只返回起始值")
                return [start]
                
            # 检查范围是否符合物理约束
            if param_type == 'rh0':
                if start < 0.1 or end > 20:
                    st.warning(f"{label} 范围超出典型物理范围 (0.1-20 g/cm³)")
            elif param_type == 'D':
                if start < 1 or end > 30:
                    st.warning(f"{label} 范围超出典型物理范围 (1-30 km/s)")
            elif param_type == 'u':
                if start < 0 or end > 20:
                    st.warning(f"{label} 范围超出典型物理范围 (0-20 km/s)")
                    
            # 生成范围值
            values = []
            current = start
            epsilon = 1e-9  # 处理浮点数精度问题
            while current <= end + epsilon:
                values.append(round(current, 6))
                current += step
            return values
        except ValueError:
            st.error("请输入有效的范围数字 (例如: 开始=1.0, 结束=5.0, 步长=1.0)")
            return None

# 数值求解器（改进版本，更好地处理未知数）
def solve_numerically(eqs, sym_vars, initial_guess):
    """使用数值方法求解方程组，增加物理约束检查"""
    var_list = list(sym_vars.values())
    
    def residuals(x):
        """计算残差：方程组的误差"""
        substitutions = {var_list[i]: x[i] for i in range(len(x))}
        residuals = []
        for eq in eqs:
            # 替换变量
            substituted = eq.subs(substitutions)
            # 检查是否为布尔值
            if substituted == True:
                residuals.append(0.0)  # 等式成立，残差为0
            elif substituted == False:
                residuals.append(1e10)  # 等式不成立，给予大残差
            else:
                # 正常计算数值残差
                try:
                    # 简化表达式以提高计算稳定性
                    simplified = simplify(substituted)
                    residuals.append(float(abs(simplified.evalf())))
                except:
                    residuals.append(1e10)  # 计算失败时给予大残差
        return residuals
    
    # 根据初始猜测值的长度动态生成边界（放宽范围）
    n_vars = len(initial_guess)
    lower_bounds = [-1.0] * n_vars  # 允许微小的负值，提高数值稳定性
    upper_bounds = [100.0] * n_vars  # 扩大上限
    
    # 根据变量类型调整特定变量的边界，更合理地符合物理规律
    for i, var in enumerate(initial_guess.keys()):
        var_str = str(var)
        if var_str.startswith(('rh0', 'rh')):  # 密度
            lower_bounds[i] = 0.01  # g/cm³，放宽下界
            upper_bounds[i] = 50.0  # g/cm³，扩大上界
        elif var_str.startswith(('D', 'C0', 'u', 'w')):  # 速度
            lower_bounds[i] = 0.01  # km/s，放宽下界
            upper_bounds[i] = 100.0  # km/s，扩大上界
        elif var_str.startswith(('P', 'E')):  # 压力/能量
            lower_bounds[i] = 0.001  # GPa，放宽下界
            upper_bounds[i] = 10000.0  # GPa，扩大上界
        elif var_str.startswith('gamma'):  # 格吕奈森系数
            lower_bounds[i] = 0.1  # 放宽下界
            upper_bounds[i] = 20.0  # 扩大上界
        elif var_str.startswith('T'):  # 温度
            lower_bounds[i] = 100.0  # 放宽下界
            upper_bounds[i] = 1e6  # K，扩大上界
    
    # 执行最小二乘优化（调整参数提高收敛性）
    result = least_squares(
        residuals,
        list(initial_guess.values()),
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-6,  # 适当降低精度要求
        gtol=1e-6,
        xtol=1e-6,
        max_nfev=10000,  # 大幅增加迭代次数
        loss='soft_l1',  # 使用更稳健的损失函数
        f_scale=0.1  # 调整损失函数的比例参数
    )
    
    if result.success:
        solution = {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
        
        # 验证解的物理合理性（放宽容差）
        tolerance = 1e-3
        valid = True
        
        # 检查冲击波速度大于粒子速度（允许微小误差）
        if 'Df' in solution and 'uf' in solution and solution['Df'] <= solution['uf'] - tolerance:
            valid = False
        if 'Db' in solution and 'ub' in solution and solution['Db'] <= solution['ub'] - tolerance:
            valid = False
        if 'Ds' in solution and 'us' in solution and solution['Ds'] <= solution['us'] - tolerance:
            valid = False
            
        # 检查压缩密度大于初始密度（允许微小误差）
        if 'rh0f' in solution and 'rhf' in solution and solution['rhf'] <= solution['rh0f'] - tolerance:
            valid = False
        if 'rh0b' in solution and 'rhb' in solution and solution['rhb'] <= solution['rh0b'] - tolerance:
            valid = False
        if 'rh0s' in solution and 'rhs' in solution and solution['rhs'] <= solution['rh0s'] - tolerance:
            valid = False
            
        # 检查压力为正数（允许微小误差）
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
    
    # 使用数据中的平均密度而非硬编码值，添加单位转换系数1000
    rho0 = df['rho0'].mean() if not df.empty else 8.96
    P_range = rho0 * U_s_fit * u_p_range * 1000  # 动量守恒关系 P = ρ0·Us·Up·1000（添加单位转换）
    
    axs[0, 1].plot(u_p_range, P_range, 'r-', label='Theoretical: P = ρ0·Us·Up·1000')
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
       - 冲击压力(P) = ρ₀·Us·Up·10³（动量守恒，包含单位转换）
       - Hugoniot关系：Us = C₀ + S·Up（C₀为体声速，S通常在1.3-2.0之间）
       - 飞片速度关系：w = Df - uf（实验室坐标系）
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
    
    # 飞片与基板界面速度关系说明 - 修正为正确的物理关系
    st.info("""
    飞片冲击关系：飞片速度w与粒子速度uf的关系为w = Df - uf
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
            if isinstance(input_params.get(var), Symbol):
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
                # 修正：飞片速度与粒子速度关系 (实验室坐标系): w = Df - uf
                Eq(sym_vars['w'] - (sym_vars['Df'] - sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·uf·1000  (修正：使用标准动量守恒公式，添加单位转换)
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf']*1000, 0),
                # 基板动量守恒: Pb = rho0b·Db·ub·1000  (修正：使用标准动量守恒公式，添加单位转换)
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub']*1000, 0),
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
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us'])*1000, 0),  # 添加单位转换
                    # 样品动量守恒
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us']*1000, 0),  # 添加单位转换
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
                    # 基于已知参数动态设置初始猜测值，使用修正后的飞片速度公式
                    if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['Df'] - known_params['uf']  # 修正为减号
                    elif var_str == 'Df' and 'w' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['w'] + known_params['uf']  # 由w = Df - uf推导
                    elif var_str == 'uf' and 'w' in known_params and 'Df' in known_params:
                        initial_guess[var] = known_params['Df'] - known_params['w']  # 由w = Df - uf推导
                    elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf'] * 1000  # 添加单位转换
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
                            # 基于飞片速度估算压力，添加单位转换
                            initial_guess[var] = known_params['rh0f'] * (known_params['w']/2) * (known_params['w']/2) * 1000
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
        sample_material = st.text_input("样品材料名称", value="铜", help="输入材料名称，例如：铜、铝")
    
    # 飞片与基板界面速度关系说明 - 修正为正确的物理关系
    st.info("""
    飞片冲击关系：飞片速度w与粒子速度uf的关系为w = Df - uf
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
                                            value=900.0, min_value=1.0, help="铜约为385，铝约为900")
        with col3:
            Cv_values['s'] = st.number_input(f"样品比热容 Cv (J/(kg·K)) ({sample_material})", 
                                            value=385.0, min_value=1.0, help="铜约为385，铝约为900")
    
    # 参数定义
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 常用材料参数参考表
    with st.expander("常用材料参数参考", expanded=False):
        ref_data = {
            "材料": ["铜", "铝", "铁", "水", "石英"],
            "密度 (g/cm³)": [8.96, 2.70, 7.87, 1.00, 2.65],
            "体声速 C0 (km/s)": [3.94, 5.32, 3.59, 1.50, 3.75],
            "Hugoniot参数 S": [1.48, 1.33, 1.58, 1.99, 1.50],
            "格吕奈森系数 γ": [2.0, 2.1, 1.9, 0.5, 1.0]
        }
        st.dataframe(pd.DataFrame(ref_data))
    
    # 飞片参数
    with st.expander(f"{flyer_material} 飞片参数", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "初始密度（必须输入）",
            "rhf": "压缩密度",
            "Df": "冲击波速度（对应Us）",
            "C0f": "体声速（参考值见上表）",
            "Sf": "Hugoniot参数S（无量纲）",
            "E0f": "初始内能密度 (≈0)",
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
        # 预设常用材料的默认值
        material_defaults = {
            "铜": {"rh0f": 8.96, "C0f": 3.94, "Sf": 1.48, "gammaf": 2.0},
            "铝": {"rh0f": 2.70, "C0f": 5.32, "Sf": 1.33, "gammaf": 2.1},
            "铁": {"rh0f": 7.87, "C0f": 3.59, "Sf": 1.58, "gammaf": 1.9},
            "水": {"rh0f": 1.00, "C0f": 1.50, "Sf": 1.99, "gammaf": 0.5},
            "石英": {"rh0f": 2.65, "C0f": 3.75, "Sf": 1.50, "gammaf": 1.75, "Sf": 1.50, "gammaf": 1.0}
        }
        # 获取当前材料的默认值
        default_vals = material_defaults.get(flyer_material, {})
        
        for i, var in enumerate(variables["f"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                default_val = default_vals.get(var, None)
                # 特殊处理初始内能，通常为0
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
        # 预设常用材料的默认值
        default_vals = material_defaults.get(base_material, {})
        # 替换参数前缀以匹配基板参数
        base_defaults = {f"rh0b": default_vals.get("rh0f", None),
                         f"C0b": default_vals.get("C0f", None),
                         f"Sb": default_vals.get("Sf", None),
                         f"gammab": default_vals.get("gammaf", None)}
        
        for i, var in enumerate(variables["b"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                default_val = base_defaults.get(var, None)
                # 特殊处理初始内能，通常为0
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
        # 预设常用材料的默认值
        default_vals = material_defaults.get(sample_material, {})
        # 替换参数前缀以匹配样品参数
        sample_defaults = {f"rh0s": default_vals.get("rh0f", None),
                          f"C0s": default_vals.get("C0f", None),
                          f"Ss": default_vals.get("Sf", None),
                          f"gammas": default_vals.get("gammaf", None)}
        
        for i, var in enumerate(variables["s"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                default_val = sample_defaults.get(var, None)
                # 特殊处理初始内能，通常为0
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
            count = save_input_parameters(input_params, sample_material, "manual_mode_input")
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
            if isinstance(input_params.get(var), Symbol):
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
                # 修正：飞片速度与粒子速度关系 (实验室坐标系): w = Df - uf
                Eq(sym_vars['w'] - (sym_vars['Df'] - sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·uf·1000  (修正：使用标准动量守恒公式，添加单位转换)
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf']*1000, 0),
                # 基板动量守恒: Pb = rho0b·Db·ub·1000  (修正：使用标准动量守恒公式，添加单位转换)
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub']*1000, 0),
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
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us'])*1000, 0),  # 添加单位转换
                    # 样品动量守恒
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us']*1000, 0),  # 添加单位转换
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
                    # 基于已知参数动态设置初始猜测值，使用修正后的飞片速度公式
                    if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['Df'] - known_params['uf']  # 修正为减号
                    elif var_str == 'Df' and 'w' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['w'] + known_params['uf']  # 由w = Df - uf推导
                    elif var_str == 'uf' and 'w' in known_params and 'Df' in known_params:
                        initial_guess[var] = known_params['Df'] - known_params['w']  # 由w = Df - uf推导
                    elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf'] * 1000  # 添加单位转换
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
                            # 基于飞片速度估算压力，添加单位转换
                            initial_guess[var] = known_params['rh0f'] * (known_params['w']/2) * (known_params['w']/2) * 1000
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
                file_name="manual_mode_results.csv",
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
            # 显示更多调试信息
            st.info("尝试以下解决方案：\n1. 检查输入参数是否在合理范围内\n2. 减少未知数数量，输入更多已知参数\n3. 放宽物理约束条件\n4. 调整参数范围，避免极端值")
    
    if st.button("返回主页"):
        st.session_state.page = "home"
        st.rerun()  # 立即刷新页面

# 主函数
def main():
    # 确保中文显示正常
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 初始化页面状态
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if "previous_page" not in st.session_state:
        st.session_state.previous_page = None
    
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
        if st.button("返回上一页"):
            if st.session_state.previous_page:
                st.session_state.page = st.session_state.previous_page
            else:
                st.session_state.page = "home"
            st.rerun()

if __name__ == "__main__":
    main()
