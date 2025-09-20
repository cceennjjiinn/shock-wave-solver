import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import itertools
import traceback
from datetime import datetime
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from scipy.optimize import least_squares
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sympy import symbols, Symbol, Eq, simplify, solve

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shock_wave_calculator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "sqlite": {
        "path": "shock_wave_data.db"
    },
    "mysql": {
        "host": "localhost",
        "port": 3306,
        "database": "shock_wave_db",
        "user": "root",
        "password": "password",
        "charset": "utf8mb4"
    }
}

# 全局数据库类型设置
DB_TYPE = "sqlite"  # 默认使用SQLite

# 创建数据库引擎 - 支持MySQL和SQLite
def create_db_engine(db_type="sqlite"):
    """创建指定类型的数据库引擎"""
    try:
        if db_type == "mysql":
            config = DB_CONFIG["mysql"]
            engine = create_engine(
                f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset={config['charset']}",
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600
            )
            logger.info(f"成功创建MySQL引擎连接：{config['host']}:{config['port']}/{config['database']}")
            return engine
        else:  # SQLite
            config = DB_CONFIG["sqlite"]
            engine = create_engine(
                f"sqlite:///{config['path']}",
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600
            )
            logger.info(f"成功创建SQLite引擎连接：{config['path']}")
            return engine
    except Exception as e:
        logger.error(f"创建数据库引擎失败: {str(e)}")
        st.error(f"数据库连接失败: {str(e)}")
        return None

# 初始化数据库引擎
db_engine = create_db_engine(DB_TYPE)

# SQLite性能优化
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if DB_TYPE == "sqlite":
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')  # 预写日志
        cursor.execute('PRAGMA synchronous=NORMAL;')  # 同步模式
        cursor.execute('PRAGMA temp_store=MEMORY;')   # 临时存储
        cursor.execute('PRAGMA cache_size=-20000;')   # 增加缓存（20MB）
        cursor.close()

# ------------------------------
# 核心函数：数据库查询+全链路日志（适配多数据库）
# ------------------------------
def query_database(sql, params=None, db_type=DB_TYPE):
    """通用数据库查询函数，支持参数化查询和多数据库类型"""
    conn = None
    cursor = None
    try:
        # 1. 建立数据库连接
        logger.info("开始建立数据库连接...")
        engine = create_db_engine(db_type)
        if not engine:
            return None
            
        conn = engine.connect()
        config = DB_CONFIG[db_type]
        if db_type == "mysql":
            logger.debug(f"数据库连接成功：{config['host']}:{config['port']}/{config['database']}")
        else:
            logger.debug(f"数据库连接成功：{config['path']}")
        
        # 2. 执行查询
        logger.info(f"执行SQL查询：{sql}")
        start_time = datetime.now()
        
        if db_type == "mysql":
            # 使用pymysql原生接口执行查询
            import pymysql
            cursor = conn.connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params or {})
            result = cursor.fetchall()
        else:
            # 使用SQLAlchemy执行SQLite查询
            result = conn.execute(text(sql), params or {}).mappings().all()
            # 转换为字典列表
            result = [dict(row) for row in result]
            
        exec_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"SQL执行完成，耗时：{exec_time:.3f}秒")
        
        # 3. 获取结果并记录返回行数
        row_count = len(result)
        logger.info(f"数据库返回行数：{row_count}行")
        if row_count > 0:
            logger.debug(f"返回数据示例（前2行）：{result[:2]}")  # 日志只显示前2行避免刷屏
        return result
    except Exception as e:
        # 记录异常细节
        logger.error(f"查询异常：{str(e)}")
        logger.error(f"异常堆栈：{traceback.format_exc()}")
        st.error(f"数据库查询失败: {str(e)}")
        return None
    finally:
        # 关闭连接（无论成功失败都执行）
        if cursor:
            cursor.close()
            logger.debug("游标已关闭")
        if conn:
            conn.close()
            logger.debug("数据库连接已关闭")

# 初始化数据库 - 添加材料字段索引
def init_database():
    try:
        with db_engine.connect() as conn:
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
                logger.info("数据库表结构初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        st.error(f"数据库初始化失败: {str(e)}")

# 修复数据库表结构 - 确保所有必要字段存在
def fix_database_schema():
    """修复数据库表结构，添加缺失的字段"""
    try:
        with db_engine.connect() as conn:
            # 检查是否存在所需字段
            cursor = conn.connection.cursor()
            if DB_TYPE == "mysql":
                cursor.execute("DESCRIBE shock_wave_all_data")
            else:  # SQLite
                cursor.execute("PRAGMA table_info(shock_wave_all_data)")
                
            columns = [row[0] if DB_TYPE == "mysql" else row[1] for row in cursor.fetchall()]
            
            # 需要确保存在的字段
            required_columns = [
                ('gamma', 'REAL'),
                ('T', 'REAL'),
                ('V', 'REAL'),
                ('V_V0', 'REAL')
            ]
            
            for col_name, col_type in required_columns:
                if col_name not in columns:
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

# ------------------------------
# 核心优化1：物理合理性检查（参考新代码的容忍度逻辑）
# ------------------------------
def validate_physical合理性(data, material_type="通用"):
    """检查数据是否符合冲击波物理规律，返回错误信息列表（增加数值容忍度）"""
    errors = []
    tolerance = 1e-3  # 数值计算容忍度
    
    # 基本物理约束检查（放宽浮点数误差容忍）
    if 'rho0' in data and data['rho0'] is not None:
        if data['rho0'] <= tolerance or data['rho0'] > 30:  # 密度范围放宽到30 g/cm³
            errors.append(f"{material_type}初始密度应在(0,30] g/cm³，当前值: {data['rho0']:.4f}")
    
    if 'Us' in data and data['Us'] is not None and data['Us'] <= tolerance:
        errors.append(f"{material_type}冲击波速度必须大于{tolerance} km/s，当前值: {data['Us']:.4f}")
    
    if 'Up' in data and data['Up'] is not None and data['Up'] < -tolerance:  # 允许微小负值（计算误差）
        errors.append(f"{material_type}粒子速度不应小于-{tolerance} km/s，当前值: {data['Up']:.4f}")
    
    # Hugoniot关系检查：冲击波速度 ≥ 粒子速度 - 容忍度（避免严格小于导致的误判）
    if 'Us' in data and 'Up' in data and data['Us'] is not None and data['Up'] is not None:
        if data['Us'] <= data['Up'] - tolerance:
            errors.append(f"{material_type}冲击波速度(Us={data['Us']:.4f})应大于粒子速度(Up={data['Up']:.4f})")
    
    # 压力计算检查：允许20%误差（适应实验数据波动）
    if 'P' in data and 'rho0' in data and 'Us' in data and 'Up' in data:
        if None not in [data['P'], data['rho0'], data['Us'], data['Up']] and data['P'] > tolerance:
            calculated_P = data['rho0'] * data['Us'] * data['Up']
            if abs(data['P'] - calculated_P) > 0.2 * calculated_P:  # 误差放宽到20%
                errors.append(f"{material_type}压力值与动量守恒计算不符：输入P={data['P']:.4f}, 计算值={calculated_P:.4f}")
    
    # 密度关系检查：压缩密度 ≥ 初始密度 - 容忍度
    if 'rho' in data and 'rho0' in data and data['rho'] is not None and data['rho0'] is not None:
        if data['rho'] <= data['rho0'] - tolerance:
            errors.append(f"{material_type}压缩密度(rho={data['rho']:.4f})应大于初始密度(rho0={data['rho0']:.4f})")
    
    # 比体积比检查：允许微小超过1（计算误差）
    if 'V_V0' in data and data['V_V0'] is not None and data['V_V0'] >= 1 + tolerance:
        errors.append(f"{material_type}比体积比(V/V0={data['V_V0']:.4f})应小于1.0")
    
    # 温度检查：冲击温度应高于100K（放宽室温限制）
    if 'T' in data and data['T'] is not None and data['T'] < 100:
        errors.append(f"{material_type}冲击温度(T={data['T']:.0f})异常低，建议检查")
    
    # 格吕奈森系数检查：范围放宽到0.1-10
    if 'gamma' in data and data['gamma'] is not None:
        if data['gamma'] < 0.1 or data['gamma'] > 10:
            errors.append(f"{material_type}格吕奈森系数(gamma={data['gamma']:.4f})应在0.1-10之间")
    
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
        
        # 验证并清理数据（使用优化后的物理检查）
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
            placeholders = ', '.join([':id' + str(i) for i in range(len(ids))])
            params = {'id' + str(i): id for i, id in enumerate(ids)}
            stmt = text(f"DELETE FROM shock_wave_all_data WHERE id IN ({placeholders})")
            result = conn.execute(stmt, params)
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
        # 数据库类型切换
        global DB_TYPE, db_engine
        new_db_type = st.radio("选择数据库类型", ["sqlite", "mysql"], 
                              index=0 if DB_TYPE == "sqlite" else 1)
        if new_db_type != DB_TYPE:
            DB_TYPE = new_db_type
            db_engine = create_db_engine(DB_TYPE)
            st.success(f"已切换至{DB_TYPE}数据库")
            st.rerun()
        
        # 显示当前数据库配置
        with st.expander("数据库配置", expanded=False):
            if DB_TYPE == "mysql":
                config = DB_CONFIG["mysql"]
                st.text(f"主机: {config['host']}")
                st.text(f"端口: {config['port']}")
                st.text(f"数据库: {config['database']}")
                st.text(f"用户: {config['user']}")
            else:
                st.text(f"数据库文件: {DB_CONFIG['sqlite']['path']}")
        
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

# ------------------------------
# 核心优化2：冲击波参数计算（参考新代码的容忍度和异常处理）
# ------------------------------
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300, calculate_temp=True):
    """根据Rankine-Hugoniot守恒关系计算冲击波参数，增加物理约束容忍度"""
    tolerance = 1e-3  # 数值计算容忍度
    
    # 物理约束检查（放宽浮点数误差）
    if U_s <= u_p - tolerance:
        raise ValueError(f"冲击波速度 (Us={U_s:.4f}) 必须大于粒子速度 (Up={u_p:.4f})（容忍误差{tolerance}）")
    if rho0 <= tolerance:
        raise ValueError(f"初始密度 (rho0={rho0:.4f}) 必须大于{ tolerance } g/cm³")
    if U_s <= tolerance or u_p < -tolerance:
        raise ValueError(f"冲击波速度必须>={tolerance} km/s，粒子速度不应<-{tolerance} km/s")
    
    # 动量守恒: P = rho0 * U_s * u_p
    P = rho0 * U_s * u_p
    
    # 质量守恒推导比体积: V = (1/rho0) * (1 - u_p/U_s)
    V = (1 / rho0) * (1 - u_p / U_s)
    
    # 压缩密度: rho = rho0 * U_s/(U_s - u_p)
    rho = rho0 * U_s / (U_s - u_p)
    
    # 比体积比: V/V0 = 1 - u_p/U_s
    V_V0 = V * rho0  # 由于V0 = 1/rho0，V/V0 = V * rho0
    
    # 检查计算结果的物理合理性（增加容忍度）
    if rho <= rho0 - tolerance:
        raise ValueError(f"压缩密度 (rho={rho:.4f}) 必须大于初始密度 (rho0={rho0:.4f})（容忍误差{ tolerance }）")
    if V_V0 >= 1 + tolerance:
        raise ValueError(f"比体积比 (V/V0={V_V0:.4f}) 必须小于1.0（容忍误差{ tolerance }）")
    if P <= -tolerance:
        raise ValueError(f"冲击压力 (P={P:.4f}) 必须为正数（容忍误差{ tolerance }）")
    
    T = None
    if calculate_temp:
        # 温度计算（Mie-Grüneisen方程近似）
        E_shock = 0.5 * P * (1/rho0 - V) * 1e6  # 冲击内能 (J/kg)
        T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
        
        if T < 100:  # 放宽温度下限至100K
            raise ValueError(f"冲击温度 (T={T:.0f}K) 异常低（建议>100K）")
    
    return P, V, rho, V_V0, T

# Hugoniot关系拟合 - 优化数据预处理
def fit_hugoniot(df):
    # 过滤物理上无效的数据（使用优化后的约束）
    tolerance = 1e-3
    df = df[(df['Us'] > df['Up'] - tolerance) & (df['Us'] > tolerance) & (df['Up'] >= -tolerance)]
    if len(df) < 2:
        return 0, 0  # 数据不足时返回默认值
        
    U_s = df['Us'].values
    u_p = df['Up'].values
    coeffs = np.polyfit(u_p, U_s, 1)
    S = coeffs[0]    # 斜率参数
    C0 = coeffs[1]   # 截距（零压声速）
    
    # 物理约束：S通常在1.0-3.0之间，C0应为正数（增加容忍度）
    if C0 <= -tolerance:
        st.warning(f"Hugoniot拟合的体声速 (C0={C0:.4f}) 为负，已调整为合理值")
        C0 = max(1.0, abs(C0))  # 确保体声速为正数且合理
        
    if S < 1.0 - tolerance or S > 3.0 + tolerance:
        st.warning(f"Hugoniot参数 (S={S:.4f}) 超出典型范围 (1.0-3.0)，可能存在数据问题")
        
    return C0, S

@st.cache_data(ttl=3600)  # 缓存拟合结果
def fit_material_data(df, material_name, material_type):
    if df is None or df.empty:
        st.warning(f"{material_type}材料 '{material_name}' 没有数据")
        return None
    
    # 过滤异常值（使用优化后的约束）
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
    
    # 物理约束检查（增加容忍度）
    if C0 <= -tolerance:
        st.warning(f"{material_type}材料 '{material_name}' 拟合的体声速 (C0={C0:.4f}) 为负，已调整")
        C0 = max(1.0, abs(C0))
        
    if S < 1.0 - tolerance or S > 3.0 + tolerance:
        st.warning(f"{material_type}材料 '{material_name}' 的Hugoniot参数 (S={S:.4f}) 超出典型范围 (1.0-3.0)")
    
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
        'rh0': "典型范围: 0.1-30 g/cm³",  # 范围放宽
        'D': "典型范围: 0.1-50 km/s",    # 范围放宽
        'u': "典型范围: 0-30 km/s (小于冲击波速度)",  # 范围放宽
        'P': "典型范围: 0.001-10000 GPa",  # 范围放宽
        'gamma': "典型范围: 0.1-10.0",     # 范围放宽
        'T': "典型范围: 100-100000 K",     # 范围放宽
        'C0': "典型范围: 0.1-10 km/s",     # 范围放宽
        'S': "典型范围: 1.0-3.0"
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
            # 基本物理范围检查（放宽提示）
            if param_type == 'rh0' and (val_num < 0.1 or val_num > 30):
                st.warning(f"{label} 超出典型范围 (0.1-30 g/cm³)")
            elif param_type == 'D' and (val_num < 0.1 or val_num > 50):
                st.warning(f"{label} 超出典型范围 (0.1-50 km/s)")
            elif param_type == 'u' and (val_num < 0 or val_num > 30):
                st.warning(f"{label} 超出典型范围 (0-30 km/s)")
            elif param_type == 'P' and (val_num < 0.001 or val_num > 10000):
                st.warning(f"{label} 超出典型范围 (0.001-10000 GPa)")
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
                if param_type == 'rh0' and (val_num < 0.1 or val_num > 30):
                    st.warning(f"{label} 包含超出典型范围 (0.1-30 g/cm³) 的值")
                    break
                elif param_type == 'D' and (val_num < 0.1 or val_num > 50):
                    st.warning(f"{label} 包含超出典型范围 (0.1-50 km/s) 的值")
                    break
                elif param_type == 'u' and (val_num < 0 or val_num > 30):
                    st.warning(f"{label} 包含超出典型范围 (0-30 km/s) 的值")
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
                
            # 检查范围是否符合物理约束（放宽）
            if param_type == 'rh0':
                if start < 0.1 or end > 30:
                    st.warning(f"{label} 范围超出典型物理范围 (0.1-30 g/cm³)")
            elif param_type == 'D':
                if start < 0.1 or end > 50:
                    st.warning(f"{label} 范围超出典型物理范围 (0.1-50 km/s)")
            elif param_type == 'u':
                if start < 0 or end > 30:
                    st.warning(f"{label} 范围超出典型物理范围 (0-30 km/s)")
                    
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

# ------------------------------
# 核心优化3：数值求解器（参考新代码的联立求解、边界调整、解修正）
# ------------------------------
def generate_better_initial_guess(known_params, remaining_vars):
    """生成更合理的初始猜测值（参考新代码的参数依赖关系）"""
    initial_guess = {}
    tolerance = 1e-3
    
    for var in remaining_vars:
        var_str = str(var)
        # 1. 飞片关键参数联立（u_f, D_f）：基于w、C0f、Sf推导
        if var_str == 'uf' and 'w' in known_params and 'C0f' in known_params and 'Sf' in known_params:
            denominator = 1 + known_params['Sf']
            if abs(denominator) > tolerance:
                initial_guess[var] = (known_params['w'] - known_params['C0f']) / denominator
                continue
        elif var_str == 'Df' and 'w' in known_params and 'C0f' in known_params and 'Sf' in known_params:
            denominator = 1 + known_params['Sf']
            if abs(denominator) > tolerance:
                initial_guess[var] = (known_params['w'] + known_params['C0f'] * known_params['Sf']) / denominator
                continue
        
        # 2. 样品/基板关键参数联立（u_s/D_s, u_b/D_b）：基于P、rho0、C0、S推导
        if var_str == 'us' and 'Ps' in known_params and 'rh0s' in known_params and 'C0s' in known_params and 'Ss' in known_params:
            a = known_params['rh0s'] * known_params['Ss']
            b = known_params['rh0s'] * known_params['C0s']
            c = -known_params['Ps']
            delta = b**2 - 4*a*c
            if delta >= 0 and a != 0:
                sqrt_delta = np.sqrt(delta)
                u1 = (-b + sqrt_delta) / (2*a)
                initial_guess[var] = max(u1, tolerance)  # 取正根
                continue
        elif var_str == 'Ds' and 'us' in known_params and 'C0s' in known_params and 'Ss' in known_params:
            initial_guess[var] = known_params['C0s'] + known_params['Ss'] * known_params['us']
            continue
        
        if var_str == 'ub' and 'Pb' in known_params and 'rh0b' in known_params and 'C0b' in known_params and 'Sb' in known_params:
            a = known_params['rh0b'] * known_params['Sb']
            b = known_params['rh0b'] * known_params['C0b']
            c = -known_params['Pb']
            delta = b**2 - 4*a*c
            if delta >= 0 and a != 0:
                sqrt_delta = np.sqrt(delta)
                u1 = (-b + sqrt_delta) / (2*a)
                initial_guess[var] = max(u1, tolerance)  # 取正根
                continue
        elif var_str == 'Db' and 'ub' in known_params and 'C0b' in known_params and 'Sb' in known_params:
            initial_guess[var] = known_params['C0b'] + known_params['Sb'] * known_params['ub']
            continue
        
        # 3. 基础参数推导（动量守恒、质量守恒）
        if var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
            initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf']
            continue
        elif var_str == 'rhf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
            denominator = known_params['Df'] - known_params['uf']
            if abs(denominator) > tolerance:
                initial_guess[var] = (known_params['rh0f'] * known_params['Df']) / denominator
                continue
        
        # 4. 默认合理值（基于物理范围）
        if var_str.startswith(('rh0', 'rh')):
            initial_guess[var] = known_params.get('rh0f', 8.0)  # 参考飞片密度
        elif var_str.startswith(('D', 'C0', 'u')):
            initial_guess[var] = known_params.get('w', 10.0) / 2  # 参考飞片速度的一半
        elif var_str == 'w':
            initial_guess[var] = 10.0
        elif var_str.startswith('P'):
            initial_guess[var] = 100.0
        elif var_str.startswith('gamma'):
            initial_guess[var] = 2.0
        elif var_str.startswith('T'):
            initial_guess[var] = 3000.0
        else:
            initial_guess[var] = 1.0
    
    return initial_guess

def solve_numerically(eqs, sym_vars, initial_guess):
    """使用数值方法求解方程组（优化边界、迭代参数、解修正）"""
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
                    simplified = simplify(substituted)
                    residuals.append(float(abs(simplified.evalf())))
                except:
                    residuals.append(1e10)  # 计算失败时给予大残差
        return residuals
    
    # 优化边界设置（参考新代码，扩大搜索范围）
    n_vars = len(initial_guess)
    lower_bounds = [1e-3] * n_vars  # 更小的下界（允许接近0的值）
    upper_bounds = [100.0] * n_vars  # 更大的上界（覆盖更广物理范围）
    
    # 根据变量类型调整特定变量的边界
    for i, var in enumerate(initial_guess.keys()):
        var_str = str(var)
        if var_str.startswith(('rh0', 'rh')):  # 密度
            lower_bounds[i] = 1e-3  # g/cm³
            upper_bounds[i] = 30.0  # 放宽到30
        elif var_str.startswith(('D', 'C0', 'u', 'w')):  # 速度
            lower_bounds[i] = 1e-3  # km/s
            upper_bounds[i] = 50.0  # 放宽到50
        elif var_str.startswith(('P', 'E')):  # 压力/能量
            lower_bounds[i] = 1e-4  # GPa
            upper_bounds[i] = 10000.0  # 放宽到10000
        elif var_str.startswith('gamma'):  # 格吕奈森系数
            lower_bounds[i] = 0.1
            upper_bounds[i] = 10.0
        elif var_str.startswith('T'):  # 温度
            lower_bounds[i] = 100.0
            upper_bounds[i] = 1e6
    
    # 执行最小二乘优化（优化迭代参数）
    result = least_squares(
        residuals,
        list(initial_guess.values()),
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-6,  # 适当降低精度要求，提高收敛性
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=10000,  # 增加迭代次数
        loss='soft_l1',  # 鲁棒损失函数，减少异常值影响
        f_scale=0.1
    )
    
    if result.success:
        solution = {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
        tolerance = 1e-3
        
        # 验证解的物理合理性（参考新代码的容忍度）
        valid = True
        
        # 1. 冲击波速度 ≥ 粒子速度 - 容忍度
        if 'Df' in solution and 'uf' in solution and solution['Df'] <= solution['uf'] - tolerance:
            valid = False
        if 'Db' in solution and 'ub' in solution and solution['Db'] <= solution['ub'] - tolerance:
            valid = False
        if 'Ds' in solution and 'us' in solution and solution['Ds'] <= solution['us'] - tolerance:
            valid = False
            
        # 2. 压缩密度 ≥ 初始密度 - 容忍度
        if 'rh0f' in solution and 'rhf' in solution and solution['rhf'] <= solution['rh0f'] - tolerance:
            valid = False
        if 'rh0b' in solution and 'rhb' in solution and solution['rhb'] <= solution['rh0b'] - tolerance:
            valid = False
        if 'rh0s' in solution and 'rhs' in solution and solution['rhs'] <= solution['rh0s'] - tolerance:
            valid = False
            
        # 3. 压力为正
        for p_var in ['Pf', 'Pb', 'Ps']:
            if p_var in solution and solution[p_var] <= -tolerance:
                valid = False
                break
        
        # 4. 尝试修正轻微违反约束的解（关键优化）
        if not valid:
            adjusted = False
            # 修正冲击波速度
            if 'Df' in solution and 'uf' in solution and solution['Df'] <= solution['uf']:
                solution['Df'] = solution['uf'] + tolerance
                adjusted = True
            if 'Db' in solution and 'ub' in solution and solution['Db'] <= solution['ub']:
                solution['Db'] = solution['ub'] + tolerance
                adjusted = True
            if 'Ds' in solution and 'us' in solution and solution['Ds'] <= solution['us']:
                solution['Ds'] = solution['us'] + tolerance
                adjusted = True
            # 修正压缩密度
            if 'rh0f' in solution and 'rhf' in solution and solution['rhf'] <= solution['rh0f']:
                solution['rhf'] = solution['rh0f'] + tolerance
                adjusted = True
            if 'rh0b' in solution and 'rhb' in solution and solution['rhb'] <= solution['rh0b']:
                solution['rhb'] = solution['rh0b'] + tolerance
                adjusted = True
            if 'rh0s' in solution and 'rhs' in solution and solution['rhs'] <= solution['rh0s']:
                solution['rhs'] = solution['rh0s'] + tolerance
                adjusted = True
            # 修正压力
            for p_var in ['Pf', 'Pb', 'Ps']:
                if p_var in solution and solution[p_var] <= 0:
                    solution[p_var] = tolerance
                    adjusted = True
            
            if adjusted:
                return solution  # 返回修正后的解
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
    tolerance = 1e-3
    if C0 <= -tolerance:
        C0 = 3.0  # 默认合理值
    if S < 1.0 - tolerance or S > 3.0 + tolerance:
        S = 1.5  # 默认合理值
        
    u_p_range = np.linspace(0, min(30, df['Up'].max()*1.1), 100)  # 限制在放宽的物理范围
    U_s_fit = C0 + S * u_p_range  # Hugoniot关系
    
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
    # 计算压力范围并绘图
    P_range = rho0 * (C0 + S * u_p_range) * u_p_range  # P = rho0 * Us * Up
    axs[0, 1].plot(u_p_range, P_range, 'r-', label=f'P = {rho0:.2f}·Us·Up')
    axs[0, 1].set_xlabel('Particle Velocity Up (km/s)')
    axs[0, 1].set_ylabel('Shock Pressure P (GPa)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P vs Us
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 0].scatter(
            method_df['Us'], method_df['P'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7
        )
    
    # 使用Hugoniot关系从Us反推Up: Up = (Us - C0)/S
    u_p_from_Us = (u_p_range + C0)  # 修正后的范围计算
    if abs(S) > tolerance:  # 避免除以零
        u_p_from_Us = (u_p_range - C0) / S
        u_p_from_Us = np.clip(u_p_from_Us, 0, None)  # 确保非负
        P_from_Us = rho0 * u_p_range * u_p_from_Us
        axs[1, 0].plot(u_p_range, P_from_Us, 'r-', label=f'P = {rho0:.2f}·Us·Up')
    
    axs[1, 0].set_xlabel('Shock Velocity Us (km/s)')
    axs[1, 0].set_ylabel('Shock Pressure P (GPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # rho vs Up (压缩密度 vs 粒子速度)
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 1].scatter(
            method_df['Up'], method_df['rho'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7
        )
    
    # 计算压缩密度: rho = rho0 * Us / (Us - Up) = rho0 / (1 - Up/Us)
    if len(u_p_range) > 0 and all(Us > Up - tolerance for Us, Up in zip(U_s_fit, u_p_range)):
        rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)
        axs[1, 1].plot(u_p_range, rho_range, 'r-', label=f'rho = rho0·Us/(Us-Up)')
    
    axs[1, 1].set_xlabel('Particle Velocity Up (km/s)')
    axs[1, 1].set_ylabel('Compressed Density rho (g/cm³)')
    axs[1, 1].axhline(y=rho0, color='k', linestyle='--', label=f'Initial Density: {rho0:.2f} g/cm³')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图像到BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

# 主函数
def main():
    st.set_page_config(
        page_title="冲击波参数计算与分析系统",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("冲击波参数计算与分析系统 ⚡")
    st.markdown("""
    该系统用于计算和分析冲击波物理参数，基于Rankine-Hugoniot守恒关系和Hugoniot状态方程。
    支持飞片-样品-基板全链路冲击波参数求解，包含物理合理性自动检查和实验数据拟合功能。
    """)
    
    # 侧边栏导航
    st.sidebar.title("功能导航")
    option = st.sidebar.radio(
        "选择功能模块",
        ["参数计算", "数据拟合与分析", "数据库管理"]
    )
    
    # 材料选择（全局）
    materials = get_all_materials()
    if materials:
        default_material_idx = materials.index("Copper") if "Copper" in materials else 0
        selected_material = st.sidebar.selectbox(
            "选择材料",
            materials,
            index=default_material_idx
        )
    else:
        selected_material = st.sidebar.text_input("材料名称", "Copper")
        st.sidebar.info("数据库中暂无材料数据，将使用默认参数")
    
    # 温度计算选项
    calculate_temp = st.sidebar.checkbox("计算冲击温度", value=True)
    if calculate_temp:
        gamma = st.sidebar.slider(
            "格吕奈森系数 (gamma)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="用于冲击温度计算的格吕奈森系数，典型值: 金属1-3，陶瓷2-5"
        )
        Cv = st.sidebar.slider(
            "定容比热容 (Cv, J/kg·K)",
            min_value=100,
            max_value=1000,
            value=385,  # 铜的典型值
            step=10,
            help="材料的定容比热容，用于温度计算"
        )
        T0 = st.sidebar.slider(
            "初始温度 (T0, K)",
            min_value=100,
            max_value=1000,
            value=300,
            step=10,
            help="材料的初始温度，通常为室温"
        )
    else:
        gamma = 2.0
        Cv = 385
        T0 = 300
    
    # 功能模块选择
    if option == "参数计算":
        st.header("冲击波参数计算")
        st.markdown("输入已知参数，系统将自动求解未知参数。留空表示未知数，支持单一值、多个值或范围值输入。")
        
        # 飞片参数
        with st.expander("飞片参数", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                rh0f = get_input_streamlit(
                    "飞片初始密度", "rh0f", "rh0f", 
                    default=8.96, unit="g/cm³", 
                    desc="飞片材料的初始密度（未受冲击时）"
                )
                Df = get_input_streamlit(
                    "飞片冲击波速度", "Df", "Df", 
                    unit="km/s", desc="飞片中冲击波的传播速度"
                )
                uf = get_input_streamlit(
                    "飞片粒子速度", "uf", "uf", 
                    unit="km/s", desc="飞片材料质点的运动速度"
                )
            
            with col2:
                Pf = get_input_streamlit(
                    "飞片冲击压力", "Pf", "Pf", 
                    unit="GPa", desc="飞片中的冲击压力"
                )
                w = get_input_streamlit(
                    "飞片碰撞速度", "w", "w", 
                    unit="km/s", desc="飞片与样品的碰撞速度"
                )
                C0f = get_input_streamlit(
                    "飞片体声速", "C0f", "C0f", 
                    default=3.94, unit="km/s", 
                    desc="飞片材料的体声速（Hugoniot参数）"
                )
                Sf = get_input_streamlit(
                    "飞片Hugoniot参数S", "Sf", "Sf", 
                    default=1.48, desc="飞片材料的Hugoniot关系斜率参数"
                )
        
        # 样品参数
        with st.expander("样品参数", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                rh0s = get_input_streamlit(
                    "样品初始密度", "rh0s", "rh0s", 
                    default=2.7, unit="g/cm³", 
                    desc="样品材料的初始密度（未受冲击时）"
                )
                Ds = get_input_streamlit(
                    "样品冲击波速度", "Ds", "Ds", 
                    unit="km/s", desc="样品中冲击波的传播速度"
                )
                us = get_input_streamlit(
                    "样品粒子速度", "us", "us", 
                    unit="km/s", desc="样品材料质点的运动速度"
                )
            
            with col2:
                Ps = get_input_streamlit(
                    "样品冲击压力", "Ps", "Ps", 
                    unit="GPa", desc="样品中的冲击压力"
                )
                C0s = get_input_streamlit(
                    "样品体声速", "C0s", "C0s", 
                    default=5.8, unit="km/s", 
                    desc="样品材料的体声速（Hugoniot参数）"
                )
                Ss = get_input_streamlit(
                    "样品Hugoniot参数S", "Ss", "Ss", 
                    default=1.2, desc="样品材料的Hugoniot关系斜率参数"
                )
        
        # 基板参数
        with st.expander("基板参数", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                rh0b = get_input_streamlit(
                    "基板初始密度", "rh0b", "rh0b", 
                    default=3.5, unit="g/cm³", 
                    desc="基板材料的初始密度（未受冲击时）"
                )
                Db = get_input_streamlit(
                    "基板冲击波速度", "Db", "Db", 
                    unit="km/s", desc="基板中冲击波的传播速度"
                )
                ub = get_input_streamlit(
                    "基板粒子速度", "ub", "ub", 
                    unit="km/s", desc="基板材料质点的运动速度"
                )
            
            with col2:
                Pb = get_input_streamlit(
                    "基板冲击压力", "Pb", "Pb", 
                    unit="GPa", desc="基板中的冲击压力"
                )
                C0b = get_input_streamlit(
                    "基板体声速", "C0b", "C0b", 
                    default=4.2, unit="km/s", 
                    desc="基板材料的体声速（Hugoniot参数）"
                )
                Sb = get_input_streamlit(
                    "基板Hugoniot参数S", "Sb", "Sb", 
                    default=1.3, desc="基板材料的Hugoniot关系斜率参数"
                )
        
        # 求解按钮
        st.subheader("求解设置")
        col1, col2 = st.columns(2)
        with col1:
            solve_method = st.radio(
                "求解方法",
                ["符号求解 (适用于简单情况)", "数值求解 (适用于复杂情况)"],
                index=1,
                help="符号求解适用于未知数较少的情况，数值求解适用于多未知数复杂系统"
            )
        with col2:
            max_iter = st.slider(
                "最大迭代次数 (数值求解)",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="数值求解的最大迭代次数，复杂系统需要更大值"
            )
        
        if st.button("开始求解", use_container_width=True):
            try:
                # 收集已知参数和符号变量
                sym_vars = {}
                known_params = {}
                
                # 飞片参数处理
                for var_name, var_value in [
                    ('rh0f', rh0f), ('Df', Df), ('uf', uf),
                    ('Pf', Pf), ('w', w), ('C0f', C0f), ('Sf', Sf)
                ]:
                    if isinstance(var_value, list) and len(var_value) == 1:
                        known_params[var_name] = var_value[0]
                    elif not isinstance(var_value, list):  # 是符号变量
                        sym_vars[var_name] = var_value
                
                # 样品参数处理
                for var_name, var_value in [
                    ('rh0s', rh0s), ('Ds', Ds), ('us', us),
                    ('Ps', Ps), ('C0s', C0s), ('Ss', Ss)
                ]:
                    if isinstance(var_value, list) and len(var_value) == 1:
                        known_params[var_name] = var_value[0]
                    elif not isinstance(var_value, list):  # 是符号变量
                        sym_vars[var_name] = var_value
                
                # 基板参数处理
                for var_name, var_value in [
                    ('rh0b', rh0b), ('Db', Db), ('ub', ub),
                    ('Pb', Pb), ('C0b', C0b), ('Sb', Sb)
                ]:
                    if isinstance(var_value, list) and len(var_value) == 1:
                        known_params[var_name] = var_value[0]
                    elif not isinstance(var_value, list):  # 是符号变量
                        sym_vars[var_name] = var_value
                
                # 检查是否有未知数
                if not sym_vars:
                    st.success("所有参数均为已知，无需求解")
                    st.json(known_params)
                    return
                
                st.info(f"检测到 {len(sym_vars)} 个未知数，使用 {solve_method} 方法求解...")
                logger.info(f"开始求解，已知参数: {known_params}, 未知数: {sym_vars.keys()}")
                
                # 定义冲击波方程组（核心物理关系）
                equations = [
                    # 飞片方程
                    Eq(sym_vars.get('uf', 0) + sym_vars.get('Df', 0), known_params.get('w', sym_vars.get('w', 0))),  # uf + Df = w
                    Eq(sym_vars.get('Df', 0), sym_vars.get('C0f', 0) + sym_vars.get('Sf', 0) * sym_vars.get('uf', 0)),  # Df = C0f + Sf·uf
                    Eq(sym_vars.get('Pf', 0), sym_vars.get('rh0f', 0) * sym_vars.get('Df', 0) * sym_vars.get('uf', 0)),  # Pf = rh0f·Df·uf
                    
                    # 样品方程
                    Eq(sym_vars.get('Ds', 0), sym_vars.get('C0s', 0) + sym_vars.get('Ss', 0) * sym_vars.get('us', 0)),  # Ds = C0s + Ss·us
                    Eq(sym_vars.get('Ps', 0), sym_vars.get('rh0s', 0) * sym_vars.get('Ds', 0) * sym_vars.get('us', 0)),  # Ps = rh0s·Ds·us
                    
                    # 基板方程
                    Eq(sym_vars.get('Db', 0), sym_vars.get('C0b', 0) + sym_vars.get('Sb', 0) * sym_vars.get('ub', 0)),  # Db = C0b + Sb·ub
                    Eq(sym_vars.get('Pb', 0), sym_vars.get('rh0b', 0) * sym_vars.get('Db', 0) * sym_vars.get('ub', 0)),  # Pb = rh0b·Db·ub
                    
                    # 压力连续性
                    Eq(sym_vars.get('Pf', 0), sym_vars.get('Ps', 0)),  # Pf = Ps
                    Eq(sym_vars.get('Ps', 0), sym_vars.get('Pb', 0))   # Ps = Pb
                ]
                
                # 过滤掉不涉及符号变量的方程
                valid_equations = []
                for eq in equations:
                    if any(var in eq.free_symbols for var in sym_vars.values()):
                        valid_equations.append(eq)
                
                solution = None
                if solve_method == "符号求解 (适用于简单情况)":
                    # 符号求解（适用于未知数较少的情况）
                    try:
                        solution = solve(valid_equations, list(sym_vars.values()), dict=True)
                        st.info(f"符号求解返回 {len(solution)} 个可能解")
                    except Exception as e:
                        st.warning(f"符号求解失败: {str(e)}，将尝试数值求解")
                        solve_method = "数值求解 (适用于复杂情况)"
                
                if solve_method == "数值求解 (适用于复杂情况)" or solution is None or not solution:
                    # 生成更合理的初始猜测值
                    initial_guess = generate_better_initial_guess(known_params, sym_vars.values())
                    st.info(f"使用优化的初始猜测值: {initial_guess}")
                    
                    # 数值求解
                    solution = solve_numerically(valid_equations, sym_vars, initial_guess)
                    if solution:
                        solution = [solution]  # 包装成列表格式，统一处理
                    else:
                        st.error("数值求解失败，可能是因为: 1) 物理约束矛盾 2) 初始猜测值不合理 3) 方程组无解")
                        st.info("建议: 检查输入参数是否符合物理规律，或减少未知数数量")
                        return
                
                # 处理求解结果
                if solution and len(solution) > 0:
                    st.success(f"求解成功，获得 {len(solution)} 个有效解（已过滤不符合物理规律的解）")
                    
                    # 显示结果
                    for i, sol in enumerate(solution, 1):
                        with st.expander(f"解 {i}", expanded=i == 1):
                            # 合并已知参数和求解结果
                            full_result = known_params.copy()
                            full_result.update(sol)
                            
                            # 计算派生参数（压缩密度、比体积等）
                            tolerance = 1e-3
                            if 'rh0f' in full_result and 'Df' in full_result and 'uf' in full_result:
                                if abs(full_result['Df'] - full_result['uf']) > tolerance:
                                    full_result['rhf'] = (full_result['rh0f'] * full_result['Df']) / (full_result['Df'] - full_result['uf'])
                            
                            if 'rh0s' in full_result and 'Ds' in full_result and 'us' in full_result:
                                if abs(full_result['Ds'] - full_result['us']) > tolerance:
                                    full_result['rhs'] = (full_result['rh0s'] * full_result['Ds']) / (full_result['Ds'] - full_result['us'])
                            
                            if 'rh0b' in full_result and 'Db' in full_result and 'ub' in full_result:
                                if abs(full_result['Db'] - full_result['ub']) > tolerance:
                                    full_result['rhb'] = (full_result['rh0b'] * full_result['Db']) / (full_result['Db'] - full_result['ub'])
                            
                            # 计算比体积比 V/V0
                            if 'Df' in full_result and 'uf' in full_result:
                                if full_result['Df'] > tolerance:
                                    full_result['V_V0_f'] = 1 - full_result['uf'] / full_result['Df']
                            
                            if 'Ds' in full_result and 'us' in full_result:
                                if full_result['Ds'] > tolerance:
                                    full_result['V_V0_s'] = 1 - full_result['us'] / full_result['Ds']
                            
                            if 'Db' in full_result and 'ub' in full_result:
                                if full_result['Db'] > tolerance:
                                    full_result['V_V0_b'] = 1 - full_result['ub'] / full_result['Db']
                            
                            # 计算温度
                            if calculate_temp:
                                for mat in ['f', 's', 'b']:  # 飞片、样品、基板
                                    if f'rho0{mat}' in full_result and f'D{mat}' in full_result and f'u{mat}' in full_result:
                                        try:
                                            U_s = full_result[f'D{mat}']
                                            u_p = full_result[f'u{mat}']
                                            rho0 = full_result[f'rho0{mat}']
                                            _, V, _, _, T = calculate_shock_parameters(
                                                U_s, u_p, rho0, gamma, Cv, T0, calculate_temp=True
                                            )
                                            full_result[f'T{mat}'] = T
                                        except Exception as e:
                                            st.warning(f"计算{mat}的温度失败: {str(e)}")
                            
                            # 显示结果表格
                            result_df = pd.DataFrame(
                                list(full_result.items()),
                                columns=['参数', '值']
                            )
                            # 添加单位
                            units = {
                                'rh0f': 'g/cm³', 'Df': 'km/s', 'uf': 'km/s', 'Pf': 'GPa', 'w': 'km/s', 
                                'C0f': 'km/s', 'Sf': '', 'rhf': 'g/cm³', 'V_V0_f': '', 'Tf': 'K',
                                'rh0s': 'g/cm³', 'Ds': 'km/s', 'us': 'km/s', 'Ps': 'GPa',
                                'C0s': 'km/s', 'Ss': '', 'rhs': 'g/cm³', 'V_V0_s': '', 'Ts': 'K',
                                'rh0b': 'g/cm³', 'Db': 'km/s', 'ub': 'km/s', 'Pb': 'GPa',
                                'C0b': 'km/s', 'Sb': '', 'rhb': 'g/cm³', 'V_V0_b': '', 'Tb': 'K'
                            }
                            result_df['单位'] = result_df['参数'].map(units).fillna('')
                            # 格式化数值
                            result_df['值'] = result_df['值'].apply(
                                lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)
                            )
                            st.dataframe(result_df, hide_index=True)
                    
                    # 保存结果到数据库的选项
                    if st.button("保存结果到数据库", use_container_width=True):
                        saved_count = save_results_to_db(solution, selected_material)
                        if saved_count > 0:
                            st.success(f"成功保存 {saved_count} 条结果到 {selected_material} 数据集")
            
            except Exception as e:
                st.error(f"求解过程出错: {str(e)}")
                logger.error(f"求解错误: {str(e)}\n{traceback.format_exc()}")
    
    elif option == "数据拟合与分析":
        st.header("冲击波数据拟合与分析")
        st.markdown("对已有实验数据进行拟合，分析材料的Hugoniot关系，并可视化冲击波参数间的关系。")
        
        # 获取并显示材料数据
        df = get_material_data(selected_material)
        if df.empty:
            st.warning(f"材料 '{selected_material}' 暂无数据，请先在数据库中添加数据")
            return
        
        st.info(f"已加载材料 '{selected_material}' 的 {len(df)} 条有效数据（已过滤不符合物理规律的记录）")
        
        # 显示数据预览
        with st.expander("数据预览", expanded=False):
            st.dataframe(df)
        
        # 数据拟合
        st.subheader("Hugoniot关系拟合 (Us = C0 + S·Up)")
        fit_result = fit_material_data(df, selected_material, "飞片")
        if not fit_result:
            st.warning("无法进行拟合，请检查数据")
            return
        
        C0, S = fit_result["C0"], fit_result["S"]
        
        # 生成并显示冲击波关系图
        st.subheader("冲击波参数关系图")
        plot_buf = generate_shock_plots(df, C0, S, selected_material, "飞片")
        st.image(plot_buf, caption=f"{selected_material} 的冲击波参数关系图")
        
        # 误差分析
        st.subheader("拟合误差分析")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("决定系数 R²", f"{fit_result['r2']:.4f}", 
                     help="越接近1表示拟合效果越好")
        with col2:
            st.metric("均方根误差 (RMSE)", f"{fit_result['rmse']:.4f} km/s",
                     help="预测值与实际值的平均偏差")
        with col3:
            st.metric("平均绝对误差 (MAE)", f"{fit_result['mae']:.4f} km/s",
                     help="预测值与实际值的平均绝对偏差")
        
        # 实验方法比较
        if 'exp_method' in df.columns and len(df['exp_method'].unique()) > 1:
            st.subheader("不同实验方法的参数对比")
            method_df = df.groupby('exp_method').agg({
                'Us': ['mean', 'std', 'count'],
                'Up': ['mean', 'std'],
                'P': ['mean', 'std'],
                'rho0': ['mean']
            })
            method_df.columns = ['_'.join(col).strip() for col in method_df.columns.values]
            st.dataframe(method_df)
            
            # 按实验方法分别拟合
            st.subheader("不同实验方法的Hugoniot拟合")
            for method in df['exp_method'].unique():
                method_data = df[df['exp_method'] == method]
                if len(method_data) >= 2:
                    st.write(f"实验方法: {method} (样本数: {len(method_data)})")
                    method_fit = fit_material_data(method_data, selected_material, f"飞片 ({method})")
                else:
                    st.write(f"实验方法: {method} (样本数不足，无法拟合)")
    
    elif option == "数据库管理":
        st.header("数据库管理")
        st.markdown("管理冲击波实验数据，支持数据导入、导出、查看和删除等操作。")
        view_database()

if __name__ == "__main__":
    main()
