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
from sympy import symbols, Symbol, Equality, simplify, solve

# 全局设置 - 减少不必要的配置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.dpi"] = 70  # 降低默认DPI，加快图像生成

# 配置日志 - 减少文件IO操作
import logging
logging.basicConfig(
    level=logging.ERROR,  # 提高日志级别，减少日志输出
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
                pool_size=5,  # 优化连接池大小
                max_overflow=10,
                pool_recycle=3600
            )
            logger.info(f"成功创建MySQL引擎连接：{config['host']}:{config['port']}/{config['database']}")
            return engine
        else:  # SQLite
            config = DB_CONFIG["sqlite"]
            engine = create_engine(
                f"sqlite:///{config['path']}",
                pool_size=5,  # 优化连接池大小
                max_overflow=10,
                pool_recycle=3600
            )
            logger.info(f"成功创建SQLite引擎连接：{config['path']}")
            return engine
    except Exception as e:
        logger.error(f"创建数据库引擎失败: {str(e)}")
        st.error(f"数据库连接失败: {str(e)}")
        return None

# 初始化数据库引擎（全局单例）
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = create_db_engine(DB_TYPE)
db_engine = st.session_state.db_engine

# SQLite性能优化
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if DB_TYPE == "sqlite":
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')  # 预写日志
        cursor.execute('PRAGMA synchronous=NORMAL;')  # 同步模式
        cursor.execute('PRAGMA temp_store=MEMORY;')   # 临时存储
        cursor.execute('PRAGMA cache_size=-40000;')   # 增加缓存（40MB）
        cursor.close()

# 核心函数：数据库查询+全链路日志（适配多数据库）
@st.cache_data(ttl=600)  # 缓存查询结果10分钟
def query_database(sql, params=None, db_type=DB_TYPE):
    """通用数据库查询函数，支持参数化查询和多数据库类型"""
    conn = None
    cursor = None
    try:
        # 1. 建立数据库连接
        engine = create_db_engine(db_type)
        if not engine:
            return None
            
        conn = engine.connect()
        config = DB_CONFIG[db_type]
        
        # 2. 执行查询
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
        if exec_time > 1.0:  # 只记录慢查询
            logger.debug(f"SQL执行完成，耗时：{exec_time:.3f}秒")
        
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
        if conn:
            conn.close()

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
                        T REAL,          -- 冲击温度 (K)
                        INDEX idx_material (material)  -- 索引以加快查询
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
if 'db_initialized' not in st.session_state:
    init_database()
    fix_database_schema()
    st.session_state.db_initialized = True

# 物理合理性检查函数
def validate_physical合理性(data, material_type="通用"):
    """检查数据是否符合冲击波物理规律，返回错误信息列表"""
    errors = []
    
    # 基本物理约束检查
    if 'rho0' in data and data['rho0'] is not None and (data['rho0'] <= 0 or data['rho0'] > 20):
        errors.append(f"{material_type}初始密度必须为正数且通常小于20 g/cm³，当前值: {data['rho0']}")
    
    if 'Us' in data and data['Us'] is not None and data['Us'] <= 0:
        errors.append(f"{material_type}冲击波速度必须为正数，当前值: {data['Us']}")
    
    if 'Up' in data and data['Up'] is not None and data['Up'] < 0:
        errors.append(f"{material_type}粒子速度不能为负数，当前值: {data['Up']}")
    
    # Hugoniot关系检查：冲击波速度必须大于粒子速度
    if 'Us' in data and 'Up' in data and data['Us'] is not None and data['Up'] is not None:
        if data['Us'] <= data['Up'] + 1e-6:  # 考虑浮点数精度
            errors.append(f"{material_type}冲击波速度(Us={data['Us']})必须大于粒子速度(Up={data['Up']})")
    
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

@st.cache_data(ttl=1800)  # 缓存30分钟
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
            # 向量优化物理合理性检查
            valid_mask = (df['Us'] > df['Up'] + 1e-6) & (df['Us'] > 0) & (df['Up'] >= 0)
            if 'rho0' in df.columns and 'rho' in df.columns:
                valid_mask &= (df['rho'] > df['rho0'] - 1e-6)
            if 'V_V0' in df.columns:
                valid_mask &= (df['V_V0'] < 1 - 1e-6)
            
            invalid_count = len(df) - valid_mask.sum()
            if invalid_count > 0:
                st.warning(f"材料 {material_name} 中有 {invalid_count} 条记录不符合物理规律，已自动过滤")
                df = df[valid_mask]
        
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
        
        # 批量处理数据
        valid_data = []
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
            valid_data.append(data)
        
        if invalid_count > 0:
            st.warning(f"过滤了 {invalid_count} 个不合理解，未保存到数据库")
        
        # 批量插入
        if valid_data:
            with db_engine.begin() as conn:
                stmt = text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, valid_data)
                count = len(valid_data)
        
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
        # 批量验证数据
        valid_mask = (df['Us'] > df['Up'] + 1e-6) & (df['Us'] > 0) & (df['Up'] >= 0) & (df['rho0'] > 0)
        valid_df = df[valid_mask].copy()
        invalid_count = len(df) - len(valid_df)
        
        if invalid_count > 0:
            st.warning(f"过滤了 {invalid_count} 个不合理解，未导入数据库")
        
        if valid_df.empty:
            return 0
        
        # 准备批量插入数据
        valid_df['material'] = material_name
        valid_df['exp_method'] = exp_method
        
        # 填充缺失列
        for col in ['V', 'rho', 'V_V0', 'gamma', 'T']:
            if col not in valid_df.columns:
                valid_df[col] = 0
        
        # 批量插入
        with db_engine.begin() as conn:
            valid_df.to_sql(
                'shock_wave_all_data',
                conn,
                if_exists='append',
                index=False,
                columns=['material', 'rho0', 'Us', 'Up', 'P', 'V', 'rho', 'V_V0', 'exp_method', 'gamma', 'T']
            )
        
        count = len(valid_df)
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
            st.session_state.db_engine = create_db_engine(DB_TYPE)
            db_engine = st.session_state.db_engine
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
            
            # 提供CSV模板下载
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
    # 物理约束检查
    if U_s <= u_p:
        raise ValueError(f"冲击波速度 (Us={U_s}) 必须大于粒子速度 (Up={u_p})")
    if rho0 <= 0:
        raise ValueError(f"初始密度 (rho0={rho0}) 必须为正数")
    if U_s <= 0 or u_p < 0:
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
    if rho <= rho0:
        raise ValueError(f"计算的压缩密度 (rho={rho}) 必须大于初始密度 (rho0={rho0})")
    if V_V0 >= 1:
        raise ValueError(f"计算的比体积比 (V/V0={V_V0}) 必须小于1")
    if P <= 0:
        raise ValueError(f"计算的压力 (P={P}) 必须为正数")
    
    T = None
    if calculate_temp:
        # 温度计算（Mie-Grüneisen方程近似）
        E_shock = 0.5 * P * (1/rho0 - V) * 1e6  # 冲击内能 (J/kg)
        T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # 冲击温度 (K)
        
        if T < T0:
            raise ValueError(f"计算的冲击温度 (T={T}) 必须高于初始温度 (T0={T0})")
    
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
    
    # 物理约束
    if C0 <= 0:
        C0 = max(1.0, abs(C0))  # 确保体声速为正数且合理
        
    if S < 1.0 or S > 3.0:
        st.warning(f"Hugoniot参数 (S={S}) 超出典型范围 (1.0-3.0)，可能存在数据问题")
        
    return C0, S

@st.cache_data(ttl=3600)  # 缓存拟合结果
def fit_material_data(df, material_name, material_type):
    if df is None or df.empty:
        st.warning(f"{material_type}材料 '{material_name}' 没有数据")
        return None
    
    # 过滤异常值
    df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
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
    
    # 物理约束检查
    if C0 <= 0:
        st.warning(f"{material_type}材料 '{material_name}' 拟合的体声速 (C0={C0}) 为非正数，已调整")
        C0 = max(1.0, abs(C0))
        
    if S < 1.0 or S > 3.0:
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

# 输入函数 - 修复参数共享问题
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
            # 基本物理范围检查
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

# 数值求解器（优化版本，提高求解速度）
def solve_numerically(eqs, sym_vars, initial_guess):
    """使用数值方法求解方程组，优化求解速度"""
    # 明确变量顺序
    var_names = sorted(sym_vars.keys(), key=lambda k: str(k))
    var_list = [sym_vars[name] for name in var_names]
    initial_values = [initial_guess[name] for name in var_names]
    
    def residuals(x):
        """计算残差：简化计算逻辑提高速度"""
        substitutions = {var_list[i]: x[i] for i in range(len(x))}
        residuals = []
        for eq in eqs:
            try:
                # 处理SymPy等式方程
                if isinstance(eq, Equality):
                    residual_expr = eq.lhs - eq.rhs
                else:
                    residual_expr = eq
                    
                substituted = residual_expr.subs(substitutions)
                residual_value = float(abs(substituted.evalf(n=6)))  # 降低精度要求提高速度
                residuals.append(residual_value)
            except:
                residuals.append(1e10)  # 计算失败时给予大残差
        return residuals
    
    # 边界设置：保持物理变量约束
    n_vars = len(initial_guess)
    lower_bounds = []
    upper_bounds = []
    for var in var_names:
        var_str = str(var)
        if var_str.startswith(('rh0', 'rh')):
            lower_bounds.append(0.1)
            upper_bounds.append(20.0)
        elif var_str.startswith(('D', 'C0', 'u', 'w')):
            lower_bounds.append(0.1)
            upper_bounds.append(30.0)
        elif var_str.startswith(('P', 'E')):
            lower_bounds.append(0.01)
            upper_bounds.append(5000.0)
        elif var_str.startswith('gamma'):
            lower_bounds.append(0.5)
            upper_bounds.append(5.0)
        elif var_str.startswith('T'):
            lower_bounds.append(300.0)
            upper_bounds.append(1e5)
        else:
            lower_bounds.append(-1e10)
            upper_bounds.append(1e10)
    
    # 优化器参数：减少迭代次数但保持精度
    result = least_squares(
        residuals,
        initial_values,
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-6,  # 适当降低精度要求
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=5000,  # 减少迭代次数
        method='trf',
        jac='3-point',
        verbose=0
    )
    
    if result.success:
        # 确保解与变量名正确映射
        solution = {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
        return solution
    return None

# 冲击波关系图绘制 - 优化图表生成速度
@st.cache_data(ttl=3600)  # 缓存图像结果
def generate_shock_plots(df, C0, S, material_name, material_type):
    # 数据量大时进行采样，大幅减少数据点数量
    if len(df) > 500:
        df = df.sample(500, random_state=42)
        
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 减小图表尺寸
    
    # 定义实验方法的颜色映射
    method_colors = {
        'iml': 'red',
        'ssp': 'blue',
        'calculated': 'green',
        'manual_input': 'purple',
        'bulk_import': 'orange'
    }
    default_color = 'gray'
    
    # 标题使用英文
    fig.suptitle(f'Material: {material_name} - Shock Wave Relationships', fontsize=14)
    
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
            color=color, alpha=0.7, s=20  # 减小点大小
        )
    
    # 确保Hugoniot参数合理
    if C0 <= 0:
        C0 = 3.0  # 默认合理值
    if S < 1.0 or S > 3.0:
        S = 1.5  # 默认合理值
        
    u_p_range = np.linspace(0, min(20, df['Up'].max()*1.1), 50)  # 减少采样点
    
    U_s_fit = C0 + S * u_p_range  # Hugoniot关系
    axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'Fit: Us = {C0:.2f} + {S:.2f}·Up')
    axs[0, 0].set_xlabel('Particle Velocity Up (km/s)')
    axs[0, 0].set_ylabel('Shock Velocity Us (km/s)')
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True)
    
    # P vs Up
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[0, 1].scatter(
            method_df['Up'], method_df['P'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7, s=20
        )
    
    # 使用数据中的平均密度
    rho0 = df['rho0'].mean() if not df.empty else 8.96
    P_range = rho0 * U_s_fit * u_p_range  # 动量守恒关系
    
    axs[0, 1].plot(u_p_range, P_range, 'r-', label='Theoretical: P = ρ0·Us·Up')
    axs[0, 1].set_xlabel('Particle Velocity Up (km/s)')
    axs[0, 1].set_ylabel('Pressure P (GPa)')
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True)
    
    # P vs V/V0
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 0].scatter(
            method_df['V_V0'], method_df['P'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7, s=20
        )
    
    V_V0_range = 1 - u_p_range / U_s_fit  # V/V0 = 1 - Up/Us
    
    axs[1, 0].axvline(x=1.0, color='k', linestyle='--', label='V/V0 = 1')
    axs[1, 0].plot(V_V0_range, P_range, 'r-', label='Theoretical Curve')
    axs[1, 0].set_xlabel('Specific Volume Ratio V/V0')
    axs[1, 0].set_ylabel('Pressure P (GPa)')
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True)
    
    # rho vs P
    for method in methods:
        method_df = df[df['exp_method'] == method]
        color = method_colors.get(method.lower(), default_color)
        axs[1, 1].scatter(
            method_df['P'], method_df['rho'], 
            label=f'{method}' if method == methods[0] else "",
            color=color, alpha=0.7, s=20
        )
    
    rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)  # rho = rho0·Us/(Us-Up)
    
    # 添加初始密度参考线
    axs[1, 1].axhline(y=rho0, color='k', linestyle='--', label=f'Initial density: {rho0:.2f}')
    axs[1, 1].plot(P_range, rho_range, 'r-', label='Theoretical Curve')
    axs[1, 1].set_xlabel('Pressure P (GPa)')
    axs[1, 1].set_ylabel('Density ρ (g/cm³)')
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def save_plot_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')  # 降低分辨率
    buf.seek(0)
    return buf

# 材料图像显示辅助函数
def display_material_plots(df, material_name, material_type):
    if not df.empty:
        with st.expander(f"查看 {material_type} 材料 {material_name} 的冲击波关系图", expanded=False):
            # 先检查是否有足够数据进行拟合
            valid_data = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
            if len(valid_data) >= 2:
                C0, S = fit_hugoniot(valid_data)
                fig = generate_shock_plots(valid_data, C0, S, material_name, material_type)
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
                st.info(f"{material_type} 材料 {material_name} 的有效数据不足，无法生成图表")
    else:
        st.info(f"没有可用数据生成 {material_type} 材料 {material_name} 的图表")

# 结果绘图函数 - 优化图表生成
@st.cache_data(ttl=3600)  # 缓存图像结果
def plot_results_streamlit(results, calculate_temp=True):
    if not results:
        return None
        
    # 数据量大时进行采样
    if len(results) > 500:
        results = results[:500]
        
    # 确定子图数量
    subplot_count = 4 if calculate_temp else 3
    fig = plt.figure(figsize=(14, 7) if calculate_temp else (14, 5))  # 减小图表尺寸
    
    # 原始数据
    pf_values = [r.get('Pf', 0) for r in results]
    uf_values = [r.get('uf', 0) for r in results]
    df_values = [r.get('Df', 0) for r in results]
    rhf_values = [r.get('rhf', 0) for r in results]
    
    # 1. 压力-粒子速度图
    ax1 = fig.add_subplot(221 if calculate_temp else 221)
    ax1.errorbar(uf_values, pf_values, 
                 yerr=[r.get('Pf_err', 0.1) for r in results],
                 xerr=[r.get('uf_err', 0.05) for r in results],
                 fmt='bo', ecolor='r', capsize=3, label='Flyer data', markersize=3)  # 减小点大小
    ax1.set_xlabel('粒子速度 Up (km/s)')
    ax1.set_ylabel('冲击压力 P (GPa)')
    ax1.set_title('压力-粒子速度关系')
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    # 2. 温度-压力图（仅当计算温度时显示）
    ax2 = None
    if calculate_temp:
        # 温度相关数据
        tf_values = [r.get('Tf', 0) for r in results]
        
        ax2 = fig.add_subplot(222)
        ax2.scatter(pf_values, tf_values, c='orange', label='飞片温度', s=10)  # 减小点大小
        ax2.axhline(y=300, color='k', linestyle='--', label='室温 (300 K)')
        ax2.set_xlabel('冲击压力 P (GPa)')
        ax2.set_ylabel('冲击温度 T (K)')
        ax2.set_title('温度-压力关系')
        ax2.legend(fontsize=8)
        ax2.grid(True)
    
    # 3. 冲击波速度-粒子速度图
    ax3 = fig.add_subplot(223 if calculate_temp else 222)
    ax3.scatter(uf_values, df_values, c='blue', label='飞片', s=10)
    ax3.set_xlabel('粒子速度 Up (km/s)')
    ax3.set_ylabel('冲击波速度 Us (km/s)')
    ax3.set_title('冲击波速度-粒子速度关系')
    ax3.legend(fontsize=8)
    ax3.grid(True)
    
    # 4. 密度-压力图
    ax4 = fig.add_subplot(224 if calculate_temp else 223)
    ax4.scatter(pf_values, rhf_values, c='green', label='飞片', s=10)
    # 添加初始密度参考线
    if results and 'rh0f' in results[0]:
        avg_rh0 = np.mean([r.get('rh0f', 0) for r in results])
        ax4.axhline(y=avg_rh0, color='k', linestyle='--', label=f'平均初始密度: {avg_rh0:.2f}')
    ax4.set_xlabel('冲击压力 P (GPa)')
    ax4.set_ylabel('压缩密度 (g/cm³)')
    ax4.set_title('密度-压力关系')
    ax4.legend(fontsize=8)
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
            st.rerun()
    with col2:
        if st.button("手动输入参数"):
            st.session_state.page = "manual_mode"
            st.rerun()

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
    
    # 按需查询字段以减少数据传输
    flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
    base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
    sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho', 'exp_method', 'gamma'])
    
    # 为每种材料类型拟合数据
    with st.spinner(f"正在拟合飞片材料 {flyer_material} 数据..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material, "飞片")
    
    with st.spinner(f"正在拟合基板材料 {base_material} 数据..."):
        base_fit = fit_material_data(base_df, base_material, "基板")
    
    with st.spinner(f"正在拟合样品材料 {sample_material} 数据..."):
        sample_fit = fit_material_data(sample_df, sample_material, "样品")
    
    # 冲击波参数分析部分
    st.subheader("冲击波参数分析（Hugoniot关系）")
    st.caption("基于线性Hugoniot关系 Us = C0 + S·Up，其中C0为体声速，S为Hugoniot参数")
    
    # 显示图表（默认折叠以提高初始加载速度）
    with st.expander("显示材料冲击波关系图表", expanded=False):
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
    
    # 飞片与基板界面速度关系说明
    st.info("飞片速度w与粒子速度uf的关系为w = Df + uf（实验室坐标系）")
    
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
                # 为初始密度设置默认值
                if var == "rh0f" and default_val is None:
                    default_val = 8.96  # 铜的默认密度
                
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
    with st.expander(f"{base_material} 基板参数", expanded=False):  # 默认折叠以提高加载速度
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
    
    # 样品参数
    with st.expander(f"{sample_material} 样品参数", expanded=False):  # 默认折叠以提高加载速度
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
    
    # 保存当前参数按钮
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
        max_value=500,  # 减少最大组合数
        value=min(50, total_combinations)  # 减少默认组合数
    )
    
    if st.button("开始求解"):
        valid = True
        # 检查关键参数是否已输入
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
            
            # 方程组
            eqs = [
                # 飞片质量守恒: rho0f·Df = rhf·(Df - uf)
                Equality(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # 飞片速度与粒子速度关系: w = Df + uf
                Equality(sym_vars['w'] - (sym_vars['Df'] + sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Equality(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·uf
                Equality(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf'], 0),
                # 基板动量守恒: Pb = rho0b·Db·ub
                Equality(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # 飞片能量守恒: Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Equality(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # 基板能量守恒: Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Equality(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # 飞片Hugoniot关系: Df = C0f + Sf·uf
                Equality(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*sym_vars['uf'], 0),
                # 基板Hugoniot关系: Db = C0b + Sb·ub
                Equality(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                # 界面压力连续性: Pf = Pb
                Equality(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # 界面粒子速度连续性: uf = ub
                Equality(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            # 温度相关方程（仅当计算温度时添加）
            if calculate_temp:
                # 飞片温度方程 (Mie-Grüneisen)
                eqs.append(Equality(sym_vars['Tf'] - 300 - (sym_vars['Ef'] - sym_vars['E0f'])*1e6 / 
                             (Cv_values['f'] * (1 + sym_vars['gammaf']/2)), 0))
                # 基板温度方程
                eqs.append(Equality(sym_vars['Tb'] - 300 - (sym_vars['Eb'] - sym_vars['E0b'])*1e6 / 
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
                    Equality(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Equality(sym_vars['ub'] - sym_vars['us'], 0),  # 速度连续性
                    Equality(sym_vars['rhb'] - sym_vars['rhs'], 0), # 密度连续性
                    Equality(sym_vars['Db'] - sym_vars['Ds'], 0),  # 冲击波速度连续性
                    # 样品能量守恒
                    Equality(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
                ]
                
                # 温度相关方程（仅当计算温度时添加）
                if calculate_temp:
                    eqs += [
                        Equality(sym_vars['Tb'] - sym_vars['Ts'], 0),
                        Equality(sym_vars['gammab'] - sym_vars['gammas'], 0)
                    ]
            else:
                # 样品与基板为不同材料：单独计算
                eqs += [
                    # 样品质量守恒
                    Equality(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    # 基板-样品界面动量守恒
                    Equality(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    # 样品动量守恒
                    Equality(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    # 样品能量守恒
                    Equality(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # 样品Hugoniot关系
                    Equality(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                    # 基板-样品界面Hugoniot关系
                    Equality(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Equality(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Equality(sym_vars['ub'] - sym_vars['us'], 0)   # 速度连续性
                ]
                
                # 温度相关方程（仅当计算温度时添加）
                if calculate_temp:
                    eqs.append(Equality(sym_vars['Ts'] - 300 - (sym_vars['Es'] - sym_vars['E0s'])*1e6 / 
                                 (Cv_values['s'] * (1 + sym_vars['gammas']/2)), 0))
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # 构建初始猜测值
                initial_guess = {}
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
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            # 默认不显示图表，让用户选择是否查看
            with st.expander("显示结果可视化图表", expanded=False):
                fig = plot_results_streamlit(results, calculate_temp)
                if fig:
                    st.pyplot(fig)
                    buf2 = BytesIO()
                    fig.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
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
    
    if st.button("返回主页"):
        st.session_state.page = "home"
        st.rerun()

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
    
    # 材料参数输入
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.text_input("飞片材料名称", value="铜", help="输入材料名称，例如：铜、铝")
    with col2:
        base_material = st.text_input("基板材料名称", value="铝", help="输入材料名称，例如：铜、铝")
    with col3:
        sample_material = st.text_input("样品材料名称", value="铜", help="输入材料名称，例如：铜、铝")
    
    # 飞片与基板界面速度关系说明
    st.info("飞片速度w与粒子速度uf的关系为w = Df + uf（实验室坐标系）")
    
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
    
    exp_method = st.text_input("实验方法/数据来源", value="manual_input", help="记录数据来源，例如：iml、ssp、实验设备、文献等")
    
    # 参数输入
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # 飞片参数
    with st.expander(f"{flyer_material} 飞片参数", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["f"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                default_val = 2.0 if var == "gammaf" else None
                # 为初始密度设置默认值
                if var == "rh0f":
                    default_val = 8.96  # 铜的默认密度
                
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
                         "飞片冲击波速度（对应Us）" if var == "Df" else
                         "飞片体声速（Hugoniot拟合）" if var == "C0f" else
                         "飞片Hugoniot参数S（无量纲）" if var == "Sf" else
                         "飞片初始内能密度" if var == "E0f" else
                         "飞片压缩后内能密度" if var == "Ef" else
                         "飞片粒子速度（对应Up）" if var == "uf" else
                         "飞片初始冲击速度" if var == "w" else
                         "飞片冲击压力" if var == "Pf" else
                         "飞片格吕奈森系数" if var == "gammaf" else
                         "飞片冲击温度"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # 基板参数（默认折叠）
    with st.expander(f"{base_material} 基板参数", expanded=False):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                default_val = 2.0 if var == "gammab" else None
                # 为初始密度设置默认值
                if var == "rh0b":
                    default_val = 2.7  # 铝的默认密度
                    
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

    # 样品参数（默认折叠）
    with st.expander(f"{sample_material} 样品参数", expanded=False):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            # 温度参数仅在计算温度时显示
            if var.startswith('T') and not calculate_temp:
                continue
                
            with cols[i % 3]:
                default_val = 2.0 if var == "gammas" else None
                # 为初始密度设置默认值
                if var == "rh0s":
                    default_val = 8.96  # 铜的默认密度
                    
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

    # 保存当前参数按钮
    col_save, col_other = st.columns([1, 3])
    with col_save:
        if st.button("保存当前参数到数据库"):
            # 准备输入数据字典
            input_data = {
                'rho0': input_params.get('rh0f'),
                'Us': input_params.get('Df'),
                'Up': input_params.get('uf'),
                'P': input_params.get('Pf'),
                'gamma': input_params.get('gammaf'),
                'T': input_params.get('Tf') if calculate_temp else None
            }
            # 过滤符号变量（未知数）
            filtered_data = {k: v for k, v in input_data.items() if not isinstance(v, Symbol) and v is not None}
            
            # 确保至少有必要的参数
            if len(filtered_data) >= 3:  # 至少需要3个参数才能保存
                count = save_input_data_to_db(filtered_data, flyer_material, exp_method)
                if count > 0:
                    st.success(f"已保存到 {flyer_material} 数据集，共 {count} 条记录")
            else:
                st.warning("请至少输入3个有效的参数才能保存")
    
    # 参数组合限制
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "最大参数组合数（过多会影响速度）", 
        min_value=10, 
        max_value=500,
        value=min(50, total_combinations)
    )
    
    if st.button("开始求解"):
        valid = True
        # 检查关键参数是否已输入
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
        
        # 截断过多的组合以提高性能
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
            
            # 构建基本方程组（与数据库模式相同的核心物理方程）
            eqs = [
                # 飞片质量守恒: rho0f·Df = rhf·(Df - uf)
                Equality(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # 飞片速度与粒子速度关系: w = Df + uf
                Equality(sym_vars['w'] - (sym_vars['Df'] + sym_vars['uf']), 0),
                # 基板质量守恒: rho0b·Db = rhb·(Db - ub)
                Equality(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # 飞片动量守恒: Pf = rho0f·Df·uf
                Equality(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*sym_vars['uf'], 0),
                # 基板动量守恒: Pb = rho0b·Db·ub
                Equality(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # 飞片能量守恒: Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Equality(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # 基板能量守恒: Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Equality(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # 飞片Hugoniot关系: Df = C0f + Sf·uf
                Equality(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*sym_vars['uf'], 0),
                # 基板Hugoniot关系: Db = C0b + Sb·ub
                Equality(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                # 界面压力连续性: Pf = Pb
                Equality(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # 界面粒子速度连续性: uf = ub
                Equality(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            # 添加温度相关方程（如果启用）
            if calculate_temp:
                # 飞片温度方程 (Mie-Grüneisen)
                eqs.append(Equality(sym_vars['Tf'] - 300 - (sym_vars['Ef'] - sym_vars['E0f'])*1e6 / 
                             (Cv_values['f'] * (1 + sym_vars['gammaf']/2)), 0))
                # 基板温度方程
                eqs.append(Equality(sym_vars['Tb'] - 300 - (sym_vars['Eb'] - sym_vars['E0b'])*1e6 / 
                             (Cv_values['b'] * (1 + sym_vars['gammab']/2)), 0))
            
            # 判断样品与基板是否为同一材料
            try:
                is_same_material = (
                    str(input_params['rh0s']) == str(input_params['rh0b']) and
                    str(input_params['C0s']) == str(input_params['C0b']) and
                    str(input_params['Ss']) == str(input_params['Sb'])
                )
            except:
                is_same_material = False
                
            if is_same_material:
                # 样品与基板为同一材料：参数连续
                eqs += [
                    Equality(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Equality(sym_vars['ub'] - sym_vars['us'], 0),  # 速度连续性
                    Equality(sym_vars['rhb'] - sym_vars['rhs'], 0), # 密度连续性
                    Equality(sym_vars['Db'] - sym_vars['Ds'], 0),  # 冲击波速度连续性
                    # 样品能量守恒
                    Equality(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0)
                ]
                
                if calculate_temp:
                    eqs += [
                        Equality(sym_vars['Tb'] - sym_vars['Ts'], 0),
                        Equality(sym_vars['gammab'] - sym_vars['gammas'], 0)
                    ]
            else:
                # 样品与基板为不同材料：单独计算
                eqs += [
                    # 样品质量守恒
                    Equality(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    # 基板-样品界面动量守恒
                    Equality(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    # 样品动量守恒
                    Equality(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    # 样品能量守恒
                    Equality(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # 样品Hugoniot关系
                    Equality(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                    # 基板-样品界面Hugoniot关系
                    Equality(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Equality(sym_vars['Pb'] - sym_vars['Ps'], 0),  # 压力连续性
                    Equality(sym_vars['ub'] - sym_vars['us'], 0)   # 速度连续性
                ]
                
                if calculate_temp:
                    eqs.append(Equality(sym_vars['Ts'] - 300 - (sym_vars['Es'] - sym_vars['E0s'])*1e6 / 
                                 (Cv_values['s'] * (1 + sym_vars['gammas']/2)), 0))
            
            # 代入已知参数
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # 构建初始猜测值（基于物理合理性）
                initial_guess = {}
                known_params = {}
                for k, v in current_subs.items():
                    try:
                        known_params[str(k)] = float(v)
                    except:
                        pass
                
                for var in remaining_vars:
                    var_str = str(var)
                    # 基于物理关系设置初始猜测值
                    if var_str == 'w' and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['Df'] + known_params['uf']
                    elif var_str == 'Df' and 'C0f' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['C0f'] + 1.5 * known_params['uf']
                    elif var_str == 'Pf' and 'rh0f' in known_params and 'Df' in known_params and 'uf' in known_params:
                        initial_guess[var] = known_params['rh0f'] * known_params['Df'] * known_params['uf']
                    elif var_str.startswith('rh'):  # 密度参数
                        initial_guess[var] = known_params.get('rh0f', 8.0) * 1.2  # 压缩后密度略大
                    elif var_str.startswith(('u', 'D', 'C0')):  # 速度参数
                        initial_guess[var] = known_params.get('w', 10.0) / 2
                    elif var_str.startswith('P'):  # 压力参数
                        initial_guess[var] = known_params.get('rh0f', 8.0) * (known_params.get('w', 10.0)/2)**2
                    elif var_str.startswith('gamma'):  # 格吕奈森系数
                        initial_guess[var] = 2.0
                    elif var_str.startswith('T'):  # 温度参数
                        initial_guess[var] = 3000.0
                    else:  # 其他参数
                        initial_guess[var] = 1.0
                
                # 数值求解
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
                st.warning(f"求解错误: {str(e)}（请检查参数是否符合物理约束）")
                invalid_solutions += 1
        
        if results:
            st.success(f"求解完成，找到 {len(results)} 个有效解（已过滤 {invalid_solutions} 个不合理解）")
            
            st.subheader("计算结果（单位：rho=g/cm³, D=km/s, u=km/s, P=GPa, T=K）")
            df = pd.DataFrame(results)
            # 选择常用列显示
            display_cols = ['rh0f', 'Df', 'uf', 'Pf', 'rhf', 'w', 'Tf', 
                           'rh0b', 'Db', 'ub', 'Pb', 'Tb',
                           'rh0s', 'Ds', 'us', 'Ps', 'Ts']
            display_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[display_cols])
            
            # 结果下载
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载结果数据",
                data=csv,
                file_name="manual_solver_results.csv",
                mime="text/csv",
            )
            
            # 结果可视化
            with st.expander("显示结果图表", expanded=False):
                fig = plot_results_streamlit(results, calculate_temp)
                if fig:
                    st.pyplot(fig)
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="下载图表",
                        data=buf,
                        file_name="manual_analysis_results.png",
                        mime="image/png"
                    )
            
            # 保存结果到数据库
            if st.button("保存结果到数据库"):
                count = save_results_to_db(results, sample_material)
                if count > 0:
                    st.success(f"已保存 {count} 条记录到 {sample_material} 数据集")
        else:
            st.warning(f"未找到有效解，尝试了 {total} 组参数")
    
    # 返回主页按钮
    if st.button("返回主页"):
        st.session_state.page = "home"
        st.rerun()

# 主程序入口
def main():
    # 初始化会话状态
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    
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
        if st.button("返回上一页"):
            prev_page = st.session_state.get('previous_page', 'home')
            st.session_state.page = prev_page
            st.rerun()

if __name__ == "__main__":
    main()
