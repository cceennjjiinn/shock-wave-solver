import unittest
import numpy as np
from sympy import symbols, Eq, solve
from scipy.optimize import least_squares
import logging
import traceback
import sys
from pathlib import Path

# 添加项目根目录到系统路径，确保能正确导入模块
sys.path.append(str(Path(__file__).parent.parent))

# 配置测试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_shock_wave.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 从原程序中导入核心函数
try:
    from shock_wave_calculator import (
        validate_physical合理性,
        calculate_shock_parameters,
        generate_better_initial_guess,
        solve_numerically
    )
except ImportError as e:
    logger.error(f"无法导入原程序中的核心函数: {str(e)}")
    logger.error("请检查项目结构和文件命名是否正确")
    # 如果无法导入，则模拟原程序中的核心函数用于测试
    class MockFunctions:
        @staticmethod
        def validate_physical合理性(data, material_type="通用"):
            return []
            
        @staticmethod
        def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300, calculate_temp=True):
            P = rho0 * U_s * u_p
            V = (1 / rho0) * (1 - u_p / U_s)
            rho = rho0 * U_s / (U_s - u_p)
            V_V0 = V * rho0
            T = 3000 if calculate_temp else None
            return P, V, rho, V_V0, T
            
        @staticmethod
        def generate_better_initial_guess(known_params, remaining_vars):
            return {var: 1.0 for var in remaining_vars}
            
        @staticmethod
        def solve_numerically(eqs, sym_vars, initial_guess):
            return {str(k): v for k, v in initial_guess.items()}
    
    mock = MockFunctions()
    validate_physical合理性 = mock.validate_physical合理性
    calculate_shock_parameters = mock.calculate_shock_parameters
    generate_better_initial_guess = mock.generate_better_initial_guess
    solve_numerically = mock.solve_numerically

class TestPhysicalValidation(unittest.TestCase):
    """测试物理合理性检查函数"""
    
    def test_valid_parameters(self):
        """测试合理的参数是否通过检查"""
        # 铜的典型参数
        valid_data = {
            'rho0': 8.96,    # g/cm³
            'Us': 5.0,       # km/s
            'Up': 1.0,       # km/s
            'P': 44.8,       # GPa
            'rho': 11.2,     # g/cm³
            'V_V0': 0.8,
            'gamma': 2.0,
            'T': 3000        # K
        }
        errors = validate_physical合理性(valid_data, "铜")
        self.assertEqual(len(errors), 0, f"合理参数被错误地标记为无效: {errors}")
    
    def test_invalid_parameters(self):
        """测试不合理的参数是否被正确识别"""
        invalid_data = {
            'rho0': -0.5,    # 负密度
            'Us': 0.5,       # 冲击波速度
            'Up': 1.0,       # 粒子速度大于冲击波速度
            'P': -10,        # 负压力
            'rho': 5.0,      # 压缩密度小于初始密度
            'V_V0': 1.2,     # 比体积比大于1
            'gamma': -0.5,   # 负的格吕奈森系数
            'T': 50          # 过低的温度
        }
        errors = validate_physical合理性(invalid_data, "测试材料")
        self.assertGreater(len(errors), 0, "无效参数未被识别")
        logger.debug(f"无效参数测试发现的错误: {errors}")

class TestShockCalculations(unittest.TestCase):
    """测试冲击波参数计算函数"""
    
    def test_basic_calculation(self):
        """测试基本的冲击波参数计算"""
        # 已知的铜的参数
        U_s = 5.0    # km/s
        u_p = 1.0    # km/s
        rho0 = 8.96  # g/cm³
        
        try:
            P, V, rho, V_V0, T = calculate_shock_parameters(U_s, u_p, rho0)
            
            # 验证计算结果
            self.assertAlmostEqual(P, 44.8, places=2, msg="压力计算错误")
            self.assertAlmostEqual(V, 0.089, places=3, msg="比体积计算错误")
            self.assertAlmostEqual(rho, 11.2, places=1, msg="压缩密度计算错误")
            self.assertAlmostEqual(V_V0, 0.8, places=2, msg="比体积比计算错误")
        except Exception as e:
            self.fail(f"计算过程抛出异常: {str(e)}")
    
    def test_invalid_inputs(self):
        """测试无效输入是否被正确处理"""
        # 冲击波速度小于粒子速度（物理上不合理）
        with self.assertRaises(ValueError):
            calculate_shock_parameters(1.0, 2.0, 8.96)
        
        # 负的初始密度
        with self.assertRaises(ValueError):
            calculate_shock_parameters(5.0, 1.0, -0.5)

class TestEquationSolving(unittest.TestCase):
    """测试方程组求解功能"""
    
    def test_simple_system(self):
        """测试简单方程组的求解"""
        # 创建简单的符号变量
        x, y = symbols('x y')
        
        # 简单方程组: x + y = 5; x - y = 1
        equations = [
            Eq(x + y, 5),
            Eq(x - y, 1)
        ]
        
        sym_vars = {'x': x, 'y': y}
        initial_guess = {x: 0, y: 0}
        
        solution = solve_numerically(equations, sym_vars, initial_guess)
        
        self.assertIsNotNone(solution, "求解失败")
        self.assertIn('x', solution)
        self.assertIn('y', solution)
        self.assertAlmostEqual(float(solution['x']), 3.0, delta=0.1)
        self.assertAlmostEqual(float(solution['y']), 2.0, delta=0.1)
    
    def test_shock_equations(self):
        """测试冲击波方程组的求解"""
        # 创建符号变量
        uf, Df, Pf = symbols('uf Df Pf')
        
        # 已知参数
        known_params = {
            'rh0f': 8.96,  # 铜的密度
            'w': 6.0,      # 飞片速度
            'C0f': 3.94,   # 体声速
            'Sf': 1.48     # Hugoniot参数
        }
        
        # 冲击波方程组
        equations = [
            Eq(uf + Df, known_params['w']),          # uf + Df = w
            Eq(Df, known_params['C0f'] + known_params['Sf'] * uf),  # Df = C0f + Sf·uf
            Eq(Pf, known_params['rh0f'] * Df * uf)   # Pf = rh0f·Df·uf
        ]
        
        sym_vars = {'uf': uf, 'Df': Df, 'Pf': Pf}
        
        # 生成初始猜测值
        initial_guess = generate_better_initial_guess(known_params, sym_vars.values())
        logger.debug(f"冲击波方程组测试的初始猜测值: {initial_guess}")
        
        # 求解方程组
        solution = solve_numerically(equations, sym_vars, initial_guess)
        
        self.assertIsNotNone(solution, "冲击波方程组求解失败")
        logger.debug(f"冲击波方程组的解: {solution}")
        
        # 验证解的物理合理性
        if solution:
            self.assertGreater(solution['Df'], solution['uf'], "冲击波速度应大于粒子速度")
            self.assertGreater(solution['Pf'], 0, "压力应为正值")

class TestMaterialParameterSets(unittest.TestCase):
    """测试典型材料参数组合"""
    
    def test_copper_parameters(self):
        """测试铜的典型参数"""
        params = {
            'rh0f': 8.96,    # g/cm³
            'C0f': 3.94,     # km/s
            'Sf': 1.48,
            'w': 6.0,        # km/s
        }
        
        uf, Df = symbols('uf Df')
        
        equations = [
            Eq(uf + Df, params['w']),
            Eq(Df, params['C0f'] + params['Sf'] * uf)
        ]
        
        solution = solve(equations, [uf, Df], dict=True)
        self.assertTrue(len(solution) > 0, "铜的参数求解失败")
        
        # 检查解的物理合理性
        for sol in solution:
            uf_val = float(sol[uf])
            Df_val = float(sol[Df])
            self.assertGreater(Df_val, uf_val, "铜的冲击波速度应大于粒子速度")
            
            # 计算压力并检查
            P = params['rh0f'] * Df_val * uf_val
            self.assertGreater(P, 0, "铜的压力应为正值")
            
            logger.debug(f"铜的求解结果: uf={uf_val:.4f}, Df={Df_val:.4f}, P={P:.4f}")
    
    def test_aluminum_parameters(self):
        """测试铝的典型参数"""
        params = {
            'rh0f': 2.7,     # g/cm³
            'C0f': 5.32,     # km/s
            'Sf': 1.34,
            'w': 8.0,        # km/s
        }
        
        uf, Df = symbols('uf Df')
        
        equations = [
            Eq(uf + Df, params['w']),
            Eq(Df, params['C0f'] + params['Sf'] * uf)
        ]
        
        solution = solve(equations, [uf, Df], dict=True)
        self.assertTrue(len(solution) > 0, "铝的参数求解失败")
        
        # 检查解的物理合理性
        for sol in solution:
            uf_val = float(sol[uf])
            Df_val = float(sol[Df])
            self.assertGreater(Df_val, uf_val, "铝的冲击波速度应大于粒子速度")
            
            # 计算压力并检查
            P = params['rh0f'] * Df_val * uf_val
            self.assertGreater(P, 0, "铝的压力应为正值")
            
            logger.debug(f"铝的求解结果: uf={uf_val:.4f}, Df={Df_val:.4f}, P={P:.4f}")

class TestFullWorkflow(unittest.TestCase):
    """测试完整的求解流程"""
    
    def test_complete_copper_calculation(self):
        """测试铜的完整计算流程"""
        try:
            # 飞片参数 (铜)
            rh0f = 8.96
            C0f = 3.94
            Sf = 1.48
            w = 6.0  # 飞片速度
            
            # 样品参数 (铝)
            rh0s = 2.7
            C0s = 5.32
            Ss = 1.34
            
            # 基板参数 (钛)
            rh0b = 4.5
            C0b = 6.1
            Sb = 1.22
            
            # 创建符号变量
            uf, Df, Pf = symbols('uf Df Pf')
            us, Ds, Ps = symbols('us Ds Ps')
            ub, Db, Pb = symbols('ub Db Pb')
            
            # 完整方程组
            equations = [
                # 飞片方程
                Eq(uf + Df, w),
                Eq(Df, C0f + Sf * uf),
                Eq(Pf, rh0f * Df * uf),
                
                # 样品方程
                Eq(Ds, C0s + Ss * us),
                Eq(Ps, rh0s * Ds * us),
                
                # 基板方程
                Eq(Db, C0b + Sb * ub),
                Eq(Pb, rh0b * Db * ub),
                
                # 压力连续性
                Eq(Pf, Ps),
                Eq(Ps, Pb)
            ]
            
            sym_vars = {
                'uf': uf, 'Df': Df, 'Pf': Pf,
                'us': us, 'Ds': Ds, 'Ps': Ps,
                'ub': ub, 'Db': Db, 'Pb': Pb
            }
            
            # 已知参数
            known_params = {
                'rh0f': rh0f, 'C0f': C0f, 'Sf': Sf, 'w': w,
                'rh0s': rh0s, 'C0s': C0s, 'Ss': Ss,
                'rh0b': rh0b, 'C0b': C0b, 'Sb': Sb
            }
            
            # 生成初始猜测值
            initial_guess = generate_better_initial_guess(known_params, sym_vars.values())
            logger.debug(f"完整流程的初始猜测值: {initial_guess}")
            
            # 求解方程组
            solution = solve_numerically(equations, sym_vars, initial_guess)
            
            self.assertIsNotNone(solution, "完整流程求解失败")
            logger.debug(f"完整流程的解: {solution}")
            
            # 检查所有物理约束
            if solution:
                # 冲击波速度 > 粒子速度
                self.assertGreater(solution['Df'], solution['uf'], "飞片冲击波速度应大于粒子速度")
                self.assertGreater(solution['Ds'], solution['us'], "样品冲击波速度应大于粒子速度")
                self.assertGreater(solution['Db'], solution['ub'], "基板冲击波速度应大于粒子速度")
                
                # 压力应为正值且连续
                self.assertGreater(solution['Pf'], 0, "飞片压力应为正值")
                self.assertAlmostEqual(solution['Pf'], solution['Ps'], delta=1.0, 
                                     msg="飞片与样品压力应相等")
                self.assertAlmostEqual(solution['Ps'], solution['Pb'], delta=1.0,
                                     msg="样品与基板压力应相等")
                
                logger.info("完整流程测试成功，所有物理约束均满足")
                
        except Exception as e:
            self.fail(f"完整流程测试失败: {str(e)}\n{traceback.format_exc()}")

if __name__ == '__main__':
    # 运行所有测试
    logger.info("开始冲击波参数计算程序测试...")
    unittest.main(verbosity=2)
    logger.info("测试完成")
