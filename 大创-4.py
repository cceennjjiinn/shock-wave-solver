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
from io import BytesIO
from PIL import Image
import itertools
import os

# Set Chinese font (retain Chinese display in interface)
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # Solve negative sign display problem

# Create SQLite engine - optimize connection pool configuration
sqlite_path = os.path.abspath('shock_wave_data.db')
sqlite_engine = create_engine(
    f'sqlite:///{sqlite_path}',
    pool_size=5,          # Maintain 5 persistent connections
    max_overflow=10,      # Create up to 10 additional temporary connections
    pool_recycle=3600     # Recycle connections after 1 hour
)

# SQLite performance optimization
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # Write-Ahead Logging
    cursor.execute('PRAGMA synchronous=NORMAL;')  # Synchronization mode
    cursor.execute('PRAGMA temp_store=MEMORY;')   # Temporary storage
    cursor.execute('PRAGMA cache_size=-20000;')   # Increase cache (20MB)
    cursor.close()

# Initialize database - add material field index
def init_database():
    try:
        with sqlite_engine.connect() as conn:
            if not conn.dialect.has_table(conn, 'shock_wave_all_data'):
                conn.execute(text("""
                    CREATE TABLE shock_wave_all_data (
                        id INTEGER PRIMARY KEY,
                        material TEXT,
                        rho0 REAL,       -- Initial density (g/cm³)
                        Us REAL,         -- Shock wave velocity (km/s)
                        Up REAL,         -- Particle velocity (km/s)
                        P REAL,          -- Shock pressure (GPa)
                        V REAL,          -- Specific volume (cm³/g)
                        rho REAL,        -- Compressed density (g/cm³)
                        V_V0 REAL,       -- Specific volume ratio (V/V0)
                        exp_method TEXT, -- Experimental method/data source
                        gamma REAL,      -- Grüneisen coefficient
                        T REAL,          -- Shock temperature (K)
                        INDEX idx_material (material)  -- New index to speed up queries
                    )
                """))
                conn.commit()
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")

init_database()

# Database operation functions - optimize query efficiency
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_materials():
    try:
        query = text("SELECT DISTINCT material FROM shock_wave_all_data")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['material'].tolist()
    except Exception as e:
        st.warning(f"Failed to get material list: {str(e)}")
        return []

def get_material_data(material_name, fields=None):
    """Query fields on demand to reduce data transmission"""
    try:
        if fields is None:
            fields = '*'  # Query all fields by default
        else:
            fields = ', '.join(fields)  # Specify fields as needed
        query = text(f"SELECT {fields} FROM shock_wave_all_data WHERE material = :material")
        with sqlite_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'material': material_name})
        return df
    except Exception as e:
        st.warning(f"Failed to get material data: {str(e)}")
        return pd.DataFrame()

def save_results_to_db(results, material_name="Copper"):
    if not results:
        st.warning("No data to save")
        return
        
    try:
        with sqlite_engine.begin() as conn:
            for result in results:
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
                    'gamma': result.get('gamma', 0),
                    'T': result.get('T', 0)
                }
                stmt = text("""
                    INSERT INTO shock_wave_all_data 
                    (material, rho0, Us, Up, P, V, rho, V_V0, exp_method, gamma, T) 
                    VALUES (:material, :rho0, :Us, :Up, :P, :V, :rho, :V_V0, :exp_method, :gamma, :T)
                """)
                conn.execute(stmt, data)
        st.success(f"Successfully saved {len(results)} calculation results to database")
    except Exception as e:
        st.error(f"Saving failed: {str(e)}")

def save_input_data_to_db(input_data, material_name, exp_method="manual_input"):
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
        st.success(f"Successfully saved input data to database (Material: {material_name})")
    except Exception as e:
        st.error(f"Failed to save input data: {str(e)}")

# Shock wave parameter calculation (including temperature calculation)
def calculate_shock_parameters(U_s, u_p, rho0, gamma=2.0, Cv=385, T0=300):
    """Calculate shock wave parameters based on Rankine-Hugoniot conservation relations"""
    # Momentum conservation: P = rho0 * U_s * u_p
    P = rho0 * U_s * u_p
    
    # Mass conservation to derive specific volume: V = (1/rho0) * (1 - u_p/U_s)
    V = (1 / rho0) * (1 - u_p / U_s)
    
    # Compressed density: rho = rho0 * U_s/(U_s - u_p)
    rho = rho0 * U_s / (U_s - u_p)
    
    # Specific volume ratio: V/V0 = 1 - u_p/U_s
    V_V0 = V * rho0  # Since V0 = 1/rho0, V/V0 = V * rho0
    
    # Temperature calculation (Mie-Grüneisen equation approximation)
    # Unit conversion: 1 GPa·cm³/g = 1e5 J/kg
    E_shock = 0.5 * P * (1/rho0 - V) * 1e6  # Shock internal energy (J/kg)
    # Based on simplified form of Mie-Grüneisen equation (applicable to weak shocks, ignoring volume correction terms)
    T = T0 + (E_shock) / (Cv * (1 + gamma/2))  # Shock temperature (K)
    
    return P, V, rho, V_V0, T

# Hugoniot relation fitting - optimize data preprocessing
def fit_hugoniot(df):
    # Filter physically invalid data
    df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
    if len(df) < 2:
        return 0, 0  # Return default values when data is insufficient
        
    U_s = df['Us'].values
    u_p = df['Up'].values
    coeffs = np.polyfit(u_p, U_s, 1)
    S = coeffs[0]    # Slope parameter
    C0 = coeffs[1]   # Intercept (zero-pressure sound speed)
    return C0, S

@st.cache_data(ttl=3600)  # Cache fitting results
def fit_material_data(df, material_name, material_type):
    if df is None or df.empty:
        st.warning(f"No data for {material_type.lower()} material '{material_name}'")
        return None
    
    # Filter outliers
    df = df[(df['Us'] > df['Up']) & (df['Us'] > 0) & (df['Up'] >= 0)]
    if len(df) < 2:
        st.warning(f"Insufficient valid data for {material_type.lower()} material '{material_name}', cannot fit")
        return None
    
    X = df['Up'].values.reshape(-1, 1)
    y = df['Us'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Fitting parameters
    C0 = model.intercept_    # Bulk sound speed (km/s)
    S = model.coef_[0]       # Hugoniot parameter S
    y_pred = model.predict(X)
    
    # Fitting error calculation
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # Root mean square error
    mae = np.mean(np.abs(y - y_pred))              # Mean absolute error
    
    st.info(f"{material_type} Material {material_name} Fitting result: Us = {C0:.4f} + {S:.4f}*Up")
    st.info(f"Fitting error: R² = {r2:.4f}, RMSE = {rmse:.4f} km/s, MAE = {mae:.4f} km/s")
    st.info(f"Average parameters: ρ₀ = {df['rho0'].mean():.4f} g/cm³, Average pressure = {df['P'].mean():.4f} GPa")
    
    return {
        "C0": C0, "S": S, "rho0": df['rho0'].mean(),
        "r2": r2, "rmse": rmse, "mae": mae
    }

# Error propagation calculation
def calculate_error(params, param_errors):
    """Calculate errors of output parameters (based on error propagation formula)"""
    rho0, Us, Up = params['rho0'], params['Us'], params['Up']
    rho0_err, Us_err, Up_err = param_errors['rho0'], param_errors['Us'], param_errors['Up']
    
    # Pressure error: P = rho0*Us*Up → sum of relative error squares
    P_rel_err = (rho0_err/rho0)**2 + (Us_err/Us)** 2 + (Up_err/Up)**2
    P_err = rho0*Us*Up * np.sqrt(P_rel_err)
    
    # Shock wave velocity error (simplified)
    Us_err = np.sqrt(Us_err**2 + (0.01*Us)** 2)  # Add 1% model error
    
    return {
        "P_err": P_err,
        "Us_err": Us_err,
        "Up_err": Up_err
    }

# Input function
def get_input_streamlit(label, var_name, key, default=None, unit="", desc=""):
    st.caption(f"{desc} | Unit: {unit}")
    input_type = st.radio(
        f"{label} input type",
        ["Single value", "Multiple values (comma-separated)", "Range (with optional step)"],
        key=f"{key}_type",
        horizontal=True
    )
    
    default_val = str(default) if default is not None else ""
    
    if input_type == "Single value":
        val = st.text_input(label, default_val, key=f"{key}_single")
        if val == "":
            return symbols(var_name)
        try:
            return [float(val)]
        except ValueError:
            st.error("Please enter a valid number")
            return None
    elif input_type == "Multiple values (comma-separated)":
        val = st.text_input(label, default_val, key=f"{key}_multi")
        if val == "":
            return symbols(var_name)
        try:
            return [float(x.strip()) for x in val.split(',')]
        except ValueError:
            st.error("Please enter valid comma-separated numbers")
            return None
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.text_input(f"{label} start value", default_val, key=f"{key}_start")
        with col2:
            end = st.text_input(f"{label} end value", "", key=f"{key}_end")
        with col3:
            step = st.text_input(f"{label} step (optional)", "0.5", key=f"{key}_step")
            
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
            st.error("Please enter valid range numbers")
            return None

# Numerical solver (replace symbolic solving for speed improvement)
def solve_numerically(eqs, sym_vars, initial_guess):
    """Solve system of equations using numerical method"""
    var_list = list(sym_vars.values())
    
    def residuals(x):
        """Calculate residuals: errors of the equations"""
        substitutions = {var_list[i]: x[i] for i in range(len(x))}
        return [float(abs(eq.subs(substitutions).evalf())) for eq in eqs]
    
    # Perform least squares optimization, adjust boundary ranges based on physical parameters
    result = least_squares(
        residuals,
        list(initial_guess.values()),
        bounds=([
            0.1,   # Lower bound for density (g/cm³)
            0.1,   # Lower bound for density (g/cm³)
            0.1,   # Lower bound for velocity (km/s)
            0.1,   # Lower bound for velocity (km/s)
            0.1,   # Lower bound for velocity (km/s)
            0.1,   # Lower bound for velocity (km/s)
            0.01,  # Lower bound for pressure (GPa)
            0.01,  # Lower bound for pressure (GPa)
            100    # Lower bound for temperature (K)
        ], [
            20,    # Upper bound for density (g/cm³)
            20,    # Upper bound for density (g/cm³)
            30,    # Upper bound for velocity (km/s)
            30,    # Upper bound for velocity (km/s)
            30,    # Upper bound for velocity (km/s)
            30,    # Upper bound for velocity (km/s)
            5000,  # Upper bound for pressure (GPa)
            5000,  # Upper bound for pressure (GPa)
            1e5    # Upper bound for temperature (K)
        ]),
        ftol=1e-6,
        max_nfev=1000
    )
    
    if result.success:
        return {str(var_list[i]): float(result.x[i]) for i in range(len(result.x))}
    return None

# Shock wave relation plots - modified to include material type in title
@st.cache_data(ttl=3600)  # Cache plot results
def generate_shock_plots(df, C0, S, material_name, material_type):
    # Sample data when the amount is large
    if len(df) > 1000:
        df = df.sample(1000)
        
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Add main title with material type and name
    fig.suptitle(f'{material_type} Material: {material_name} - Shock Wave Relationships', fontsize=16)
    
    # Us vs Up
    axs[0, 0].scatter(df['Up'], df['Us'], label='Experimental data')
    u_p_range = np.linspace(0, df['Up'].max()*1.1, 100)
    U_s_fit = C0 + S * u_p_range
    axs[0, 0].plot(u_p_range, U_s_fit, 'r-', label=f'Fit: Us = {C0:.2f} + {S:.2f}·Up')
    axs[0, 0].set_xlabel('Particle velocity Up (km/s)')
    axs[0, 0].set_ylabel('Shock wave velocity Us (km/s)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # P vs Up
    axs[0, 1].scatter(df['Up'], df['P'], label='Experimental data')
    # Use average density from data instead of hard-coded value
    rho0 = df['rho0'].mean() if not df.empty else 8.96
    P_range = rho0 * U_s_fit * u_p_range  # P = rho0 * Us * Up
    axs[0, 1].plot(u_p_range, P_range, 'r-', label='Theoretical curve: P = ρ0·Us·Up')
    axs[0, 1].set_xlabel('Particle velocity Up (km/s)')
    axs[0, 1].set_ylabel('Pressure P (GPa)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P vs V/V0
    axs[1, 0].scatter(df['V_V0'], df['P'], label='Experimental data')
    V_V0_range = 1 - u_p_range / U_s_fit  # V/V0 = 1 - Up/Us
    axs[1, 0].plot(V_V0_range, P_range, 'r-', label='Theoretical curve')
    axs[1, 0].set_xlabel('Specific volume ratio V/V0')
    axs[1, 0].set_ylabel('Pressure P (GPa)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # rho vs P
    axs[1, 1].scatter(df['P'], df['rho'], label='Experimental data')
    rho_range = rho0 * U_s_fit / (U_s_fit - u_p_range)  # rho = rho0·Us/(Us-Up)
    axs[1, 1].plot(P_range, rho_range, 'r-', label='Theoretical curve')
    axs[1, 1].set_xlabel('Pressure P (GPa)')
    axs[1, 1].set_ylabel('Density ρ (g/cm³)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def save_plot_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')  # Reduce resolution for speed
    buf.seek(0)
    return buf

# Helper function to display material plots
def display_material_plots(df, material_name, material_type):
    if not df.empty:
        with st.expander(f"View {material_type} Material {material_name} Shock Wave Plots", expanded=True):
            C0, S = fit_hugoniot(df)
            fig = generate_shock_plots(df, C0, S, material_name, material_type)
            st.pyplot(fig)
            buf = save_plot_to_bytes(fig)
            st.download_button(
                label=f"Download {material_type.lower()} {material_name} shock wave plots",
                data=buf,
                file_name=f"{material_type.lower()}_{material_name}_shock_relations.png",
                mime="image/png"
            )
    else:
        st.info(f"No available data to generate {material_type.lower()} material plots for {material_name}")

# Plotting function
@st.cache_data(ttl=3600)  # Cache plot results
def plot_results_streamlit(results):
    if not results:
        return None
        
    # Sample data when the amount is large
    if len(results) > 1000:
        results = results[:1000]
        
    fig = plt.figure(figsize=(18, 9))
    
    # Temperature-related data
    tf_values = [r.get('Tf', 0) for r in results]
    tb_values = [r.get('Tb', 0) for r in results]
    ts_values = [r.get('Ts', 0) for r in results]
    
    # Original data
    pf_values = [r.get('Pf', 0) for r in results]
    uf_values = [r.get('uf', 0) for r in results]
    df_values = [r.get('Df', 0) for r in results]
    rhf_values = [r.get('rhf', 0) for r in results]
    
    # 1. Pressure-particle velocity plot (with error bars)
    ax1 = fig.add_subplot(221)
    ax1.errorbar(uf_values, pf_values, 
                 yerr=[r.get('Pf_err', 0.1) for r in results],
                 xerr=[r.get('uf_err', 0.05) for r in results],
                 fmt='bo', ecolor='r', capsize=5, label='Flyer data')
    ax1.set_xlabel('Particle velocity Up (km/s)')
    ax1.set_ylabel('Shock pressure P (GPa)')
    ax1.set_title('Pressure-Particle Velocity Relationship (with error range)')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Temperature-pressure plot
    ax2 = fig.add_subplot(222)
    ax2.scatter(pf_values, tf_values, c='orange', label='Flyer temperature')
    ax2.set_xlabel('Shock pressure P (GPa)')
    ax2.set_ylabel('Shock temperature T (K)')
    ax2.set_title('Temperature-Pressure Relationship')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Shock wave velocity-particle velocity plot
    ax3 = fig.add_subplot(223)
    ax3.scatter(uf_values, df_values, c='blue', label='Flyer')
    ax3.set_xlabel('Particle velocity Up (km/s)')
    ax3.set_ylabel('Shock wave velocity Us (km/s)')
    ax3.set_title('Shock Wave Velocity-Particle Velocity Relationship')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Density-pressure plot
    ax4 = fig.add_subplot(224)
    ax4.scatter(pf_values, rhf_values, c='green', label='Flyer')
    ax4.set_xlabel('Shock pressure P (GPa)')
    ax4.set_ylabel('Compressed density (g/cm³)')
    ax4.set_title('Density-Pressure Relationship')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

# Page functions
def home_page():
    st.title("Shock Wave Parameter Calculation and Analysis System")
    st.info("""
    System Core Model Description:
    1. Based on Rankine-Hugoniot conservation equations (mass, momentum, energy conservation)
    2. Assumptions: Planar shock wave, steady propagation, initial pressure ignored
    3. Unit system: Density (g/cm³), velocity (km/s), pressure (GPa)
    """)
    st.write("Select operation mode:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use Database Data"):
            st.session_state.page = "database_mode"
    with col2:
        if st.button("Manual Parameter Input"):
            st.session_state.page = "manual_mode"

def database_mode_page():
    st.title("Database Mode")
    st.write("Load material data from database, fit parameters based on Hugoniot relations and solve")
    
    materials = get_all_materials()
    if not materials:
        st.error("No available materials in database")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        flyer_material = st.selectbox("Flyer material", materials, key="flyer_material")
    with col2:
        base_material = st.selectbox("Base material", materials, key="base_material")
    with col3:
        sample_material = st.selectbox("Sample material", materials, key="sample_material")
    
    # Query fields on demand to reduce data transmission
    flyer_df = get_material_data(flyer_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho'])
    base_df = get_material_data(base_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho'])
    sample_df = get_material_data(sample_material, fields=['Us', 'Up', 'rho0', 'P', 'V_V0', 'rho'])
    
    # Fit data for each material type with clear labeling
    with st.spinner(f"Fitting flyer material {flyer_material} data..."):
        flyer_fit = fit_material_data(flyer_df, flyer_material, "Flyer")
    with st.spinner(f"Fitting base material {base_material} data..."):
        base_fit = fit_material_data(base_df, base_material, "Base")
    with st.spinner(f"Fitting sample material {sample_material} data..."):
        sample_fit = fit_material_data(sample_df, sample_material, "Sample")
    
    # Shock wave parameter analysis section with separate plots for each material
    st.subheader("Shock Wave Parameter Analysis (Hugoniot Relation)")
    st.caption("""
    Analysis based on linear Hugoniot relation Us = C0 + S·Up, where:
    - C0: Bulk sound speed of material (sound speed at zero pressure, km/s)
    - S: Hugoniot parameter (describing the rate of change of shock wave velocity with particle velocity, dimensionless)
    - Application note: Deviations may occur at high pressures (e.g., >100 GPa), phase transitions or nonlinear terms need to be considered
    """)
    
    # Display separate plots for each material type
    display_material_plots(flyer_df, flyer_material, "Flyer")
    display_material_plots(base_df, base_material, "Base")
    display_material_plots(sample_df, sample_material, "Sample")
    
    default_params = {"f": flyer_fit, "b": base_fit, "s": sample_fit}
    # Parameter definitions
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    # Flyer parameters
    with st.expander(f"{flyer_material} Flyer Parameters", expanded=True):
        cols = st.columns(3)
        var_descs = {
            "rh0f": "Initial density",
            "rhf": "Compressed density",
            "Df": "Shock wave velocity (corresponding to Us)",
            "C0f": "Bulk sound speed (Hugoniot fit)",
            "Sf": "Hugoniot parameter S (dimensionless)",
            "E0f": "Initial internal energy density",
            "Ef": "Compressed internal energy density",
            "uf": "Particle velocity (corresponding to Up)",
            "w": "Initial impact velocity of flyer",
            "Pf": "Shock pressure",
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
                    default_val = 2.0  # Default Grüneisen coefficient
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
    
    # Base parameters
    with st.expander(f"{base_material} Base Parameters_material} Base Parameters", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
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
                    default_val = 2.0  # Default Grüneisen coefficient
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
                    desc="Initial density" if var == "rh0b" else
                         "Compressed density" if var == "rhb" else
                         "Shock wave velocity" if var == "Db" else
                         "Bulk sound speed" if var == "C0b" else
                         "Hugoniot parameter" if var == "Sb" else
                         "Initial internal energy density" if var == "E0b" else
                         "Compressed internal energy density" if var == "Eb" else
                         "Particle velocity" if var == "ub" else
                         "Shock pressure" if var == "Pb" else
                         "Grüneisen coefficient" if var == "gammab" else
                         "Shock temperature"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # Sample parameters
    with st.expander(f"{sample_material} Sample Parameters", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
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
                    default_val = 2.0  # Default Grüneisen coefficient
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
                    desc="Initial density" if var == "rh0s" else
                         "Compressed density" if var == "rhs" else
                         "Shock wave velocity" if var == "Ds" else
                         "Bulk sound speed" if var == "C0s" else
                         "Hugoniot parameter" if var == "Ss" else
                         "Initial internal energy density" if var == "E0s" else
                         "Compressed internal energy density" if var == "Es" else
                         "Particle velocity" if var == "us" else
                         "Shock pressure" if var == "Ps" else
                         "Grüneisen coefficient" if var == "gammas" else
                         "Shock temperature"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # Parameter combination limit
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "Maximum number of parameter combinations (too many will affect speed)", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("Start solving"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"Invalid input for {var}, please check")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # Truncate excessive combinations
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"Too many parameter combinations ({len(combinations)}), truncated to {max_combinations} to improve speed")
            combinations = combinations[:max_combinations]
        
        results = []
        progress_bar = st.progress(0)
        total = len(combinations)
        count = 0
        
        for combo in combinations:
            count += 1
            # Update progress bar every 10 times to reduce UI overhead
            if count % 10 == 0 or count == total:
                progress_bar.progress(count / total)
                
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # System of equations
            eqs = [
                # Flyer mass conservation: rho0f·Df = rhf·(Df - uf)
                Eq(sym_vars['rh0f']*sym_vars['Df'] - sym_vars['rhf']*(sym_vars['Df'] - sym_vars['uf']), 0),
                # Base mass conservation: rho0b·Db = rhb·(Db - ub)
                Eq(sym_vars['rh0b']*sym_vars['Db'] - sym_vars['rhb']*(sym_vars['Db'] - sym_vars['ub']), 0),
                # Flyer momentum conservation: Pf = rho0f·Df·(w - uf)
                Eq(sym_vars['Pf'] - sym_vars['rh0f']*sym_vars['Df']*(sym_vars['w'] - sym_vars['uf']), 0),
                # Base momentum conservation: Pb = rho0b·Db·ub
                Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*sym_vars['ub'], 0),
                # Flyer energy conservation: Ef = E0f + 0.5·Pf·(1/rho0f - 1/rhf)
                Eq(sym_vars['Ef'] - sym_vars['E0f'] - 0.5*sym_vars['Pf']*(1/sym_vars['rh0f'] - 1/sym_vars['rhf']), 0),
                # Base energy conservation: Eb = E0b + 0.5·Pb·(1/rho0b - 1/rhb)
                Eq(sym_vars['Eb'] - sym_vars['E0b'] - 0.5*sym_vars['Pb']*(1/sym_vars['rh0b'] - 1/sym_vars['rhb']), 0),
                # Flyer Hugoniot relation: Df = C0f + Sf·(w - uf)
                Eq(sym_vars['Df'] - sym_vars['C0f'] - sym_vars['Sf']*(sym_vars['w'] - sym_vars['uf']), 0),
                # Base Hugoniot relation: Db = C0b + Sb·ub
                Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*sym_vars['ub'], 0),
                # Interface pressure continuity: Pf = Pb
                Eq(sym_vars['Pf'] - sym_vars['Pb'], 0),
                # Interface particle velocity continuity: uf = ub
                Eq(sym_vars['uf'] - sym_vars['ub'], 0)
            ]
            
            try:
                # Check if sample and base are the same material
                cond = all([
                    current_subs.get(sym_vars['rh0s'], sym_vars['rh0s']) == current_subs.get(sym_vars['rh0b'], sym_vars['rh0b']),
                    current_subs.get(sym_vars['C0b'], sym_vars['C0b']) == current_subs.get(sym_vars['C0s'], sym_vars['C0s']),
                    current_subs.get(sym_vars['Sb'], sym_vars['Sb']) == current_subs.get(sym_vars['Ss'], sym_vars['Ss']),
                    current_subs.get(sym_vars['E0b'], sym_vars['E0b']) == current_subs.get(sym_vars['E0s'], sym_vars['E0s'])
                ])
            except TypeError:
                cond = False
                
            if cond:
                # Sample and base are the same material: parameters are consistent with base
                eqs += [
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # Pressure continuity
                    Eq(sym_vars['ub'] - sym_vars['us'], 0),  # Velocity continuity
                    Eq(sym_vars['rhb'] - sym_vars['rhs'], 0), # Density continuity
                    Eq(sym_vars['Db'] - sym_vars['Ds'], 0),  # Shock wave velocity continuity
                    # Sample energy conservation
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # Temperature parameter continuity
                    Eq(sym_vars['Tb'] - sym_vars['Ts'], 0),
                    Eq(sym_vars['gammab'] - sym_vars['gammas'], 0)
                ]
            else:
                # Sample and base are different materials: calculate separately
                eqs += [
                    # Sample mass conservation
                    Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    # Base-sample interface momentum conservation
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    # Sample momentum conservation
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    # Sample energy conservation
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # Sample Hugoniot relation
                    Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                    # Base-sample interface Hugoniot relation
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # Pressure continuity
                    Eq(sym_vars['ub'] - sym_vars['us'], 0)   # Velocity continuity
                ]
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # Build initial guess values (based on physically reasonable ranges)
                initial_guess = {}
                for var in remaining_vars:
                    var_str = str(var)
                    if var_str.startswith(('rh0', 'rh')):  # Density
                        initial_guess[var] = 8.0
                    elif var_str.startswith(('D', 'C0', 'u', 'w')):  # Velocity
                        initial_guess[var] = 5.0
                    elif var_str.startswith(('P', 'E')):  # Pressure/energy
                        initial_guess[var] = 100.0
                    elif var_str.startswith('gamma'):  # Grüneisen coefficient
                        initial_guess[var] = 2.0
                    elif var_str.startswith('T'):  # Temperature
                        initial_guess[var] = 3000.0
                    else:  # Other parameters
                        initial_guess[var] = 1.0
                
                # Solve using numerical method
                solution = solve_numerically(substituted_eqs, {v:v for v in remaining_vars}, initial_guess)
                
                if solution:
                    record = solution.copy()
                    # Add known parameters
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
                st.warning(f"Solution error: {str(e)} (may be caused by nonlinear effects at high pressure, please check parameter range)")
        
        if results:
            st.success(f"Solution completed, {len(results)} solutions found (results are based on ideal shock wave assumptions, practical application requires verification)")
            
            st.subheader("Result data (Units: rho=g/cm³, D=km/s, u=km/s, P=GPa, T=K)")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download result data",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            st.subheader("Result visualization")
            fig = plot_results_streamlit(results)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="Download charts",
                    data=buf2,
                    file_name="analysis_with_temp_error.png",
                    mime="image/png"
                )
            
            if st.button("Save results to database"):
                save_results_to_db(results, sample_material)
        else:
            st.warning("No valid solutions found (please check if parameters conform to physical ranges, e.g., shock wave velocity > particle velocity)")
    
    if st.button("Return to homepage"):
        st.session_state.page = "home"

def manual_mode_page():
    st.title("Manual Input Mode")
    st.write("Solve by manually inputting parameters, suitable for scenarios without database data")
    
    # Material parameter input
    col1, col2 = st.columns(2)
    with col1:
        material_name = st.text_input("Material name", value="Copper", help="Enter material name, e.g., Copper, Aluminum, etc.")
        gamma = st.number_input("Grüneisen coefficient Γ", value=2.0, min_value=0.1, help="Approximately 2.0 for copper, 2.13 for aluminum")
    with col2:
        exp_method = st.text_input("Experimental method/data source", value="manual_input", help="Record data source, e.g., experimental equipment, literature, etc.")
        Cv = st.number_input("Specific heat at constant volume Cv (J/(kg·K))", value=385, help="Approximately 385 for copper, 900 for aluminum")
    
    # Quick calculation of shock wave parameters
    st.subheader("Quick Calculation of Shock Wave Parameters")
    st.caption("""
    Based on Rankine-Hugoniot conservation equations, applicable to ideal planar shock waves:
    - Formula: P = ρ0·Us·Up, ρ = ρ0·Us/(Us-Up), V/V0 = 1 - Up/Us
    - Input requirement: Us > Up (shock wave velocity must be greater than particle velocity)
    - Unit: ρ0(g/cm³), Us(km/s), Up(km/s) → Output P(GPa)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        U_s = st.number_input("Shock wave velocity Us (km/s)", min_value=0.01, value=5.0, help="Must be greater than particle velocity Up")
        Us_err = st.number_input("Us error (km/s)", 0.1, help="Measurement error")
    with col2:
        u_p = st.number_input("Particle velocity Up (km/s)", min_value=0.0, value=1.0, help="Must be less than shock wave velocity Us")
        Up_err = st.number_input("Up error (km/s)", 0.05, help="Measurement error")
    with col3:
        rho0 = st.number_input("Initial density ρ0 (g/cm³)", min_value=0.01, value=8.96, help="e.g., initial density of copper is approximately 8.96 g/cm³")
        rho0_err = st.number_input("ρ0 error (g/cm³)", 0.02, help="Measurement error")
    
    # Store calculation results for saving
    calculation_result = None
    
    if st.button("Calculate shock wave parameters"):
        if U_s <= u_p:
            st.error("Physical parameter error: Shock wave velocity Us must be greater than particle velocity Up")
        else:
            P, V, rho, V_V0, T = calculate_shock_parameters(
                U_s, u_p, rho0, gamma, Cv
            )
            
            # Calculate error
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
            Calculation results (based on ideal shock wave assumptions):
            - Shock pressure P = {P:.2f} ± {error_params['P_err']:.2f} GPa
            - Shock temperature T = {T:.0f} K
            - Compressed density ρ = {rho:.2f} g/cm³
            - Specific volume ratio V/V0 = {V_V0:.4f}
            """)
    
    # Save input data to database
    if calculation_result:
        if st.button("Save input data to database"):
            save_input_data_to_db(calculation_result, material_name, exp_method)
    
    # Parameter input
    variables = {
        "f": ["rh0f", "rhf", "Df", "C0f", "Sf", "E0f", "Ef", "uf", "w", "Pf", "gammaf", "Tf"],
        "b": ["rh0b", "rhb", "Db", "C0b", "Sb", "E0b", "Eb", "ub", "Pb", "gammab", "Tb"],
        "s": ["rh0s", "rhs", "Ds", "C0s", "Ss", "E0s", "Es", "us", "Ps", "gammas", "Ts"]
    }
    
    input_params = {}
    sym_vars = {}
    
    with st.expander("Flyer Parameters", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["f"]):
            with cols[i % 3]:
                default_val = 2.0 if var == "gammaf" else None
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
                         "Flyer compressed internal energy density" if var == "Ef" else
                         "Flyer particle velocity" if var == "uf" else
                         "Flyer initial impact velocity" if var == "w" else
                         "Flyer shock pressure" if var == "Pf" else
                         "Flyer Grüneisen coefficient" if var == "gammaf" else
                         "Flyer shock temperature"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # Base parameter input
    with st.expander("Base Parameters", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["b"]):
            with cols[i % 3]:
                default_val = 2.0 if var == "gammab" else None
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
                    desc="Base initial density" if var == "rh0b" else
                         "Base compressed density" if var == "rhb" else
                         "Base shock wave velocity" if var == "Db" else
                         "Base bulk sound speed" if var == "C0b" else
                         "Base Hugoniot parameter S" if var == "Sb" else
                         "Base initial internal energy density" if var == "E0b" else
                         "Base compressed internal energy density" if var == "Eb" else
                         "Base particle velocity" if var == "ub" else
                         "Base shock pressure" if var == "Pb" else
                         "Base Grüneisen coefficient" if var == "gammab" else
                         "Base shock temperature"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # Sample parameter input
    with st.expander("Sample Parameters", expanded=True):
        cols = st.columns(3)
        for i, var in enumerate(variables["s"]):
            with cols[i % 3]:
                default_val = 2.0 if var == "gammas" else None
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
                         "Sample compressed internal energy density" if var == "Es" else
                         "Sample particle velocity" if var == "us" else
                         "Sample shock pressure" if var == "Ps" else
                         "Sample Grüneisen coefficient" if var == "gammas" else
                         "Sample shock temperature"
                )
                input_params[var] = val
                sym_vars[var] = symbols(var)
    
    # Parameter combination limit
    range_params = {k: v for k, v in input_params.items() if isinstance(v, list)}
    total_combinations = 1
    for v in range_params.values():
        total_combinations *= len(v)
    
    max_combinations = st.slider(
        "Maximum number of parameter combinations (too many will affect speed)", 
        min_value=10, 
        max_value=1000, 
        value=min(100, total_combinations)
    )
    
    if st.button("Start solving equations"):
        valid = True
        for var, val in input_params.items():
            if val is None:
                valid = False
                st.error(f"Invalid input for {var}, please check")
        
        if not valid:
            return
            
        combinations = itertools.product(*[[(k, val) for val in v] for k, v in range_params.items()])
        
        # Truncate excessive combinations
        combinations = list(combinations)
        if len(combinations) > max_combinations:
            st.warning(f"Too many parameter combinations ({len(combinations)}), truncated to {max_combinations} to improve speed")
            combinations = combinations[:max_combinations]
        
        results = []
        progress_bar = st.progress(0)
        total = len(combinations)
        count = 0
        
        for combo in combinations:
            count += 1
            # Update progress bar every 10 times
            if count % 10 == 0 or count == total:
                progress_bar.progress(count / total)
                
            current_subs = {sym_vars[k]: v for k, v in combo}
            
            # Check physical rationality
            try:
                if current_subs.get(sym_vars['Df'], 0) <= current_subs.get(sym_vars['uf'], 0):
                    st.warning("Flyer parameter error: Df (shock wave velocity) must be greater than uf (particle velocity)")
                    continue
                if current_subs.get(sym_vars['Db'], 0) <= current_subs.get(sym_vars['ub'], 0):
                    st.warning("Base parameter error: Db (shock wave velocity) must be greater than ub (particle velocity)")
                    continue
            except:
                pass
            
            # System of equations definition
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
                # Check if sample and base are the same material
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
                    # Sample mass conservation
                    Eq(sym_vars['rh0s']*sym_vars['Ds'] - sym_vars['rhb']*(sym_vars['Ds'] - sym_vars['us']), 0),
                    # Base-sample interface momentum conservation
                    Eq(sym_vars['Pb'] - sym_vars['rh0b']*sym_vars['Db']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    # Sample momentum conservation
                    Eq(sym_vars['Ps'] - sym_vars['rh0s']*sym_vars['Ds']*sym_vars['us'], 0),
                    # Sample energy conservation
                    Eq(sym_vars['Es'] - sym_vars['E0s'] - 0.5*sym_vars['Ps']*(1/sym_vars['rh0s'] - 1/sym_vars['rhs']), 0),
                    # Sample Hugoniot relation
                    Eq(sym_vars['Ds'] - sym_vars['C0s'] - sym_vars['Ss']*sym_vars['us'], 0),
                    # Base-sample interface Hugoniot relation
                    Eq(sym_vars['Db'] - sym_vars['C0b'] - sym_vars['Sb']*(2*sym_vars['ub'] - sym_vars['us']), 0),
                    Eq(sym_vars['Pb'] - sym_vars['Ps'], 0),  # Pressure continuity
                    Eq(sym_vars['ub'] - sym_vars['us'], 0)   # Velocity continuity
                ]
            
            substituted_eqs = [eq.subs(current_subs) for eq in eqs]
            remaining_vars = list(set().union(*[eq.free_symbols for eq in substituted_eqs]))
            
            if not remaining_vars:
                continue
                
            try:
                # Build initial guess values
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
                
                # Numerical solution
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
                st.warning(f"Solution error: {str(e)} (may be due to parameter range exceeding model applicable conditions)")
        
        if results:
            st.success(f"Solution completed, {len(results)} solutions found")
            
            st.subheader("Result data")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download result data",
                data=csv,
                file_name="solver_results.csv",
                mime="text/csv",
            )
            
            st.subheader("Result visualization")
            fig = plot_results_streamlit(results)
            if fig:
                st.pyplot(fig)
                buf2 = BytesIO()
                fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="Download charts",
                    data=buf2,
                    file_name="analysis_with_temp_error.png",
                    mime="image/png"
                )
            
            if st.button("Save calculation results to database"):
                save_results_to_db(results, material_name)
        else:
            st.warning("No valid solutions found (please check if parameters conform to physical laws, e.g., shock wave velocity > particle velocity)")
    
    if st.button("Return to homepage"):
        st.session_state.page = "home"

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    st.set_page_config(
        page_title="Shock Wave Parameter Calculation and Analysis System",
        page_icon="✨",
        layout="wide"
    )
    
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "database_mode":
        database_mode_page()
    elif st.session_state.page == "manual_mode":
        manual_mode_page()

if __name__ == "__main__":
    main()
