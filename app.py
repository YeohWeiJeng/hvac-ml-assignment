import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN

# ==========================================
# 1. SHARED ENVIRONMENT DEFINITION
# ==========================================

class HVACEnv(gym.Env):
    def __init__(self):
        super(HVACEnv, self).__init__()
        # --- 1. State & Action Spaces ---
        self.action_space = spaces.Dict({
            "adjust_setpoint": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "change_fan_speed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "modulate_ventilation": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "dehumidification": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "adjust_preconditioning": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "switch_mode": spaces.Discrete(4) 
        })

        self.observation_space = spaces.Dict({
            "indoor_temperature": spaces.Box(low=15.0, high=35.0, shape=(1,), dtype=np.float32),
            "outdoor_temperature": spaces.Box(low=-15.0, high=45.0, shape=(1,), dtype=np.float32),
            "humidity": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "co2_level": spaces.Box(low=300.0, high=2000.0, shape=(1,), dtype=np.float32),
            "occupancy": spaces.Discrete(100),
            "time_of_day": spaces.Discrete(24),
            "power_consumption": spaces.Box(low=0.0, high=10000.0, shape=(1,), dtype=np.float32),
            "water_flow_rate": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "hot_water_tank_temp": spaces.Box(low=20.0, high=90.0, shape=(1,), dtype=np.float32),
            "cooler_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        })

        self.weights = {'alpha': 1.0, 'beta': 10.0, 'gamma': 2.0}
        self.TARGET_TEMP = 24.0
        self.state = {}
        self.current_step = 0
        self.setpoint_temperature = 24.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = {
            "indoor_temperature": np.array([np.random.uniform(24.0, 27.0)], dtype=np.float32),
            "outdoor_temperature": np.array([28.0], dtype=np.float32),
            "humidity": np.array([np.random.uniform(60.0, 80.0)], dtype=np.float32),
            "co2_level": np.array([400.0], dtype=np.float32),
            "occupancy": 0,
            "time_of_day": 8,
            "power_consumption": np.array([0.0], dtype=np.float32),
            "water_flow_rate": np.array([0.0], dtype=np.float32),
            "hot_water_tank_temp": np.array([60.0], dtype=np.float32),
            "cooler_level": np.array([0.0], dtype=np.float32)
        }
        return self.state, {}

    def step(self, action):
        setpoint_adj = action["adjust_setpoint"][0]
        fan_speed = action["change_fan_speed"][0]
        vent_damper = action["modulate_ventilation"][0]
        dehumid_power = action["dehumidification"][0]
        precond_shift = action["adjust_preconditioning"][0]
        mode = action["switch_mode"]

        self._simulate_physics(setpoint_adj, fan_speed, vent_damper, dehumid_power, precond_shift, mode)
        reward = self._calculate_reward()

        terminated = False
        indoor_t = self.state["indoor_temperature"][0]
        if indoor_t > 35.0 or indoor_t < 16.0:
            terminated = True 

        return self.state, float(reward), terminated, False, {}

    def _simulate_physics(self, setpoint_adj, fan_speed, vent, dehumid, precond, mode):
        self.state["time_of_day"] = (8 + (self.current_step // 60)) % 24
        hour = self.state["time_of_day"]
        
        base_outdoor = 30.0
        self.state["outdoor_temperature"][0] = base_outdoor + 5.0 * np.sin((hour - 8) * np.pi / 12)

        if 9 <= hour <= 17:
            self.state["occupancy"] = np.random.randint(5, 20)
        else:
            self.state["occupancy"] = 0

        self.setpoint_temperature = np.clip(self.setpoint_temperature + setpoint_adj * 0.5, 18.0, 30.0)

        if mode == 1: # COOL
            target_flow = fan_speed * 100.0
        else:
            target_flow = 0.0

        current_flow = self.state["water_flow_rate"][0]
        self.state["water_flow_rate"][0] = current_flow + (target_flow - current_flow) * 0.2
        self.state["cooler_level"][0] = self.state["water_flow_rate"][0]

        T_room = self.state["indoor_temperature"][0]
        T_out = self.state["outdoor_temperature"][0]
        RH = self.state["humidity"][0]
        occ = self.state["occupancy"]

        humidity_penalty = max(0.0, (RH - 50.0) / 100.0)
        cooling_efficiency = 1.0 - humidity_penalty
        boost = max(0.0, precond) * 0.2
        cooling_power_kw = (self.state["water_flow_rate"][0] / 100.0) * 5.0 * (cooling_efficiency + boost)

        heat_leak = (T_out - T_room) * 0.1
        internal_heat = (occ * 0.1)
        vent_heat = vent * (T_out - T_room) * 0.2

        dt_temp = heat_leak + internal_heat + vent_heat - cooling_power_kw
        self.state["indoor_temperature"][0] = T_room + dt_temp * 0.1

        humid_gain = (vent * 0.5) + (occ * 0.1)
        humid_loss = max(0, dehumid) * 2.0
        self.state["humidity"][0] = np.clip(RH + humid_gain - humid_loss, 20.0, 100.0)

        co2_gain = occ * 5.0
        co2_loss = vent * 10.0
        self.state["co2_level"][0] = np.clip(self.state["co2_level"][0] + co2_gain - co2_loss, 300.0, 2000.0)

        power_kw = (cooling_power_kw) + (fan_speed * 0.5) + (abs(dehumid) * 0.8) + (vent * 0.2)
        self.state["power_consumption"][0] = power_kw * 1000.0
        self.current_step += 1

    def _calculate_reward(self):
        T_room = self.state["indoor_temperature"][0]
        E_consumed = self.state["power_consumption"][0] / 1000.0
        term_energy = self.weights['alpha'] * E_consumed
        if 23.0 <= T_room <= 25.0:
            term_comfort = self.weights['beta'] * 1.0
        else:
            term_comfort = 0.0
        term_deviation = self.weights['gamma'] * abs(T_room - self.TARGET_TEMP)
        return float(-term_energy + term_comfort - term_deviation)

# ==========================================
# 2. DQN SPECIFIC CLASSES (Standard)
# ==========================================

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_map = {
            0: {"adjust_setpoint": np.array([0.0], dtype=np.float32), "change_fan_speed": np.array([0.0], dtype=np.float32), "modulate_ventilation": np.array([0.0], dtype=np.float32), "dehumidification": np.array([0.0], dtype=np.float32), "adjust_preconditioning": np.array([0.0], dtype=np.float32), "switch_mode": 0},
            1: {"adjust_setpoint": np.array([-0.5], dtype=np.float32), "change_fan_speed": np.array([0.5], dtype=np.float32), "modulate_ventilation": np.array([0.1], dtype=np.float32), "dehumidification": np.array([0.0], dtype=np.float32), "adjust_preconditioning": np.array([0.0], dtype=np.float32), "switch_mode": 1},
            2: {"adjust_setpoint": np.array([-1.0], dtype=np.float32), "change_fan_speed": np.array([1.0], dtype=np.float32), "modulate_ventilation": np.array([0.1], dtype=np.float32), "dehumidification": np.array([0.5], dtype=np.float32), "adjust_preconditioning": np.array([0.0], dtype=np.float32), "switch_mode": 1},
            3: {"adjust_setpoint": np.array([0.0], dtype=np.float32), "change_fan_speed": np.array([0.8], dtype=np.float32), "modulate_ventilation": np.array([1.0], dtype=np.float32), "dehumidification": np.array([0.0], dtype=np.float32), "adjust_preconditioning": np.array([0.0], dtype=np.float32), "switch_mode": 3}
        }
        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def action(self, act):
        if hasattr(act, "item"): act = int(act.item())
        selection = self.action_map[act]
        return {k: np.array(v, dtype=np.float32) if k != "switch_mode" else v for k, v in selection.items()}

# ==========================================
# 3. DDPG SPECIFIC CLASSES (Continuous)
# ==========================================

class ContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 6 continuous values
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def action(self, act):
        # Map 6 floats to dictionary
        # Act[5] maps to mode (Discrete 0-3)
        mode_val = act[5]
        if mode_val < -0.5: mode = 0
        elif mode_val < 0.0: mode = 1
        elif mode_val < 0.5: mode = 2
        else: mode = 3
        
        return {
            "adjust_setpoint": np.array([act[0]], dtype=np.float32),
            "change_fan_speed": np.array([np.clip(act[1], 0, 1)], dtype=np.float32),
            "modulate_ventilation": np.array([np.clip(act[2], 0, 1)], dtype=np.float32),
            "dehumidification": np.array([act[3]], dtype=np.float32),
            "adjust_preconditioning": np.array([act[4]], dtype=np.float32),
            "switch_mode": mode
        }

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

def flatten_state(obs):
    return np.concatenate([v.flatten() if isinstance(v, np.ndarray) else [v] for v in obs.values()])

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="HVAC Model Visualizer", layout="wide")
st.title("🌡️ HVAC Unified Model Visualizer (Standard DQN & DDPG)")

# --- SIDEBAR ---
st.sidebar.header("1. Upload Model")
uploaded_file = st.sidebar.file_uploader("Upload .zip (DQN) or .pth (DDPG)", type=["zip", "pth"])
st.sidebar.header("2. Simulation Settings")
n_steps = st.sidebar.slider("Simulation Steps", min_value=100, max_value=2000, value=600)
deterministic = st.sidebar.checkbox("Deterministic (Best Action)", value=True)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    
    # Identify Model Type
    file_name = uploaded_file.name
    model_type = "DQN" if file_name.endswith(".zip") else "DDPG"
    
    with open(f"temp_model.{'zip' if model_type == 'DQN' else 'pth'}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    model = None
    
    if model_type == "DQN":
        try:
            # Standard DQN Load
            model = DQN.load("temp_model.zip")
            st.sidebar.success(f"✅ Standard DQN Loaded ({file_name})")
        except Exception as e:
            st.sidebar.error(f"Error loading DQN: {e}")
    else:
        # Load DDPG Actor
        try:
            temp_env = HVACEnv()
            state_dim = sum(np.prod(v.shape) if isinstance(v, np.ndarray) else 1 for v in temp_env.reset()[0].values())
            action_dim = 6 # Continuous wrapper dimension
            
            model = Actor(state_dim, action_dim)
            model.load_state_dict(torch.load("temp_model.pth"))
            model.eval()
            st.sidebar.success(f"✅ DDPG Actor Loaded ({file_name})")
        except Exception as e:
            st.sidebar.error(f"Error loading DDPG: {e}")

    if 'simulation_logs' not in st.session_state:
        st.session_state.simulation_logs = None

    if st.sidebar.button("Run Simulation"):
        with st.spinner("Simulating..."):
            
            # Setup Correct Environment Wrapper
            raw_env = HVACEnv()
            if model_type == "DQN":
                env = DiscreteActionWrapper(raw_env)
            else:
                env = ContinuousActionWrapper(raw_env)
            
            obs, _ = env.reset()
            logs = []
            
            steps_comfort = 0
            steps_too_hot = 0
            steps_too_cold = 0
            
            for step in range(n_steps):
                
                # PREDICTION LOGIC
                if model_type == "DQN":
                    action, _ = model.predict(obs, deterministic=deterministic)
                    display_action = int(action) 
                else:
                    # DDPG Prediction
                    state_flat = flatten_state(obs)
                    state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)
                    with torch.no_grad():
                        action_raw = model(state_tensor).squeeze(0).numpy()
                    
                    if not deterministic:
                        action_raw += np.random.normal(0, 0.1, size=action_raw.shape)
                        action_raw = np.clip(action_raw, -1.0, 1.0)
                        
                    action = action_raw # This is the array [6]
                    
                    # For visualization, infer the "Mode" from index 5
                    mode_val = action[5]
                    if mode_val < -0.5: display_action = 0
                    elif mode_val < 0.0: display_action = 1
                    elif mode_val < 0.5: display_action = 2
                    else: display_action = 3

                # STEP
                obs, reward, terminated, truncated, info = env.step(action)
                
                raw_state = env.env.state 
                indoor_temp = raw_state["indoor_temperature"][0]
                
                # Track Comfort
                if 23.0 <= indoor_temp <= 25.0:
                    steps_comfort += 1
                    comfort_status = "Comfortable"
                elif indoor_temp < 23.0:
                    steps_too_cold += 1
                    comfort_status = "Too Cold"
                else:
                    steps_too_hot += 1
                    comfort_status = "Too Hot"

                log_entry = {
                    "Step": step,
                    "Indoor Temp": indoor_temp,
                    "Outdoor Temp": raw_state["outdoor_temperature"][0],
                    "Power (W)": raw_state["power_consumption"][0],
                    "Action Index": display_action,
                    "Reward": reward
                }
                logs.append(log_entry)
                
                if terminated or truncated:
                    break
            
            st.session_state.simulation_logs = pd.DataFrame(logs)
            st.session_state.steps_comfort = steps_comfort
            st.session_state.steps_too_cold = steps_too_cold
            st.session_state.steps_too_hot = steps_too_hot
            st.session_state.n_steps = n_steps 

    # VISUALIZATION
    if st.session_state.simulation_logs is not None:
        df = st.session_state.simulation_logs
        
        # 1. Metrics
        col1, col2, col3, col4 = st.columns(4)
        avg_temp = df["Indoor Temp"].mean()
        total_energy = df["Power (W)"].sum() / 1000.0 
        success = "Yes" if len(df) >= st.session_state.n_steps and not df["Indoor Temp"].iloc[-1] > 35 else "No (Crash)"
        
        comfort_score = (st.session_state.steps_comfort / len(df)) * 100
        
        col1.metric("Avg Indoor Temp", f"{avg_temp:.2f} °C")
        col2.metric("Total Energy", f"{total_energy:.2f} kWh")
        col3.metric("Comfort Score", f"{comfort_score:.1f}%")
        col4.metric("Episode Success", success)
        
        # 2. Charts
        st.subheader("Temperature Profile & Comfort Zone")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Step'], df['Indoor Temp'], label="Indoor Temp", color="#d62728", linewidth=2)
        ax.plot(df['Step'], df['Outdoor Temp'], label="Outdoor Temp", color="#1f77b4", linestyle="--", alpha=0.5)
        ax.axhspan(23.0, 25.0, color='green', alpha=0.2, label="Comfort Zone")
        ax.axhline(y=23.0, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=25.0, color='green', linestyle=':', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 3. Breakdown & Actions
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
             # Pie Chart
            labels = ['Too Cold', 'Comfortable', 'Too Hot']
            sizes = [st.session_state.steps_too_cold, st.session_state.steps_comfort, st.session_state.steps_too_hot]
            colors = ['#3498db', '#2ecc71', '#e74c3c'] 
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
            ax3.set_title("Thermal Comfort Breakdown")
            st.pyplot(fig3)
            
        with col_chart2:
            st.subheader("Action Mode Distribution")
            action_counts = df['Action Index'].value_counts().sort_index()
            # Map labels for display
            action_map_labels = {0: "0: Off/Idle", 1: "1: Cool Low", 2: "2: Cool High", 3: "3: Fan Only"}
            action_counts.index = action_counts.index.map(action_map_labels)
            st.bar_chart(action_counts)

        if st.button("Clear Results"):
            st.session_state.simulation_logs = None
            st.rerun()

else:
    st.info("👈 Please upload a .zip (DQN) or .pth (DDPG) model file.")
