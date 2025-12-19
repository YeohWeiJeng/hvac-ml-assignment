import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from gymnasium import spaces
from torch.distributions import Categorical
from stable_baselines3 import DQN

# ==========================================
# 1. CLASS DEFINITIONS (Must match training)
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

        # Rewards & Weights
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
            terminated = True # Safety cutoff

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

class SoftmaxDQN(DQN):
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="HVAC Model Visualizer", layout="wide")
st.title("🌡️ HVAC Reinforcement Learning Visualizer")

# --- SIDEBAR: Upload Model ---
st.sidebar.header("1. Upload Model")
uploaded_file = st.sidebar.file_uploader("Upload .zip model file", type="zip")

# --- SIDEBAR: Settings ---
st.sidebar.header("2. Simulation Settings")
n_steps = st.sidebar.slider("Simulation Steps", min_value=100, max_value=2000, value=600)
deterministic = st.sidebar.checkbox("Deterministic (Best Action)", value=True)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    # Save the uploaded file temporarily so SB3 can load it
    with open("temp_model.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load Model
    try:
        # We try loading as SoftmaxDQN, fallback to standard DQN if needed
        model = SoftmaxDQN.load("temp_model.zip", custom_objects={"SoftmaxDQN": SoftmaxDQN})
        st.sidebar.success("✅ SoftmaxDQN Model Loaded!")
    except:
        model = DQN.load("temp_model.zip")
        st.sidebar.success("✅ Standard DQN Model Loaded!")

    # Run Simulation Button
    if st.sidebar.button("Run Simulation"):
        # Setup Env
        raw_env = HVACEnv()
        env = DiscreteActionWrapper(raw_env)
        
        obs, _ = env.reset()
        done = False
        logs = []
        
        progress_bar = st.progress(0)
        
        for step in range(n_steps):
            # Predict
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Log Data (Extract from raw environment state for better visibility)
            # The 'env' is the wrapper, 'env.env' is the raw HVACEnv
            raw_state = env.env.state 
            
            log_entry = {
                "Step": step,
                "Indoor Temp": raw_state["indoor_temperature"][0],
                "Outdoor Temp": raw_state["outdoor_temperature"][0],
                "Occupancy": raw_state["occupancy"],
                "Power (W)": raw_state["power_consumption"][0],
                "Action Index": int(action),
                "Reward": reward
            }
            logs.append(log_entry)
            
            if terminated or truncated:
                break
            
            progress_bar.progress((step + 1) / n_steps)

        # Convert to DataFrame
        df = pd.DataFrame(logs)
        
        # --- VISUALIZATION ---
        
        # 1. Metrics Row
        col1, col2, col3 = st.columns(3)
        avg_temp = df["Indoor Temp"].mean()
        total_energy = df["Power (W)"].sum() / 1000.0 # kWh estimate (roughly)
        success = "Yes" if len(df) >= n_steps and not df["Indoor Temp"].iloc[-1] > 35 else "No (Crash)"
        
        col1.metric("Avg Indoor Temp", f"{avg_temp:.2f} °C")
        col2.metric("Total Power Est.", f"{total_energy:.2f} kW-steps")
        col3.metric("Episode Success", success)
        
        # 2. Temperature Plot
        st.subheader("Temperature Profile")
        chart_data = df[["Step", "Indoor Temp", "Outdoor Temp"]].set_index("Step")
        # Add target lines
        chart_data["Target Max"] = 25.0
        chart_data["Target Min"] = 23.0
        st.line_chart(chart_data, color=["#FF4B4B", "#1F77B4", "#00FF00", "#00FF00"])
        
        # 3. Power & Occupancy
        st.subheader("Power Consumption vs Occupancy")
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Power (Watts)', color='tab:red')
        ax1.plot(df['Step'], df['Power (W)'], color='tab:red', alpha=0.6, label="Power")
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Occupancy', color='tab:blue')
        ax2.plot(df['Step'], df['Occupancy'], color='tab:blue', linestyle='--', alpha=0.5, label="Occupancy")
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        st.pyplot(fig)
        
        # 4. Actions Taken Distribution
        st.subheader("Actions Taken Distribution")
        action_counts = df['Action Index'].value_counts().sort_index()
        action_map_labels = {0: "0: Off/Idle", 1: "1: Cool Low", 2: "2: Cool High", 3: "3: Fan Only"}
        action_counts.index = action_counts.index.map(action_map_labels)
        st.bar_chart(action_counts)
        
else:
    st.info("👈 Please upload a .zip model file from the sidebar to start.")