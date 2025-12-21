import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gymnasium import spaces

# SB3 Imports
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 1. SHARED ENVIRONMENT
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
            "occupancy": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32), 
            "time_of_day": spaces.Box(low=0.0, high=24.0, shape=(1,), dtype=np.float32), 
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
            "occupancy": np.array([0.0], dtype=np.float32),
            "time_of_day": np.array([8.0],
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
        self.state["time_of_day"][0] = (8 + (self.current_step // 60)) % 24
        hour = self.state["time_of_day"]
        
        base_outdoor = 30.0
        self.state["outdoor_temperature"][0] = base_outdoor + 5.0 * np.sin((hour - 8) * np.pi / 12)

        if 9 <= hour <= 17:
            self.state["occupancy"][0] = float(np.random.randint(5, 20))
        else:
            self.state["occupancy"][0] = 0.0

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
# 2. WRAPPERS
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

class ContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 6 continuous values
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def action(self, act):
        # Map 6 floats to dictionary
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

# ==========================================
# 3. DDPG ACTOR (Custom Class)
# ==========================================
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
st.title("🌡️ HVAC Unified Visualizer (DQN, DDPG, SAC)")

# --- SIDEBAR: Model Selector ---
st.sidebar.header("1. Select Model Type")
model_option = st.sidebar.selectbox("Choose Algorithm", ["DQN (Discrete)", "DDPG (Continuous)", "SAC (Continuous + Normalized)"])

# --- SIDEBAR: File Uploader Logic ---
st.sidebar.header("2. Upload Files")
model = None
vec_norm_env = None # Specific for SAC

if model_option == "DQN (Discrete)":
    uploaded_file = st.sidebar.file_uploader("Upload DQN .zip", type=["zip"])
    if uploaded_file:
        with open("temp_dqn.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            model = DQN.load("temp_dqn.zip")
            st.sidebar.success("✅ DQN Loaded")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

elif model_option == "DDPG (Continuous)":
    uploaded_file = st.sidebar.file_uploader("Upload DDPG .pth", type=["pth"])
    if uploaded_file:
        with open("temp_ddpg.pth", "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            temp_env = HVACEnv()
            state_dim = sum(np.prod(v.shape) if isinstance(v, np.ndarray) else 1 for v in temp_env.reset()[0].values())
            action_dim = 6
            model = Actor(state_dim, action_dim)
            model.load_state_dict(torch.load("temp_ddpg.pth"))
            model.eval()
            st.sidebar.success("✅ DDPG Actor Loaded")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

elif model_option == "SAC (Continuous + Normalized)":
    st.info("SAC requires both the model (.zip) and the normalization stats (.pkl).")
    uploaded_model = st.sidebar.file_uploader("1. Upload SAC .zip", type=["zip"])
    uploaded_stats = st.sidebar.file_uploader("2. Upload vec_normalize.pkl", type=["pkl"])
    
    if uploaded_model and uploaded_stats:
        with open("temp_sac.zip", "wb") as f1: f1.write(uploaded_model.getbuffer())
        with open("vec_normalize.pkl", "wb") as f2: f2.write(uploaded_stats.getbuffer())
        
        try:
            # Load Model
            model = SAC.load("temp_sac.zip")
            # Load Stats into a dummy env
            # Note: We must recreate the exact env structure
            dummy_env = DummyVecEnv([lambda: ContinuousActionWrapper(HVACEnv())])
            vec_norm_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
            vec_norm_env.training = False # Turn off training mode
            vec_norm_env.norm_reward = False
            
            st.sidebar.success("✅ SAC & Normalization Stats Loaded")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# --- Simulation Settings ---
st.sidebar.header("3. Settings")
n_steps = st.sidebar.slider("Simulation Steps", min_value=100, max_value=2000, value=600)
deterministic = st.sidebar.checkbox("Deterministic (Best Action)", value=True)

# --- Initialize Session State ---
if 'simulation_logs' not in st.session_state:
    st.session_state.simulation_logs = None

# --- RUN SIMULATION BUTTON ---
if st.sidebar.button("Run Simulation"):
    if model is None:
        st.error("Please upload the required model files first!")
    else:
        with st.spinner("Simulating..."):
            logs = []
            steps_comfort = 0
            steps_too_hot = 0
            steps_too_cold = 0
            
            # Setup Environment based on Model Type
            if model_option == "DQN (Discrete)":
                env = DiscreteActionWrapper(HVACEnv())
                obs, _ = env.reset()
                
            elif model_option == "DDPG (Continuous)":
                env = ContinuousActionWrapper(HVACEnv())
                obs, _ = env.reset()
                
            elif model_option == "SAC (Continuous + Normalized)":
                # SAC uses the Vectorized Env we created earlier
                env = vec_norm_env
                obs = env.reset() # This returns a vectorized observation
            
            # SIMULATION LOOP
            for step in range(n_steps):
                display_action = 0
                
                # --- PREDICT ACTION ---
                if model_option == "DQN (Discrete)":
                    action, _ = model.predict(obs, deterministic=deterministic)
                    display_action = int(action)
                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Get Raw State
                    raw_state = env.env.state
                    
                elif model_option == "DDPG (Continuous)":
                    state_flat = flatten_state(obs)
                    state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)
                    with torch.no_grad():
                        action_raw = model(state_tensor).squeeze(0).numpy()
                    
                    if not deterministic:
                        action_raw += np.random.normal(0, 0.1, size=action_raw.shape)
                        action_raw = np.clip(action_raw, -1.0, 1.0)
                    
                    action = action_raw
                    # Determine mode for display
                    if action[5] < -0.5: display_action = 0
                    elif action[5] < 0.0: display_action = 1
                    elif action[5] < 0.5: display_action = 2
                    else: display_action = 3
                    
                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Get Raw State
                    raw_state = env.env.state

                elif model_option == "SAC (Continuous + Normalized)":
                    # SAC obs is already vectorized from env.reset()
                    action, _ = model.predict(obs, deterministic=deterministic)
                    
                    # Determine mode for display (action is [N_Envs, 6])
                    act_val = action[0][5]
                    if act_val < -0.5: display_action = 0
                    elif act_val < 0.0: display_action = 1
                    elif act_val < 0.5: display_action = 2
                    else: display_action = 3
                    
                    # Step (VecEnv returns: obs, rewards, dones, infos)
                    obs, reward, dones, infos = env.step(action)
                    done = dones[0]
                    
                    # Get Raw State from the Unwrapped Env inside VecEnv
                    # env -> VecNormalize -> DummyVecEnv -> ContinuousActionWrapper -> HVACEnv
                    raw_state = env.venv.envs[0].unwrapped.state

                # --- LOGGING ---
                indoor_temp = raw_state["indoor_temperature"][0]
                
                if 23.0 <= indoor_temp <= 25.0: steps_comfort += 1
                elif indoor_temp < 23.0: steps_too_cold += 1
                else: steps_too_hot += 1

                logs.append({
                    "Step": step,
                    "Indoor Temp": indoor_temp,
                    "Outdoor Temp": raw_state["outdoor_temperature"][0],
                    "Power (W)": raw_state["power_consumption"][0],
                    "Action Index": display_action,
                })
                
                if done:
                    break

            # Save to Session
            st.session_state.simulation_logs = pd.DataFrame(logs)
            st.session_state.steps_comfort = steps_comfort
            st.session_state.steps_too_cold = steps_too_cold
            st.session_state.steps_too_hot = steps_too_hot
            st.session_state.n_steps = n_steps

# --- VISUALIZATION ---
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
    
    # 2. Temperature Plot
    st.subheader("Temperature Profile & Comfort Zone")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Step'], df['Indoor Temp'], label="Indoor Temp", color="#d62728", linewidth=2)
    ax.plot(df['Step'], df['Outdoor Temp'], label="Outdoor Temp", color="#1f77b4", linestyle="--", alpha=0.5)
    ax.axhspan(23.0, 25.0, color='green', alpha=0.2, label="Comfort Zone")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 3. Breakdown & Actions
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
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
        action_map_labels = {0: "0: Off/Idle", 1: "1: Cool Low", 2: "2: Cool High", 3: "3: Fan Only"}
        action_counts.index = action_counts.index.map(action_map_labels)
        st.bar_chart(action_counts)

    if st.button("Clear Results"):
        st.session_state.simulation_logs = None
        st.rerun()

