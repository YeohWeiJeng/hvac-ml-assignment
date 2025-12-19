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

# --- SIDEBAR ---
st.sidebar.header("1. Upload Model")
uploaded_file = st.sidebar.file_uploader("Upload .zip model file", type="zip")
st.sidebar.header("2. Simulation Settings")
n_steps = st.sidebar.slider("Simulation Steps", min_value=100, max_value=2000, value=600)
deterministic = st.sidebar.checkbox("Deterministic (Best Action)", value=True)

# --- MAIN LOGIC ---
if uploaded_file is not None:
    with open("temp_model.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
        model = DQN.load("temp_model.zip")
        st.sidebar.success("✅ Standard DQN Model Loaded!")

    if st.sidebar.button("Run Simulation"):
        # Setup Env
        raw_env = HVACEnv()
        env = DiscreteActionWrapper(raw_env)
        
        obs, _ = env.reset()
        done = False
        logs = []
        
        progress_bar = st.progress(0)
        
        # --- Metrics Tracking ---
        comfort_zone_min = 23.0
        comfort_zone_max = 25.0
        steps_comfort = 0
        steps_too_hot = 0
        steps_too_cold = 0
        
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

