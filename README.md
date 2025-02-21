# MarketPulse-The-Live-AI-Trading-Hub

Brief Description of Key Components
Custom Trading Environment: A Gym environment that simulates stock and option trading with dynamic pricing, simulated news events, and transaction logic.
Reinforcement Learning Model: Utilizes PPO from Stable Baselines3 to learn trading strategies.
Live Dashboard: A Tkinter-based GUI (with ttkthemes) featuring multiple tabs that display portfolio metrics, real-time charts, market data, trade histories, and AI insights.
Data Visualization: Uses Matplotlib to render interactive charts (bar charts, heatmaps, pie charts) for market trends and performance metrics.
Background Training & Updates: A background thread continuously trains the model, while the GUI updates every second to reflect live trading data.
