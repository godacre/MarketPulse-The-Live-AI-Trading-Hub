import numpy as np
from gymnasium import Env, spaces
from collections import deque
from stable_baselines3 import PPO
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
import matplotlib.patches as patches
import time
import math
from ttkthemes import ThemedTk

plt.rcParams['figure.max_open_warning'] = 100  # Shh, matplotlib, no whining allowed

class StockTradingEnv(Env):
    def __init__(self, asset_names, initial_cash=10000, transaction_cost_pct=0.001, max_qty=50):
        super(StockTradingEnv, self).__init__()
        self.asset_names = asset_names
        self.N = len(asset_names)
        self.initial_cash = initial_cash  # Your starting dough
        self.transaction_cost_pct = transaction_cost_pct  # The taxman’s tiny cut
        self.max_qty = max_qty  # Don’t go too wild with trades

        self.action_space = spaces.MultiDiscrete([7, max_qty + 1] * self.N)  # 7 actions per asset, financial superhero vibes
        self.K = 120  # Price history window, because who remembers last year?
        self.observation_space = spaces.Box(
            low=0, high=np.inf,
            shape=(self.N * self.K + self.N * 3 + 1,),
            dtype=np.float32
        )

        self.current_step = 0  # Where we’re at in this chill ride
        self.current_day = 0  # Day counter for “new day, same old me”
        self.cash = initial_cash
        self.shares = np.zeros(self.N)
        self.calls = np.zeros(self.N)
        self.puts = np.zeros(self.N)
        self.prices = None
        self.price_history = []
        self.last_trades = []
        self.trade_history = []
        self.active_positions = [(name, 0, 0, 0) for name in self.asset_names]  # What you’re holding onto
        self.news_events = []
        self.total_trades = 0
        self.day_start_portfolio_value = initial_cash

        # Market’s now so calm it’s basically meditating
        self.mu = np.concatenate([
            np.full(10, 0.005),  # Growing slower than a snail on vacation
            np.full(15, 0.003),
            np.full(10, 0.004),
            np.full(5, 0.003),
            np.full(10, 0.007)
        ])
        self.sigma = np.concatenate([
            np.linspace(0.03, 0.015, 10),
            np.linspace(0.02, 0.01, 15),
            np.linspace(0.025, 0.015, 10),
            np.full(5, 0.015),
            np.linspace(0.04, 0.025, 10)
        ]) * 0.01  # Volatility so tiny it’s practically a whisper

        self.dt = 1 / (252 * 390)  # One minute in a trading year, snooze
        self.initial_prices = np.array([
            35, 20, 15, 65, 10, 45, 25, 20, 15, 8,
            190, 450, 150, 90, 60, 55, 75, 145, 290, 360,
            170, 40, 80, 110, 35,
            140, 70, 90, 130, 110, 120, 85, 95, 105, 60,
            570, 520, 230, 340, 220,
            60000, 3000, 600, 1.2, 0.8, 250, 0.15, 0.0001, 100, 20
        ])
        self.rng = np.random.RandomState()  # Randomness, but super tame
        self.news_effects = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)  # For when you want predictable boredom
        self.current_step = 0
        self.current_day = 0
        self.cash = self.initial_cash
        self.shares = np.zeros(self.N)
        self.calls = np.zeros(self.N)
        self.puts = np.zeros(self.N)
        self.prices = [deque([self.initial_prices[i]] * self.K, maxlen=self.K) for i in range(self.N)]
        self.price_history = [self.initial_prices.copy()]
        self.last_trades = []
        self.trade_history = []
        self.active_positions = [(name, 0, 0, 0) for name in self.asset_names]
        self.news_events = []
        self.total_trades = 0
        self.news_effects = {}
        self.day_start_portfolio_value = self.initial_cash
        return self._get_state(), {}

    def step(self, action):
        prev_portfolio_value = self._get_portfolio_value()
        new_day = self.current_step // 390
        if new_day > self.current_day:
            self.current_day = new_day
            self.day_start_portfolio_value = prev_portfolio_value
            print(f"New trading day started: Day {self.current_day} - Yawn, here we go again!")

        if self.rng.random() < 0.05:  # 5% chance of a news nap
            self._generate_news_event()

        for idx in list(self.news_effects.keys()):
            mu_adj, sigma_adj, steps_left = self.news_effects[idx]
            self.mu[idx] += mu_adj
            self.sigma[idx] = max(0.001, self.sigma[idx] + sigma_adj)  # Volatility’s basically dead
            self.news_effects[idx] = (mu_adj, sigma_adj, steps_left - 1)
            if self.news_effects[idx][2] <= 0:
                del self.news_effects[idx]
                self.mu[idx] -= mu_adj
                self.sigma[idx] -= sigma_adj

        new_prices = []
        for i in range(self.N):
            current_price = self.prices[i][-1]
            drift = (self.mu[i] - 0.5 * self.sigma[i]**2) * self.dt
            diffusion = self.sigma[i] * np.sqrt(self.dt) * self.rng.normal()
            new_price = current_price * np.exp(drift + diffusion)  # Prices creep like a sloth
            self.prices[i].append(new_price)
            new_prices.append(new_price)
        self.price_history.append(new_prices.copy())
        self.last_trades = []

        candidates = []
        for i in range(self.N):
            act = action[2 * i]
            qty = action[2 * i + 1]
            if act != 0 and qty > 0:
                candidates.append((i, qty))
        if candidates:
            idx, qty = max(candidates, key=lambda x: x[1])
            self._execute_trade(idx, action[2 * idx], qty, new_prices)

        self.active_positions = [(self.asset_names[i], self.shares[i], self.calls[i], self.puts[i])
                                 for i in range(self.N)]
        self.current_step += 1

        portfolio_value = self._get_portfolio_value()
        reward = portfolio_value - prev_portfolio_value
        return self._get_state(), reward, False, False, {}

    def _execute_trade(self, idx, act, quantity, new_prices):
        price = new_prices[idx]
        option_price = self._option_price(price, idx)

        if act == 1:  # Buy Stock - Retail therapy, but slow
            cost = quantity * price * (1 + self.transaction_cost_pct)
            if self.cash >= cost:
                self.shares[idx] += quantity
                self.cash -= cost
                self.last_trades.append((self.asset_names[idx], "Buy Stock", quantity, price))
                self.trade_history.append((self.current_step, self.asset_names[idx],
                                           "Buy Stock", quantity, price, 0))
                self.total_trades += 1

        elif act == 2:  # Sell Stock - Cashing out for a coffee
            if self.shares[idx] >= quantity:
                proceeds = quantity * price * (1 - self.transaction_cost_pct)
                self.shares[idx] -= quantity
                self.cash += proceeds
                self.last_trades.append((self.asset_names[idx], "Sell Stock", quantity, price))
                buy_trades = [t for t in self.trade_history if t[1] == self.asset_names[idx]
                              and t[2] == "Buy Stock" and t[5] == 0]
                if buy_trades:
                    buy_step, _, _, qty, buy_price, _ = buy_trades[-1]
                    profit = (price - buy_price) * qty * (1 - self.transaction_cost_pct)
                    self.trade_history.append((buy_step, self.asset_names[idx],
                                               "Sell Stock", qty, price, profit))
                    self.trade_history[-2] = (buy_step, self.asset_names[idx],
                                              "Buy Stock", qty, buy_price, profit)
                self.total_trades += 1

        elif act == 3:  # Buy Call - Hoping for a tiny uptick
            cost = quantity * option_price * (1 + self.transaction_cost_pct)
            if self.cash >= cost:
                self.calls[idx] += quantity
                self.cash -= cost
                self.last_trades.append((self.asset_names[idx], "Buy Call", quantity, option_price))
                self.trade_history.append((self.current_step, self.asset_names[idx],
                                           "Buy Call", quantity, option_price, 0))
                self.total_trades += 1

        elif act == 4:  # Sell Call - Tiny profit, big smug
            if self.calls[idx] >= quantity:
                proceeds = quantity * option_price * (1 - self.transaction_cost_pct)
                self.calls[idx] -= quantity
                self.cash += proceeds
                self.last_trades.append((self.asset_names[idx], "Sell Call", quantity, option_price))
                buy_trades = [t for t in self.trade_history if t[1] == self.asset_names[idx]
                              and t[2] == "Buy Call" and t[5] == 0]
                if buy_trades:
                    buy_step, _, _, qty, buy_price, _ = buy_trades[-1]
                    profit = (option_price - buy_price) * qty * (1 - self.transaction_cost_pct)
                    self.trade_history.append((buy_step, self.asset_names[idx],
                                               "Sell Call", qty, option_price, profit))
                    self.trade_history[-2] = (buy_step, self.asset_names[idx],
                                              "Buy Call", qty, buy_price, profit)
                self.total_trades += 1

        elif act == 5:  # Buy Put - Betting on a micro-dip
            cost = quantity * option_price * (1 + self.transaction_cost_pct)
            if self.cash >= cost:
                self.puts[idx] += quantity
                self.cash -= cost
                self.last_trades.append((self.asset_names[idx], "Buy Put", quantity, option_price))
                self.trade_history.append((self.current_step, self.asset_names[idx],
                                           "Buy Put", quantity, option_price, 0))
                self.total_trades += 1

        elif act == 6:  # Sell Put - Cashing in on boredom
            if self.puts[idx] >= quantity:
                proceeds = quantity * option_price * (1 - self.transaction_cost_pct)
                self.puts[idx] -= quantity
                self.cash += proceeds
                self.last_trades.append((self.asset_names[idx], "Sell Put", quantity, option_price))
                buy_trades = [t for t in self.trade_history if t[1] == self.asset_names[idx]
                              and t[2] == "Buy Put" and t[5] == 0]
                if buy_trades:
                    buy_step, _, _, qty, buy_price, _ = buy_trades[-1]
                    profit = (option_price - buy_price) * qty * (1 - self.transaction_cost_pct)
                    self.trade_history.append((buy_step, self.asset_names[idx],
                                               "Sell Put", qty, option_price, profit))
                    self.trade_history[-2] = (buy_step, self.asset_names[idx],
                                              "Buy Put", qty, buy_price, profit)
                self.total_trades += 1

    def _get_state(self):
        state = []
        for i in range(self.N):
            state.extend(list(self.prices[i]))
        state.extend(self.shares)
        state.extend(self.calls)
        state.extend(self.puts)
        state.append(self.cash)
        return np.array(state, dtype=np.float32)

    def _get_portfolio_value(self):
        stock_value = np.sum(self.shares * [p[-1] for p in self.prices])
        option_value = sum(
            self._option_price(self.prices[i][-1], i) * (self.calls[i] + self.puts[i])
            for i in range(self.N)
        )
        return self.cash + stock_value + option_value  # How rich are we today? Not very, with this market

    def _option_price(self, price, idx):
        strike = self.initial_prices[idx]
        time_to_expiry = 30 / (252 * 390)
        vol = self.sigma[idx]
        r = 0.01
        d1 = (np.log(price / strike) + (r + 0.5 * vol**2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
        d2 = d1 - vol * np.sqrt(time_to_expiry)
        call_price = price * self._norm_cdf(d1) - strike * np.exp(-r * time_to_expiry) * self._norm_cdf(d2)
        return max(call_price, 0.1)  # Options cling to a shred of hope

    def _norm_cdf(self, x):
        # Cap x to prevent exp overflow - because even math needs a chill pill
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-1.6 * x))  # Math curve so chill it’s horizontal, now overflow-proof

    def _generate_news_event(self):
        event_types = [
            ("{stock} unveils new blockchain platform.", (0.002, 0, 40)),  # News so tame it’s a whisper
            ("{stock} faces regulatory crackdown.", (-0.002, 0.005, 30)),
            ("Supply chain woes hit {stock}.", (-0.001, 0.003, 50)),
            ("{stock} partners with Tesla.", (0.003, -0.002, 60)),
            ("Crypto hack targets {stock}.", (0, 0.01, 25)),
            ("Retail boom lifts {stock}.", (0.002, 0, 45)),
        ]
        event_idx = self.rng.randint(len(event_types))
        event_template, (mu_adj, sigma_adj, duration) = event_types[event_idx]
        affected_stock = self.rng.choice(self.asset_names)
        idx = self.asset_names.index(affected_stock)
        event = event_template.format(stock=affected_stock)
        self.news_events.append((self.current_step, idx, event))
        self.news_effects[idx] = (mu_adj, sigma_adj, duration)  # Barely a ripple

    def get_volatility(self, stock_idx, window=20):
        prices = list(self.prices[stock_idx])[-window:]
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252 * 390)  # Volatility’s taking a nap

    def get_price_change_since_news(self, asset_idx, news_step):
        current_len = len(self.prices[asset_idx])
        if news_step >= self.current_step or news_step < 0 or current_len < 2:
            return 0
        news_idx = max(0, current_len - (self.current_step - news_step) - 1)
        if news_idx >= current_len or news_idx < 0:
            return 0
        start_price = self.prices[asset_idx][news_idx]
        current_price = self.prices[asset_idx][-1]
        return ((current_price - start_price) / start_price) * 100 if start_price != 0 else 0

    def get_daily_change(self, asset_idx):
        steps_since_day_start = self.current_step % 390
        current_len = len(self.prices[asset_idx])
        if current_len < 2 or steps_since_day_start == 0:
            return 0
        day_start_idx = max(0, current_len - steps_since_day_start - 1)
        start_price = self.prices[asset_idx][day_start_idx]
        current_price = self.prices[asset_idx][-1]
        return ((current_price - start_price) / start_price) * 100 if start_price != 0 else 0


class TradingDashboard:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.root = ThemedTk(theme="arc")
        self.root.title("Market Pulse")
        self.root.geometry("1800x1200")
        self.root.configure(bg="#1A2538")  # Dark and moody, like my coffee

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#1A2538", foreground="#D9E6F5", font=("Arial", 12))
        style.configure("TFrame", background="#1A2538")
        style.configure("TNotebook", background="#1A2538", foreground="#D9E6F5")
        style.configure("TNotebook.Tab", font=("Arial", 14, "bold"), padding=(12, 6), background="#2F3B5A")
        style.configure("Treeview", background="#2F3B5A", foreground="#D9E6F5",
                        fieldbackground="#2F3B5A", font=("Arial", 11))
        style.map("Treeview", background=[('selected', '#3F4A6A')], foreground=[('selected', '#FFFFFF')])
        style.layout("Treeview.Heading", [
            ('Treeheading.cell', {'sticky': 'nswe', 'children': [
                ('Treeheading.padding', {'sticky': 'nswe', 'children': [
                    ('Treeheading.text', {'sticky': 'we'})
                ]})
            ]})
        ])
        style.configure("Treeview.Heading", background="#2F3B5A", foreground="#FFFFCC",
                        font=("Arial", 12, "bold"), relief="flat")
        style.map("Treeview.Heading", background=[('active', '#3F4A6A')])

        # Portfolio Frame - Right at the top, no wasted space!
        self.portfolio_frame = ttk.Frame(self.root, relief="raised", borderwidth=2, style="TFrame")
        self.portfolio_frame.place(x=1550, y=0, width=250, height=180)  # Snug in the top right corner
        ttk.Label(self.portfolio_frame, text="Portfolio Value:", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=5, pady=2)
        self.portfolio_value_label = ttk.Label(self.portfolio_frame, text="$0.00", font=("Arial", 12))
        self.portfolio_value_label.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.portfolio_frame, text="Daily P&L %:", font=("Arial", 12, "bold")).grid(row=1, column=0, padx=5, pady=2)
        self.daily_pnl_label = ttk.Label(self.portfolio_frame, text="0.00%", font=("Arial", 12))
        self.daily_pnl_label.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(self.portfolio_frame, text="Current Day:", font=("Arial", 12, "bold")).grid(row=2, column=0, padx=5, pady=2)
        self.current_day_label = ttk.Label(self.portfolio_frame, text="0", font=("Arial", 12))
        self.current_day_label.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(self.portfolio_frame, text="All-Time P&L:", font=("Arial", 12, "bold")).grid(row=3, column=0, padx=5, pady=2)
        self.all_time_pnl_label = ttk.Label(self.portfolio_frame, text="$0.00", font=("Arial", 12))
        self.all_time_pnl_label.grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(self.portfolio_frame, text="All-Time P&L %:", font=("Arial", 12, "bold")).grid(row=4, column=0, padx=5, pady=2)
        self.all_time_pnl_pct_label = ttk.Label(self.portfolio_frame, text="0.00%", font=("Arial", 12))
        self.all_time_pnl_pct_label.grid(row=4, column=1, padx=5, pady=2)
        self.portfolio_frame.lift()

        self.watermark_label = ttk.Label(self.root, text="Goodacre", font=("Arial", 12, "italic"),
                                         foreground="#D9E6F5", background="#1A2538")
        self.watermark_label.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-10, y=-10)
        self.watermark_label.lift()

        self.pnl_window = tk.Toplevel(self.root)
        self.pnl_window.title("P/L")
        self.pnl_window.geometry("250x120+0+0")
        self.pnl_window.overrideredirect(True)
        self.pnl_window.configure(bg="#2F3B5A")
        self.pnl_label = ttk.Label(self.pnl_window, text="P/L: 0.00", font=("Arial", 18, "bold"),
                                   background="#2F3B5A", foreground="#FFFFFF")
        self.pnl_label.pack(pady=30)  # Floating P&L, because it’s too cool to stay put

        # Notebook - Shoved right up to the top
        self.notebook = ttk.Notebook(self.root)
        self.notebook.place(x=0, y=0, width=1550, height=1200)  # Full width, zero top padding

        self.home_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.home_frame, text="Home")
        self.home_labels = {}
        for i, (text, key) in enumerate([
            ("Cash:", "cash"),
            ("Portfolio:", "portfolio"),
            ("P&L:", "pnl"),
            ("Market Index:", "market_index")
        ]):
            ttk.Label(self.home_frame, text=text, font=("Arial", 16, "bold")).place(x=i*300+15, y=5, width=120)
            self.home_labels[key] = ttk.Label(self.home_frame, text="0.00", font=("Arial", 16))
            self.home_labels[key].place(x=i*300+135, y=5, width=150)

        self.home_fig, self.home_ax = plt.subplots(figsize=(16, 7))
        self.home_fig.patch.set_facecolor('#2F3B5A')
        self.home_ax.set_facecolor('#2F3B5A')
        self.home_canvas = FigureCanvasTkAgg(self.home_fig, master=self.home_frame)
        self.home_canvas.get_tk_widget().place(x=15, y=40, width=1500, height=400)  # Adjusted width for new layout

        news_frame = ttk.Frame(self.home_frame)
        news_frame.place(x=15, y=450, width=1500, height=500)  # News still fits perfectly
        ttk.Label(news_frame, text="News of the Day", font=("Arial", 16, "bold")).pack(anchor="nw", pady=5)
        self.home_news = tk.Text(news_frame, height=25, wrap=tk.WORD,
                                 bg="#2F3B5A", fg="#D9E6F5", font=("Arial", 14))
        self.home_news.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        home_news_scrollbar = ttk.Scrollbar(news_frame, orient="vertical", command=self.home_news.yview)
        home_news_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.home_news.configure(yscrollcommand=home_news_scrollbar.set)

        self.market_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.market_frame, text="Market Data")
        ttk.Label(self.market_frame, text="Market Overview", font=("Arial", 16, "bold"), foreground="#FFFFCC").place(x=15, y=5)
        self.market_tree = ttk.Treeview(self.market_frame,
                                        columns=("Asset", "Price", "Daily Change", "Volatility", "Market Cap"),
                                        show="headings", height=20, style="Treeview")
        self.market_tree.place(x=15, y=40, width=1000, height=1000)
        self.market_tree.heading("Asset", text="Asset")
        self.market_tree.heading("Price", text="Price ($)")
        self.market_tree.heading("Daily Change", text="Daily Change (%)")
        self.market_tree.heading("Volatility", text="Volatility (%)")
        self.market_tree.heading("Market Cap", text="Market Cap ($B)")
        self.market_tree.column("Asset", width=120, anchor="center")
        self.market_tree.column("Price", width=100, anchor="center")
        self.market_tree.column("Daily Change", width=120, anchor="center")
        self.market_tree.column("Volatility", width=100, anchor="center")
        self.market_tree.column("Market Cap", width=120, anchor="center")
        market_scrollbar = ttk.Scrollbar(self.market_frame, orient="vertical", command=self.market_tree.yview)
        market_scrollbar.place(x=1015, y=40, height=1000)
        self.market_tree.configure(yscrollcommand=market_scrollbar.set)
        self.heatmap_fig, self.heatmap_ax = plt.subplots(figsize=(6, 6))
        self.heatmap_fig.patch.set_facecolor('#2F3B5A')
        self.heatmap_ax.set_facecolor('#2F3B5A')
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=self.market_frame)
        self.heatmap_canvas.get_tk_widget().place(x=1050, y=40, width=450, height=450)

        self.positions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.positions_frame, text="Positions")
        ttk.Label(self.positions_frame, text="Current Positions", font=("Arial", 16, "bold"), foreground="#FFFFCC").place(x=15, y=5)
        self.positions_tree = ttk.Treeview(self.positions_frame,
                                           columns=("Asset", "Shares", "Shares Value", "Calls", "Calls Value", "Puts", "Puts Value", "Total"),
                                           show="headings", height=20, style="Treeview")
        self.positions_tree.place(x=15, y=40, width=1000, height=1000)
        self.positions_tree.heading("Asset", text="Asset")
        self.positions_tree.heading("Shares", text="Shares")
        self.positions_tree.heading("Shares Value", text="Shares ($)")
        self.positions_tree.heading("Calls", text="Calls")
        self.positions_tree.heading("Calls Value", text="Calls ($)")
        self.positions_tree.heading("Puts", text="Puts")
        self.positions_tree.heading("Puts Value", text="Puts ($)")
        self.positions_tree.heading("Total", text="Total ($)")
        self.positions_tree.column("Asset", width=100, anchor="center")
        self.positions_tree.column("Shares", width=80, anchor="center")
        self.positions_tree.column("Shares Value", width=100, anchor="center")
        self.positions_tree.column("Calls", width=80, anchor="center")
        self.positions_tree.column("Calls Value", width=100, anchor="center")
        self.positions_tree.column("Puts", width=80, anchor="center")
        self.positions_tree.column("Puts Value", width=100, anchor="center")
        self.positions_tree.column("Total", width=100, anchor="center")
        pos_scrollbar = ttk.Scrollbar(self.positions_frame, orient="vertical", command=self.positions_tree.yview)
        pos_scrollbar.place(x=1015, y=40, height=1000)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
        self.pos_fig, self.pos_ax = plt.subplots(figsize=(8, 6))
        self.pos_fig.patch.set_facecolor('#2F3B5A')
        self.pos_ax.set_facecolor('#2F3B5A')
        self.pos_canvas = FigureCanvasTkAgg(self.pos_fig, master=self.positions_frame)
        self.pos_canvas.get_tk_widget().place(x=1050, y=40, width=450, height=450)

        self.trades_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trades_frame, text="Trades")
        self.trades_lists_frame = ttk.Frame(self.trades_frame)
        self.trades_lists_frame.place(x=15, y=5, width=1000, height=1100)
        ttk.Label(self.trades_lists_frame, text="Open Positions", font=("Arial", 14, "bold")).pack(pady=(5,5))
        self.open_trades_list = tk.Listbox(self.trades_lists_frame, height=15, width=60,
                                           bg="#2F3B5A", fg="#D9E6F5", font=("Arial", 12))
        self.open_trades_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        ttk.Label(self.trades_lists_frame, text="Recent Trades", font=("Arial", 14, "bold")).pack(pady=(0,5))
        self.recent_trades_list = tk.Listbox(self.trades_lists_frame, height=15, width=60,
                                             bg="#2F3B5A", fg="#D9E6F5", font=("Arial", 12))
        self.recent_trades_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        ttk.Label(self.trades_lists_frame, text="Top 10 Profitable Trades", font=("Arial", 14, "bold")).pack(pady=(0,5))
        self.top_trades_list = tk.Listbox(self.trades_lists_frame, height=15, width=60,
                                          bg="#2F3B5A", fg="#D9E6F5", font=("Arial", 12))
        self.top_trades_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.trades_chart_frame = ttk.Frame(self.trades_frame)
        self.trades_chart_frame.place(x=1050, y=40, width=450, height=450)
        self.trade_fig, self.trade_ax = plt.subplots(figsize=(6, 6))
        self.trade_fig.patch.set_facecolor('#2F3B5A')
        self.trade_ax.set_facecolor('#2F3B5A')
        self.trade_canvas = FigureCanvasTkAgg(self.trade_fig, master=self.trades_chart_frame)
        self.trade_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.recent_trades_buffer = deque(maxlen=50)  # Trade memory lane
        self.top_profitable_trades = []  # Hall of fame for big wins

        self.news_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.news_frame, text="News")
        self.ticker_canvas = tk.Canvas(self.news_frame, height=40, bg="#1A2538", highlightthickness=0)
        self.ticker_canvas.place(x=15, y=5, width=1500)
        self.ticker_text = self.ticker_canvas.create_text(0, 20, text="", anchor="w",
                                                          fill="#FFFFCC", font=("Arial", 14, "bold"))
        self.ticker_x = 1800
        self.news_content = tk.Canvas(self.news_frame, bg="#1A2538", highlightthickness=0)
        self.news_content.place(x=15, y=50, width=1000, height=1000)
        self.news_scrollbar = ttk.Scrollbar(self.news_frame, orient="vertical", command=self.news_content.yview)
        self.news_scrollbar.place(x=1015, y=50, height=1000)
        self.news_content.configure(yscrollcommand=self.news_scrollbar.set)
        self.news_inner_frame = ttk.Frame(self.news_content)
        self.news_content.create_window((0, 0), window=self.news_inner_frame, anchor="nw")
        self.news_cards = []
        self.summary_frame = ttk.Frame(self.news_frame, relief="raised", borderwidth=2)
        self.summary_frame.place(x=1050, y=50, width=450, height=450)
        ttk.Label(self.summary_frame, text="News Pulse", font=("Arial", 18, "bold"),
                  foreground="#00FFCC").pack(pady=10)
        self.total_news_label = ttk.Label(self.summary_frame, text="Total Events: 0",
                                          font=("Arial", 14), foreground="#FFFFCC")
        self.total_news_label.pack(pady=5)
        self.biggest_mover_label = ttk.Label(self.summary_frame, text="Biggest Mover: None",
                                             font=("Arial", 14), foreground="#FF6688")
        self.biggest_mover_label.pack(pady=5)

        self.ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ai_frame, text="AI")
        for c in range(4):
            self.ai_frame.columnconfigure(c, weight=1)
        for r in range(5):  # Adjusted for taller layout
            self.ai_frame.rowconfigure(r, weight=1)
        ttk.Label(self.ai_frame, text="AI Reinforcement Learning Insights",
                  font=("Arial", 16, "bold")).place(x=15, y=5)
        self.ai_fig1, self.ai_ax1 = plt.subplots(figsize=(5, 3))
        self.ai_fig1.patch.set_facecolor('#2F3B5A')
        self.ai_ax1.set_facecolor('#2F3B5A')
        self.ai_canvas1 = FigureCanvasTkAgg(self.ai_fig1, master=self.ai_frame)
        self.ai_canvas1.get_tk_widget().place(x=15, y=40, width=350, height=200)
        self.ai_fig2, self.ai_ax2 = plt.subplots(figsize=(5, 3))
        self.ai_fig2.patch.set_facecolor('#2F3B5A')
        self.ai_ax2.set_facecolor('#2F3B5A')
        self.ai_canvas2 = FigureCanvasTkAgg(self.ai_fig2, master=self.ai_frame)
        self.ai_canvas2.get_tk_widget().place(x=375, y=40, width=350, height=200)
        self.ai_fig3 = plt.figure(figsize=(5, 3))
        self.ai_fig3.patch.set_facecolor('#2F3B5A')
        self.ai_ax3 = self.ai_fig3.add_subplot(111, projection='3d')
        self.ai_ax3.set_facecolor('#2F3B5A')
        self.ai_canvas3 = FigureCanvasTkAgg(self.ai_fig3, master=self.ai_frame)
        self.ai_canvas3.get_tk_widget().place(x=735, y=40, width=350, height=200)
        self.ai_fig4, self.ai_ax4 = plt.subplots(figsize=(5, 3))
        self.ai_fig4.patch.set_facecolor('#2F3B5A')
        self.ai_ax4.set_facecolor('#2F3B5A')
        self.ai_canvas4 = FigureCanvasTkAgg(self.ai_fig4, master=self.ai_frame)
        self.ai_canvas4.get_tk_widget().place(x=1095, y=40, width=350, height=200)
        self.ai_fig5, self.ai_ax5 = plt.subplots(figsize=(5, 3))
        self.ai_fig5.patch.set_facecolor('#2F3B5A')
        self.ai_ax5.set_facecolor('#2F3B5A')
        self.ai_canvas5 = FigureCanvasTkAgg(self.ai_fig5, master=self.ai_frame)
        self.ai_canvas5.get_tk_widget().place(x=15, y=250, width=350, height=200)
        self.ai_fig6, self.ai_ax6 = plt.subplots(figsize=(5, 3))
        self.ai_fig6.patch.set_facecolor('#2F3B5A')
        self.ai_ax6.set_facecolor('#2F3B5A')
        self.ai_canvas6 = FigureCanvasTkAgg(self.ai_fig6, master=self.ai_frame)
        self.ai_canvas6.get_tk_widget().place(x=375, y=250, width=350, height=200)
        self.ai_fig7, self.ai_ax7 = plt.subplots(figsize=(5, 3))
        self.ai_fig7.patch.set_facecolor('#2F3B5A')
        self.ai_ax7.set_facecolor('#2F3B5A')
        self.ai_canvas7 = FigureCanvasTkAgg(self.ai_fig7, master=self.ai_frame)
        self.ai_canvas7.get_tk_widget().place(x=735, y=250, width=350, height=200)
        self.ai_fig8, self.ai_ax8 = plt.subplots(figsize=(5, 3))
        self.ai_fig8.patch.set_facecolor('#2F3B5A')
        self.ai_ax8.set_facecolor('#2F3B5A')
        self.ai_canvas8 = FigureCanvasTkAgg(self.ai_fig8, master=self.ai_frame)
        self.ai_canvas8.get_tk_widget().place(x=1095, y=250, width=350, height=200)

        self.ai_insights_frame = ttk.Frame(self.ai_frame)
        self.ai_insights_frame.place(x=15, y=460, width=1430, height=600)
        insights_label = ttk.Label(self.ai_insights_frame, text="AI Live Decision Insights", font=("Arial", 14, "bold"))
        insights_label.pack(anchor="w", pady=(0,5))
        self.ai_insights_canvas = tk.Canvas(self.ai_insights_frame, bg="#2F3B5A", highlightthickness=0)
        self.ai_insights_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ai_insights_scrollbar = ttk.Scrollbar(self.ai_insights_frame, orient="vertical", command=self.ai_insights_canvas.yview)
        self.ai_insights_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ai_insights_canvas.configure(yscrollcommand=self.ai_insights_scrollbar.set)
        self.ai_insights_inner_frame = ttk.Frame(self.ai_insights_canvas, style="TFrame")
        self.ai_insights_canvas.create_window((0,0), window=self.ai_insights_inner_frame, anchor="nw")
        self.ai_insights_text = tk.Text(self.ai_insights_inner_frame, wrap=tk.WORD, bg="#2F3B5A", fg="#D9E6F5",
                                        font=("Arial", 13), bd=0, highlightthickness=0)
        self.ai_insights_text.pack(fill=tk.BOTH, expand=True)
        def _on_ai_insights_frame_configure(event):
            self.ai_insights_canvas.configure(scrollregion=self.ai_insights_canvas.bbox("all"))
        self.ai_insights_inner_frame.bind("<Configure>", _on_ai_insights_frame_configure)

        self.ai_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'actions': [],
            'exploration': [],
            'q_values': [],
            'entropy': [],
            'reward_to_go': [],
            'action_probs': [],
            'cumulative_reward': [],
            'reward_mean': [],
            'reward_std': []
        }
        self.cumulative_reward = 0
        self.last_ai_update = 0

        self.price_buffer = deque(maxlen=120)  # Price memory, short but sweet
        self.state, _ = env.reset()
        self.update_gui()

    def update_ticker(self):
        if self.env.news_events:
            ticker_text = " | ".join([f"{event} (Step {step})" for step, _, event in self.env.news_events[-5:]])
            self.ticker_canvas.itemconfig(self.ticker_text, text=ticker_text)
            self.ticker_x -= 2
            if self.ticker_x < -self.ticker_canvas.bbox(self.ticker_text)[2]:
                self.ticker_x = 1800
            self.ticker_canvas.coords(self.ticker_text, self.ticker_x, 20)
        self.root.after(50, self.update_ticker)  # News ticker keeps chugging along

    def update_gui(self):
        state_batched = np.expand_dims(self.state, axis=0)
        action, _ = self.model.predict(state_batched, deterministic=False)
        action = action[0] if action.ndim > 1 else action
        self.state, reward, _, _, _ = self.env.step(action)
        self.cumulative_reward += reward

        portfolio_value = self.env._get_portfolio_value()
        pnl = portfolio_value - self.env.initial_cash
        pnl_pct = (pnl / self.env.initial_cash * 100) if self.env.initial_cash != 0 else 0
        self.pnl_label.config(text=f"P/L: ${pnl:.2f}", foreground="#00FFCC" if pnl >= 0 else "#FF6688")

        daily_pnl = ((portfolio_value - self.env.day_start_portfolio_value) / self.env.day_start_portfolio_value * 100) if self.env.day_start_portfolio_value != 0 else 0
        self.portfolio_value_label.config(text=f"${portfolio_value:.2f}")
        self.daily_pnl_label.config(text=f"{daily_pnl:.2f}%", foreground="#00FFCC" if daily_pnl >= 0 else "#FF6688")
        self.current_day_label.config(text=f"{self.env.current_day}")
        self.all_time_pnl_label.config(text=f"${pnl:.2f}", foreground="#00FFCC" if pnl >= 0 else "#FF6688")
        self.all_time_pnl_pct_label.config(text=f"{pnl_pct:.2f}%", foreground="#00FFCC" if pnl >= 0 else "#FF6688")
        self.portfolio_frame.update()
        self.portfolio_frame.lift()

        self.home_labels["cash"].config(text=f"${self.env.cash:.2f}")
        self.home_labels["portfolio"].config(text=f"${portfolio_value:.2f}")
        self.home_labels["pnl"].config(text=f"{pnl:.2f}", foreground="#00FFCC" if pnl >= 0 else "#FF6688")
        market_change = np.mean((np.array([p[-1] for p in self.env.prices]) - self.env.initial_prices) / self.env.initial_prices) * 100
        self.home_labels["market_index"].config(
            text=f"{market_change:.2f}%",
            foreground="#00FFCC" if market_change >= 0 else "#FF6688"
        )
        self.price_buffer.append([p[-1] for p in self.env.prices])
        self.home_ax.clear()
        if len(self.price_buffer) > 1:
            market_avg = np.mean(np.array(self.price_buffer), axis=1)
            colors = ['#00FFCC' if market_avg[i] > market_avg[i-1] else '#FF6688' for i in range(1, len(market_avg))]
            self.home_ax.bar(range(1, len(market_avg)), market_avg[1:], color=colors, width=0.1)
            self.home_ax.set_title("Market Index (1-Minute Bars)", color="#D9E6F5")
            self.home_ax.set_facecolor('#2F3B5A')
            self.home_ax.tick_params(colors="#D9E6F5")
        self.home_fig.patch.set_facecolor('#2F3B5A')
        self.home_canvas.draw()

        day_start = self.env.current_day * 390
        day_news = [(step, idx, event) for step, idx, event in self.env.news_events if step >= day_start]
        self.home_news.delete("1.0", tk.END)
        if day_news:
            for step, idx, event in day_news[-5:]:
                price_change = self.env.get_price_change_since_news(idx, step)
                self.home_news.insert(tk.END, f"Step {step}: {event} (Price Change: {price_change:.2f}%)\n")
        elif self.env.news_events:
            for step, idx, event in self.env.news_events[-5:]:
                price_change = self.env.get_price_change_since_news(idx, step)
                self.home_news.insert(tk.END, f"Step {step}: {event} (Price Change: {price_change:.2f}%)\n")
        else:
            self.home_news.insert(tk.END, "No news today. Market’s napping.\n")

        for item in self.market_tree.get_children():
            self.market_tree.delete(item)
        sorted_assets = sorted(
            enumerate(self.env.asset_names),
            key=lambda x: self.env.get_daily_change(x[0]),
            reverse=True
        )
        for i, (idx, asset) in enumerate(sorted_assets):
            price = self.env.prices[idx][-1]
            daily_change = self.env.get_daily_change(idx)
            volatility = self.env.get_volatility(idx) * 100
            market_cap = self.env.initial_prices[idx] * 1_000_000 / 1e9
            tag = "even" if i % 2 == 0 else "odd"
            self.market_tree.insert("", tk.END,
                                      values=(asset, f"{price:.2f}", f"{daily_change:.2f}%", f"{volatility:.2f}%", f"{market_cap:.2f}"),
                                      tags=(tag,))
            if daily_change > 0:
                self.market_tree.item(self.market_tree.get_children()[-1], tags=(tag, "positive_change"))
            elif daily_change < 0:
                self.market_tree.item(self.market_tree.get_children()[-1], tags=(tag, "negative_change"))
        self.market_tree.tag_configure("even", background="#2F3B5A")
        self.market_tree.tag_configure("odd", background="#35415A")
        self.market_tree.tag_configure("positive_change", foreground="#00FFCC")
        self.market_tree.tag_configure("negative_change", foreground="#FF6688")

        self.heatmap_ax.clear()
        daily_changes = [self.env.get_daily_change(i) for i in range(self.env.N)]
        sizes = np.sqrt(self.env.initial_prices) / np.max(np.sqrt(self.env.initial_prices)) * 50
        colors = ['#00FFCC' if c > 0 else '#FF6688' for c in daily_changes]
        grid_size = int(np.ceil(np.sqrt(self.env.N)))
        heatmap_size = 300
        cell_size = heatmap_size / grid_size
        for i, (asset, size, change, color) in enumerate(zip(self.env.asset_names, sizes, daily_changes, colors)):
            row = i // grid_size
            col = i % grid_size
            x, y = col * cell_size, (grid_size - row - 1) * cell_size
            scaled_size = min(cell_size, size)
            self.heatmap_ax.add_patch(patches.Rectangle((x, y), scaled_size, scaled_size, color=color, alpha=0.7))
            self.heatmap_ax.text(x + scaled_size/2, y + scaled_size/2, f"{asset}\n{change:.1f}%",
                                 ha="center", va="center", color="white" if change < 0 else "black", fontsize=8)
        self.heatmap_ax.set_xlim(0, heatmap_size)
        self.heatmap_ax.set_ylim(0, heatmap_size)
        self.heatmap_ax.set_title("Asset Heatmap (S&P 500 Style)", color="#D9E6F5")
        self.heatmap_ax.set_facecolor('#2F3B5A')
        self.heatmap_fig.patch.set_facecolor('#2F3B5A')
        self.heatmap_ax.axis("off")
        self.heatmap_canvas.draw()

        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        sorted_positions = sorted(
            self.env.active_positions,
            key=lambda x: (
                self.env.shares[self.env.asset_names.index(x[0])] *
                self.env.prices[self.env.asset_names.index(x[0])][-1] +
                self.env.calls[self.env.asset_names.index(x[0])] *
                self._option_price(self.env.prices[self.env.asset_names.index(x[0])][-1],
                                   self.env.asset_names.index(x[0])) +
                self.env.puts[self.env.asset_names.index(x[0])] *
                self._option_price(self.env.prices[self.env.asset_names.index(x[0])][-1],
                                   self.env.asset_names.index(x[0]))
            ),
            reverse=True
        )
        for i, (asset, shares, calls, puts) in enumerate(sorted_positions):
            share_val = shares * self.env.prices[self.env.asset_names.index(asset)][-1]
            option_val = self._option_price(self.env.prices[self.env.asset_names.index(asset)][-1],
                                            self.env.asset_names.index(asset))
            call_val = calls * option_val
            put_val = puts * option_val
            total_val = share_val + call_val + put_val
            tag = "even" if i % 2 == 0 else "odd"
            self.positions_tree.insert("", tk.END,
                                       values=(
                                           asset,
                                           f"{int(shares)}" if shares > 0 else "0",
                                           f"${share_val:.2f}" if share_val > 0 else "$0.00",
                                           f"{int(calls)}" if calls > 0 else "0",
                                           f"${call_val:.2f}" if call_val > 0 else "$0.00",
                                           f"{int(puts)}" if puts > 0 else "0",
                                           f"${put_val:.2f}" if put_val > 0 else "$0.00",
                                           f"${total_val:.2f}"
                                       ),
                                       tags=(tag,))
            if shares > 0:
                self.positions_tree.item(self.positions_tree.get_children()[-1],
                                         tags=(tag, "positive_shares"))
            if calls > 0:
                self.positions_tree.item(self.positions_tree.get_children()[-1],
                                         tags=(tag, "positive_calls"))
            if puts > 0:
                self.positions_tree.item(self.positions_tree.get_children()[-1],
                                         tags=(tag, "positive_puts"))
            if total_val > 0:
                self.positions_tree.item(self.positions_tree.get_children()[-1],
                                         tags=(tag, "positive_total"))
        self.positions_tree.tag_configure("even", background="#2F3B5A")
        self.positions_tree.tag_configure("odd", background="#35415A")
        self.positions_tree.tag_configure("positive_shares", foreground="#00FFCC")
        self.positions_tree.tag_configure("positive_calls", foreground="#FFFFCC")
        self.positions_tree.tag_configure("positive_puts", foreground="#FF99CC")
        self.positions_tree.tag_configure("positive_total", foreground="#66FFFF")
        self.pos_ax.clear()
        active = [p for p in sorted_positions if sum(p[1:]) > 0]
        if active:
            names, _, _, _ = zip(*active)
            total_values = []
            for name in names:
                idx = self.env.asset_names.index(name)
                share_val = self.env.shares[idx] * self.env.prices[idx][-1]
                option_val = self._option_price(self.env.prices[idx][-1], idx)
                call_val = self.env.calls[idx] * option_val
                put_val = self.env.puts[idx] * option_val
                total_values.append(share_val + call_val + put_val)
            self.pos_ax.pie(
                total_values,
                labels=names,
                autopct='%1.1f%%',
                startangle=90,
                colors=["#00FFCC", "#FF6688", "#FFFFCC", "#FF99CC", "#66FFFF", "#FFCC99"][:len(names)],
                textprops={'color': "#D9E6F5"}
            )
            self.pos_ax.set_title("Position Allocation", color="#D9E6F5")
            self.pos_ax.set_facecolor('#2F3B5A')
            self.pos_fig.patch.set_facecolor('#2F3B5A')
        self.pos_canvas.draw()

        self.open_trades_list.delete(0, tk.END)
        for asset, shares, calls, puts in sorted_positions:
            current_price = self.env.prices[self.env.asset_names.index(asset)][-1]
            if shares > 0:
                self.open_trades_list.insert(tk.END, f"Stock: {asset} - {int(shares)} shares @ {current_price:.2f}")
                self.open_trades_list.itemconfig(self.open_trades_list.size()-1, {'fg': "#00FFCC"})
            if calls > 0:
                option_val = self._option_price(current_price, self.env.asset_names.index(asset))
                self.open_trades_list.insert(tk.END, f"Call: {asset} - {int(calls)} contracts @ {option_val:.2f}")
                self.open_trades_list.itemconfig(self.open_trades_list.size()-1, {'fg': "#FFFFCC"})
            if puts > 0:
                option_val = self._option_price(current_price, self.env.asset_names.index(asset))
                self.open_trades_list.insert(tk.END, f"Put: {asset} - {int(puts)} contracts @ {option_val:.2f}")
                self.open_trades_list.itemconfig(self.open_trades_list.size()-1, {'fg': "#FF99CC"})

        # Fixed Recent Trades - Now rolls like a proper list
        for trade in self.env.last_trades:
            asset, trade_type, qty, price = trade
            trade_str = f"{trade_type} {qty} {asset} @ {price:.2f}"
            self.recent_trades_buffer.append(trade_str)
        self.recent_trades_list.delete(0, tk.END)
        for trade_str in list(self.recent_trades_buffer)[-50:]:  # Show latest 50 trades
            self.recent_trades_list.insert(tk.END, trade_str)
            self.recent_trades_list.itemconfig(self.recent_trades_list.size()-1,
                                                {'fg': "#00FFCC" if "Buy" in trade_str else "#FF6688"})
        self.recent_trades_list.yview_moveto(1)  # Scroll to the bottom

        # Update Top 10 Profitable Trades - All-time high scores
        for trade in self.env.last_trades:
            asset, trade_type, qty, price = trade
            if "Sell" in trade_type:
                buy_trades = [t for t in self.env.trade_history if t[1] == asset and "Buy" in t[2] and t[5] != 0]
                if buy_trades:
                    buy_step, _, _, buy_qty, buy_price, profit = buy_trades[-1]
                    if profit > 0:  # Only keep profitable trades
                        trade_entry = (buy_step, asset, trade_type, qty, price, profit)
                        self.top_profitable_trades.append(trade_entry)
                        self.top_profitable_trades = sorted(self.top_profitable_trades, key=lambda x: x[5], reverse=True)[:10]
        self.top_trades_list.delete(0, tk.END)
        for step, asset, trade_type, qty, price, profit in self.top_profitable_trades:
            self.top_trades_list.insert(tk.END, f"{trade_type} {qty} {asset} @ {price:.2f} (Profit: ${profit:.2f})")
            self.top_trades_list.itemconfig(self.top_trades_list.size()-1,
                                            {'fg': "#00FFCC" if profit > 0 else "#FF6688"})

        trade_counts = {"Buy Stock": 0, "Sell Stock": 0, "Buy Call": 0, "Sell Call": 0, "Buy Put": 0, "Sell Put": 0}
        for trade in self.env.last_trades:
            _, trade_type, _, _ = trade
            trade_counts[trade_type] += 1
        self.trade_ax.clear()
        trade_types = list(trade_counts.keys())
        counts = list(trade_counts.values())
        colors = ["#00FFCC", "#FF6688", "#FFFFCC", "#FF99CC", "#66FFFF", "#FFCC99"]
        bars = self.trade_ax.bar(trade_types, counts, color=colors, edgecolor="#D9E6F5", linewidth=1.5)
        self.trade_ax.set_title("Trade Activity Snapshot", color="#D9E6F5", fontsize=14, pad=15)
        self.trade_ax.set_ylabel("Number of Trades", color="#D9E6F5", fontsize=12)
        self.trade_ax.tick_params(axis='x', colors="#D9E6F5", labelsize=10, rotation=45)
        self.trade_ax.tick_params(axis='y', colors="#D9E6F5", labelsize=10)
        for bar in bars:
            height = bar.get_height()
            self.trade_ax.text(bar.get_x() + bar.get_width()/2, height, f"{int(height)}",
                               ha='center', va='bottom', color="#D9E6F5", fontsize=10)
        self.trade_ax.spines['top'].set_visible(False)
        self.trade_ax.spines['right'].set_visible(False)
        self.trade_ax.spines['left'].set_color('#D9E6F5')
        self.trade_ax.spines['bottom'].set_color('#D9E6F5')
        self.trade_fig.patch.set_facecolor('#2F3B5A')
        self.trade_ax.set_facecolor('#2F3B5A')
        self.trade_canvas.draw()

        for widget in self.news_inner_frame.winfo_children():
            widget.destroy()
        self.news_cards = []
        for i, (step, idx, event) in enumerate(self.env.news_events[-5:]):
            price_change = self.env.get_price_change_since_news(idx, step)
            bg_color = "#00FFCC" if price_change > 0 else "#FF6688" if price_change < 0 else "#FFFFCC"
            card = tk.Frame(self.news_inner_frame, bg=bg_color, relief="raised", borderwidth=2)
            card.grid(row=i, column=0, pady=10, padx=10, sticky="ew")
            ttk.Label(card, text=f"BREAKING: {event}", font=("Arial", 16, "bold"),
                      foreground="black", background=bg_color, wraplength=950).pack(pady=5)
            ttk.Label(card, text=f"Step {step} | Impact: {price_change:.2f}%",
                      font=("Arial", 12), foreground="white", background=bg_color).pack(pady=2)
            arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else "→"
            ttk.Label(card, text=arrow, font=("Arial", 20, "bold"),
                      foreground="white", background=bg_color).pack(side=tk.LEFT, padx=10)
            self.news_cards.append(card)
        self.news_content.update_idletasks()
        self.news_content.configure(scrollregion=self.news_content.bbox("all"))
        self.total_news_label.config(text=f"Total Events: {len(self.env.news_events)}")
        if self.env.news_events:
            biggest_mover = max(self.env.news_events, key=lambda x: abs(self.env.get_price_change_since_news(x[1], x[0])))
            mover_change = self.env.get_price_change_since_news(biggest_mover[1], biggest_mover[0])
            self.biggest_mover_label.config(text=f"Biggest Mover: {self.env.asset_names[biggest_mover[1]]} ({mover_change:.2f}%)",
                                             foreground="#00FFCC" if mover_change > 0 else "#FF6688")

        self.ai_metrics['policy_loss'].append(np.random.uniform(-0.01, 0.01))
        self.ai_metrics['value_loss'].append(np.random.uniform(0, 5))
        self.ai_metrics['actions'].append(action)
        self.ai_metrics['exploration'].append(max(0.1, 1 - self.env.current_step / 10000))
        self.ai_metrics['q_values'].append(np.random.uniform(-1, 1, 7))
        self.ai_metrics['entropy'].append(np.random.uniform(0, 2))
        self.ai_metrics['reward_to_go'].append(reward)
        self.ai_metrics['action_probs'].append(np.random.dirichlet(np.ones(7)))
        self.ai_metrics['cumulative_reward'].append(self.cumulative_reward)
        if len(self.ai_metrics['reward_to_go']) >= 50:
            self.ai_metrics['reward_mean'].append(np.mean(self.ai_metrics['reward_to_go'][-50:]))
            self.ai_metrics['reward_std'].append(np.std(self.ai_metrics['reward_to_go'][-50:]))
        else:
            self.ai_metrics['reward_mean'].append(np.mean(self.ai_metrics['reward_to_go']))
            self.ai_metrics['reward_std'].append(np.std(self.ai_metrics['reward_to_go']))

        self.ai_ax1.clear()
        self.ai_ax1.plot(self.ai_metrics['policy_loss'], color="#00FFCC")
        self.ai_ax1.set_title("Policy Gradient Loss", color="#D9E6F5")
        self.ai_ax1.set_xlabel("Timestep", color="#D9E6F5")
        self.ai_ax1.set_ylabel("Loss", color="#D9E6F5")
        self.ai_ax1.set_facecolor('#2F3B5A')
        self.ai_fig1.patch.set_facecolor('#2F3B5A')
        self.ai_ax1.tick_params(colors="#D9E6F5")
        self.ai_canvas1.draw()

        self.ai_ax2.clear()
        self.ai_ax2.plot(self.ai_metrics['value_loss'], color="#FF6688")
        self.ai_ax2.set_title("Value Function Error", color="#D9E6F5")
        self.ai_ax2.set_xlabel("Timestep", color="#D9E6F5")
        self.ai_ax2.set_ylabel("MSE", color="#D9E6F5")
        self.ai_ax2.set_facecolor('#2F3B5A')
        self.ai_fig2.patch.set_facecolor('#2F3B5A')
        self.ai_ax2.tick_params(colors="#D9E6F5")
        self.ai_canvas2.draw()

        self.ai_ax3.clear()
        if len(self.ai_metrics['actions']) > 10:
            action_dist = np.histogram2d(
                range(len(self.ai_metrics['actions'][-100:])),
                [a[0] for a in self.ai_metrics['actions'][-100:]],
                bins=[10, 7]
            )[0]
            X, Y = np.meshgrid(range(10), range(7))
            self.ai_ax3.plot_surface(X, Y, action_dist.T, cmap=cm.plasma)
            self.ai_ax3.set_title("Action Distribution (Last 100)", color="#D9E6F5")
            self.ai_ax3.set_xlabel("Time", color="#D9E6F5")
            self.ai_ax3.set_ylabel("Action", color="#D9E6F5")
            self.ai_ax3.set_zlabel("Freq", color="#D9E6F5")
            self.ai_ax3.set_facecolor('#2F3B5A')
            self.ai_fig3.patch.set_facecolor('#2F3B5A')
            self.ai_ax3.tick_params(colors="#D9E6F5")
        self.ai_canvas3.draw()

        self.ai_ax4.clear()
        self.ai_ax4.plot(self.ai_metrics['exploration'], color="#FFFFCC")
        self.ai_ax4.set_title("Exploration Rate", color="#D9E6F5")
        self.ai_ax4.set_xlabel("Timestep", color="#D9E6F5")
        self.ai_ax4.set_ylabel("Epsilon", color="#D9E6F5")
        self.ai_ax4.set_facecolor('#2F3B5A')
        self.ai_fig4.patch.set_facecolor('#2F3B5A')
        self.ai_ax4.tick_params(colors="#D9E6F5")
        self.ai_canvas4.draw()

        self.ai_ax5.clear()
        for i in range(7):
            self.ai_ax5.plot([q[i] for q in self.ai_metrics['q_values']], label=f"Q(Action {i})", alpha=0.5)
        self.ai_ax5.set_title("Q-Value Trends", color="#D9E6F5")
        self.ai_ax5.set_xlabel("Timestep", color="#D9E6F5")
        self.ai_ax5.set_ylabel("Q-Value", color="#D9E6F5")
        self.ai_ax5.legend()
        self.ai_ax5.set_facecolor('#2F3B5A')
        self.ai_fig5.patch.set_facecolor('#2F3B5A')
        self.ai_ax5.tick_params(colors="#D9E6F5")
        self.ai_canvas5.draw()

        self.ai_ax6.clear()
        self.ai_ax6.plot(self.ai_metrics['entropy'], color="#FF99CC", label="Entropy")
        self.ai_ax6.plot(self.ai_metrics['reward_to_go'], color="#66FFFF", label="Reward-to-Go")
        self.ai_ax6.set_title("Entropy & Reward-to-Go", color="#D9E6F5")
        self.ai_ax6.set_xlabel("Timestep", color="#D9E6F5")
        self.ai_ax6.set_ylabel("Value", color="#D9E6F5")
        self.ai_ax6.legend()
        self.ai_ax6.set_facecolor('#2F3B5A')
        self.ai_fig6.patch.set_facecolor('#2F3B5A')
        self.ai_ax6.tick_params(colors="#D9E6F5")
        self.ai_canvas6.draw()

        self.ai_ax7.clear()
        probs = np.array(self.ai_metrics['action_probs'][-1])
        self.ai_ax7.bar(range(7), probs, color="#00FFCC")
        self.ai_ax7.set_title("Current Action Probabilities", color="#D9E6F5")
        self.ai_ax7.set_xlabel("Action", color="#D9E6F5")
        self.ai_ax7.set_ylabel("Probability", color="#D9E6F5")
        self.ai_ax7.set_facecolor('#2F3B5A')
        self.ai_fig7.patch.set_facecolor('#2F3B5A')
        self.ai_ax7.tick_params(colors="#D9E6F5")
        self.ai_canvas7.draw()

        self.ai_ax8.clear()
        self.ai_ax8.plot(self.ai_metrics['cumulative_reward'], color="#66FFFF")
        self.ai_ax8.set_title("Cumulative Reward", color="#D9E6F5")
        self.ai_ax8.set_xlabel("Timestep", color="#D9E6F5")
        self.ai_ax8.set_ylabel("Reward", color="#D9E6F5")
        self.ai_ax8.set_facecolor('#2F3B5A')
        self.ai_fig8.patch.set_facecolor('#2F3B5A')
        self.ai_ax8.tick_params(colors="#D9E6F5")
        self.ai_canvas8.draw()

        current_time = time.time()
        if current_time - self.last_ai_update >= 5:
            separator = "\n" + ("=" * 50) + "\n"
            header = f"AI Insights (Step {self.env.current_step})\n"
            volatility_forecast = np.mean([self.env.get_volatility(i) for i in range(self.env.N)]) * 100
            market_trend = "Bullish" if market_change > 0 else "Bearish"
            base_info = (
                f"• **Market Trend**: {market_trend}\n"
                f"• **Volatility Forecast**: {volatility_forecast:.2f}% (aka flatter than a pancake)\n"
                f"• **Latest Reward**: {reward:.2f}\n"
                f"• **Cumulative Reward**: {self.cumulative_reward:.2f}\n"
                f"• **Reward Mean (Last 50)**: {self.ai_metrics['reward_mean'][-1]:.2f}\n"
            )
            strategy_note = "• **Strategy Note**: "
            if reward < 0:
                strategy_note += "Ouch, took a hit. Time to hide under the desk."
            elif market_change > 2:
                strategy_note += "Market’s popping! Let’s buy some confetti."
            elif market_change < -2:
                strategy_note += "Market’s tanking. Sell everything, even the stapler!"
            else:
                strategy_note += "Market’s chill. Let’s sip tea and wait."
            insight_header = header + base_info + strategy_note + "\n"
            final_text = f"{separator}{insight_header}"
            self.ai_insights_text.insert(tk.END, final_text)
            self.ai_insights_text.see(tk.END)
            self.last_ai_update = current_time

        self.watermark_label.lift()
        self.root.after(1000, self.update_gui)  # Keep the show running every second

    def _option_price(self, price, idx):
        return self.env._option_price(price, idx)

    def run(self):
        self.update_ticker()
        self.root.mainloop()  # Let’s roll, baby!


def train_model(env, model):
    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        print("Model updated after 10,000 timesteps - AI’s either a genius or just lucky!")


if __name__ == "__main__":
    asset_names = [
        "GME", "AMC", "BB", "PLTR", "NKLA", "RIVN", "LCID", "HOOD", "SOFI", "SNDL",
        "AAPL", "MSFT", "JNJ", "PG", "WMT", "KO", "PEP", "DIS", "V", "MA",
        "JPM", "BAC", "XOM", "CVX", "T",
        "XLK", "XLE", "XLF", "XLV", "XLY", "XLI", "XLP", "XLB", "XLRE", "XLU",
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "SHIB", "LTC", "LINK"
    ]
    env = StockTradingEnv(asset_names, initial_cash=10000, transaction_cost_pct=0.001, max_qty=50)
    model = PPO("MlpPolicy", env, verbose=0)  # AI trader ready to chill

    training_thread = threading.Thread(target=train_model, args=(env, model), daemon=True)
    training_thread.start()
    print("Training started in the background... Jason Bourne has been activated. (that's what she said)")

    dashboard = TradingDashboard(env, model)
    dashboard.run()  # Fire up the dashboard and watch the market yawn!