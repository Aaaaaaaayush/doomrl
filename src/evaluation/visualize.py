import argparse
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_metrics(metrics_file):
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    with open(metrics_file, 'r') as f:
        return json.load(f)

def generate_interactive_charts(metrics_file, output_dir="frontend/charts"):
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_metrics(metrics_file)
    scenario_name = data["scenario"]
    metrics = data["metrics"]
    
    episodes = metrics["episodes"]
    rewards = metrics["episode_rewards"]
    mean_rewards_100 = metrics["mean_rewards_100"]
    losses = metrics["losses"]
    epsilons = metrics["epsilons"]
    kill_counts = metrics["kill_counts"]
    
    # Theme Styling Constants
    grid_color = "#222222"
    paper_bg = "rgba(10, 10, 10, 0.9)"
    plot_bg = "rgba(10, 10, 10, 0.9)"
    font_family = "'Share Tech Mono', 'Courier New', monospace"
    text_color = "#e0e0e0"
    
    # ── CHART 1: REWARD CURVE (Neon Green / Neon Red) ───────────────────────
    fig_reward = go.Figure()
    # Raw Reward (scatter)
    fig_reward.add_trace(go.Scatter(
        x=episodes, y=rewards,
        mode='lines',
        name='Raw Reward',
        line=dict(color='rgba(255, 36, 0, 0.25)', width=1) # doom-red transparent
    ))
    # Rolling Mean
    fig_reward.add_trace(go.Scatter(
        x=episodes, y=mean_rewards_100,
        mode='lines',
        name='100-Ep Rolling Mean',
        line=dict(color='#00FF41', width=2.5) # terminal-green
    ))
    fig_reward.update_layout(
        title=f"TRAINING REWARD OVER TIME - {scenario_name.upper()}",
        xaxis_title="Episodes",
        yaxis_title="Total Reward",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=text_color),
        grid=dict(rows=1, columns=1),
        xaxis=dict(gridcolor=grid_color, linecolor="#00FF41"),
        yaxis=dict(gridcolor=grid_color, linecolor="#00FF41"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,10,10,0.8)")
    )
    
    # ── CHART 2: LOSS CURVE (Orange / Smooth) ──────────────────────────────
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=episodes, y=losses,
        mode='lines',
        name='Smooth L1 Loss',
        line=dict(color='#FF5722', width=1.5)
    ))
    fig_loss.update_layout(
        title=f"TEMPORAL DIFFERENCE LOSS - {scenario_name.upper()}",
        xaxis_title="Episodes",
        yaxis_title="Loss",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=text_color),
        xaxis=dict(gridcolor=grid_color, linecolor="#FF5722"),
        yaxis=dict(gridcolor=grid_color, linecolor="#FF5722")
    )
    
    # ── CHART 3: KILL COUNT (Blue / Histograms) ────────────────────────────
    fig_kills = go.Figure()
    # Compute simple rolling mean for kills
    roll_window = 100
    rolling_kills = [np.mean(kill_counts[max(0, i - roll_window):i + 1]) for i in range(len(kill_counts))] if 'np' in globals() else []
    if len(rolling_kills) == 0:
        import numpy as np
        rolling_kills = [float(np.mean(kill_counts[max(0, i - roll_window):i + 1])) for i in range(len(kill_counts))]
        
    fig_kills.add_trace(go.Scatter(
        x=episodes, y=kill_counts,
        mode='markers',
        name='Raw Kills',
        marker=dict(color='rgba(0, 191, 255, 0.3)', size=4)
    ))
    fig_kills.add_trace(go.Scatter(
        x=episodes, y=rolling_kills,
        mode='lines',
        name='100-Ep Avg Kills',
        line=dict(color='#00BFFF', width=2)
    ))
    fig_kills.update_layout(
        title=f"KILL COUNT PROGRESSION - {scenario_name.upper()}",
        xaxis_title="Episodes",
        yaxis_title="Kills per Episode",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=text_color),
        xaxis=dict(gridcolor=grid_color, linecolor="#00BFFF"),
        yaxis=dict(gridcolor=grid_color, linecolor="#00BFFF")
    )
    
    # ── CHART 4: EPSILON DECAY ─────────────────────────────────────────────
    fig_epsilon = go.Figure()
    fig_epsilon.add_trace(go.Scatter(
        x=episodes, y=epsilons,
        mode='lines',
        name='Epsilon (Explore Rate)',
        line=dict(color='#E0E000', width=2)
    ))
    fig_epsilon.update_layout(
        title=f"EPSILON DECAY CURVE (EXPLORE VS EXPLOIT) - {scenario_name.upper()}",
        xaxis_title="Episodes",
        yaxis_title="Epsilon Value",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=text_color),
        xaxis=dict(gridcolor=grid_color, linecolor="#E0E000"),
        yaxis=dict(gridcolor=grid_color, linecolor="#E0E000")
    )
    
    # Save files as HTML
    fig_reward.write_html(os.path.join(output_dir, f"{scenario_name}_reward.html"), include_plotlyjs="cdn")
    fig_loss.write_html(os.path.join(output_dir, f"{scenario_name}_loss.html"), include_plotlyjs="cdn")
    fig_kills.write_html(os.path.join(output_dir, f"{scenario_name}_kills.html"), include_plotlyjs="cdn")
    fig_epsilon.write_html(os.path.join(output_dir, f"{scenario_name}_epsilon.html"), include_plotlyjs="cdn")
    
    print(f"[OK] Plotly HTML charts generated successfully inside {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Interactive Plotly Charts from MLflow JSON metrics")
    parser.add_argument("--metrics", type=str, required=True, help="Path to JSON metrics file")
    args = parser.parse_args()
    
    generate_interactive_charts(args.metrics)
