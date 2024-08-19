import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.dates as mdates

# Data Simulation

# Step 1: Generating the data
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", end="2023-12-31 23:00", freq="H")
regions = ["Region_A", "Region_B", "Region_C", "Region_D"]

load_data = {region: np.random.normal(loc=500, scale=100, size=len(date_range)) for region in regions}
wind_data = {region: np.random.normal(loc=200, scale=50, size=len(date_range)) for region in regions}
solar_data = {region: np.random.normal(loc=100, scale=30, size=len(date_range)) for region in regions}

net_data = pd.DataFrame(load_data, index=date_range)
wind_data = pd.DataFrame(wind_data, index=date_range)
solar_data = pd.DataFrame(solar_data, index=date_range)

outlier_indices = np.random.choice(net_data.index, size=int(0.01 * len(net_data)), replace=False)
net_data.loc[outlier_indices, :] *= np.random.uniform(1.5, 2.0, size=(len(outlier_indices), len(regions)))
wind_data.loc[outlier_indices, :] *= np.random.uniform(1.5, 2.0, size=(len(outlier_indices), len(regions)))
solar_data.loc[outlier_indices, :] *= np.random.uniform(1.5, 2.0, size=(len(outlier_indices), len(regions)))

# Power Plants Data
power_plants = pd.DataFrame({
    "Kraftwerk": ["Plant_1", "Plant_2", "Plant_3", "Plant_4", "Plant_5"],
    "Region": ["Region_A", "Region_B", "Region_C", "Region_D", "Region_B"],
    "Max_Leistung_MW": [300, 250, 200, 150, 100],
    "Variable_Kosten_Euro_per_MWh": [50, 60, 55, 70, 65],
})

# Bottleneck Reports
engpass_dates = np.random.choice(date_range, size=100, replace=False)
engpass_regions = np.random.choice(regions, size=100, replace=True)
engpass_reports = pd.DataFrame({
    "Datum": engpass_dates,
    "Region": engpass_regions,
    "Engpass_MW": np.random.randint(50, 150, size=100),
})

# Transmission Line Data
lines = [("Region_A", "Region_B"), ("Region_B", "Region_C"), ("Region_C", "Region_D"), ("Region_D", "Region_A"), ("Region_A", "Region_C")]
line_capacities = [500, 400, 450, 300, 350]
line_utilization = {line: np.random.uniform(0.5, 1.2, size=len(date_range)) * capacity for line, capacity in zip(lines, line_capacities)}
line_data = pd.DataFrame(line_utilization, index=date_range)

# Network Graph
G = nx.Graph()
G.add_edges_from(lines)
pos = nx.circular_layout(G)

# Step 2: Data Analysis

engpass_count = engpass_reports['Region'].value_counts()
engpass_times = engpass_reports['Datum']
load_during_engpass = net_data.loc[engpass_times].mean()
wind_during_engpass = wind_data.loc[engpass_times].mean()
solar_during_engpass = solar_data.loc[engpass_times].mean()

redispatch_usage = {
    "Plant_1": engpass_reports[engpass_reports['Region'] == "Region_A"]['Engpass_MW'].sum(),
    "Plant_2": engpass_reports[engpass_reports['Region'] == "Region_B"]['Engpass_MW'].sum(),
    "Plant_3": engpass_reports[engpass_reports['Region'] == "Region_C"]['Engpass_MW'].sum(),
    "Plant_4": engpass_reports[engpass_reports['Region'] == "Region_D"]['Engpass_MW'].sum(),
    "Plant_5": engpass_reports[engpass_reports['Region'] == "Region_B"]['Engpass_MW'].sum() * 0.5,
}

redispatch_costs = {plant: usage * power_plants.loc[power_plants['Kraftwerk'] == plant, 'Variable_Kosten_Euro_per_MWh'].values[0] for plant, usage in redispatch_usage.items()}

outliers_detected = (net_data.loc[outlier_indices] > net_data.mean() + 3 * net_data.std()) | (net_data.loc[outlier_indices] < net_data.mean() - 3 * net_data.std())

line_utilization_during_engpass = line_data.loc[engpass_times].mean()
overloaded_lines = line_utilization_during_engpass[line_utilization_during_engpass > line_capacities]

# Step 3: Visualization

# 1. Load Data with Outliers
plt.figure(figsize=(14, 7))
plt.plot(net_data.index, net_data["Region_A"], label="Load (Region A)", color="blue")
plt.scatter(net_data.loc[outlier_indices].index, net_data.loc[outlier_indices, "Region_A"], color="red", label="Outliers", s=15)
plt.title("Load Data for Region A with Outliers Highlighted", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Load (MW)", fontsize=14)
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.show()

# Step 2: Wind and Solar Data with Enhanced Bottleneck Highlight
bottleneck_period = engpass_reports.iloc[0]["Datum"]
plt.figure(figsize=(14, 7))

# Plot Wind Data
plt.plot(wind_data.index, wind_data["Region_A"], label="Wind Energy (Region A)", color="green")

# Highlight Bottleneck Period with increased opacity and border
plt.axvspan(bottleneck_period - pd.Timedelta(hours=6), bottleneck_period + pd.Timedelta(hours=6), 
            color='orange', alpha=0.5, label="Bottleneck Period", edgecolor='red', linestyle='--')

# Plot Solar Data
plt.plot(solar_data.index, solar_data["Region_A"], label="Solar Energy (Region A)", color="gold")

# Annotate the bottleneck period
plt.annotate('Bottleneck Period', xy=(bottleneck_period, max(wind_data["Region_A"].max(), solar_data["Region_A"].max())),
             xytext=(bottleneck_period + pd.Timedelta(days=10), max(wind_data["Region_A"].max(), solar_data["Region_A"].max()) + 50),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')

plt.title("Wind and Solar Data for Region A with Bottleneck Period Highlighted", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Energy (MW)", fontsize=14)
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.show()

# 3. Network Graph with Line Capacities
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=15)
labels = {line: f"{cap} MW" for line, cap in zip(lines, line_capacities)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Network Graph with Transmission Line Capacities", fontsize=16)
plt.show()

# 4. Redispatch Costs per Power Plant
plt.figure(figsize=(10, 6))
plt.bar(redispatch_costs.keys(), redispatch_costs.values(), color='teal')
plt.title("Redispatch Costs per Power Plant", fontsize=16)
plt.xlabel("Power Plant", fontsize=14)
plt.ylabel("Cost (Euro)", fontsize=14)
plt.grid(True)
plt.show()

# Step 1: Analyze Line Loadings Before Each Bottleneck

# Define a time window before each bottleneck to examine
time_window_before_bottleneck = pd.Timedelta(hours=6)

# Prepare a dictionary to store line loadings before each bottleneck
line_loadings_before_bottlenecks = {}

# Iterate through each bottleneck event
for index, row in engpass_reports.iterrows():
    bottleneck_time = row['Datum']
    region = row['Region']
    
    # Find lines connected to the bottleneck region
    affected_lines = [line for line in lines if region in line]
    
    # Get line loadings in the time window before the bottleneck
    start_time = bottleneck_time - time_window_before_bottleneck
    end_time = bottleneck_time
    
    # Store the loadings
    line_loadings_before_bottlenecks[index] = line_data.loc[start_time:end_time, affected_lines]

# Step 1: Identify the time just before the bottleneck happens (e.g., 1 hour before)
bottleneck_index = 0  # You can change this to any other bottleneck event
bottleneck_time = engpass_reports.iloc[bottleneck_index]["Datum"]
time_before_bottleneck = bottleneck_time - pd.Timedelta(hours=1)

# Step 2: Get the line loadings at that specific time
line_loadings_at_time = line_data.loc[time_before_bottleneck]

# Step 3: Plot the Network Graph with Line Loadings

plt.figure(figsize=(10, 8))

# Draw the network graph
pos = nx.circular_layout(G)
edges = G.edges()

# Map the line loadings to the edges in the graph
edge_colors = []
edge_widths = []

for line in lines:
    loading = line_loadings_at_time[line]
    capacity = line_capacities[lines.index(line)]
    
    # Normalize loading to determine the edge width (scale it appropriately)
    edge_width = 5 * (loading / capacity)
    edge_widths.append(edge_width)
    
    # Color based on loading (green to red)
    if loading < 0.7 * capacity:
        edge_color = 'green'
    elif loading < 1.0 * capacity:
        edge_color = 'orange'
    else:
        edge_color = 'red'
    
    edge_colors.append(edge_color)

# Draw the graph with varying edge widths and colors
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=15, 
        edge_color=edge_colors, width=edge_widths)

plt.title(f"Network Graph with Line Loadings Before Bottleneck at {time_before_bottleneck}", fontsize=16)
plt.show()