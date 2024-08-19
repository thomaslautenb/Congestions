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

# 2. Wind and Solar Data with Bottleneck Highlight
bottleneck_period = engpass_reports.iloc[0]["Datum"]
plt.figure(figsize=(14, 7))
plt.plot(wind_data.index, wind_data["Region_A"], label="Wind Energy (Region A)", color="green")
plt.axvspan(bottleneck_period - pd.Timedelta(hours=6), bottleneck_period + pd.Timedelta(hours=6), color='orange', alpha=0.3, label="Bottleneck Period")
plt.plot(solar_data.index, solar_data["Region_A"], label="Solar Energy (Region A)", color="gold")
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

# Step 2: Visualization of Line Loadings Before Bottlenecks

# Plot line loadings for each bottleneck event
for bottleneck_id, line_loadings in line_loadings_before_bottlenecks.items():
    plt.figure(figsize=(14, 7))
    
    for line in line_loadings.columns:
        plt.plot(line_loadings.index, line_loadings[line], label=f"Line {line}")
        plt.axhline(y=line_capacities[lines.index(line)], color='r', linestyle='--', label=f"Capacity of {line}")
    
    plt.title(f"Line Loadings Before Bottleneck Event {bottleneck_id}", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Loading (MW)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()