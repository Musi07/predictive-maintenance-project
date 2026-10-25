# ============================================================
# LAB 1: INTRODUCTION TO PYTHON FOR DATA MINING
# Topic: Predictive Maintenance
# ============================================================

# ------------------------------------------------------------
# Section 1: Variables and Data Types
# ------------------------------------------------------------

machine_name = "AI4I Predictive Maintenance"
machine_id = 1001
temperature = 300.5
machine_operational = True

print("Machine Name:", machine_name)
print("Machine ID:", machine_id)
print("Temperature:", temperature)
print("Machine Operational:", machine_operational)

print("\nData Types:")
print(type(machine_name))
print(type(machine_id))
print(type(temperature))
print(type(machine_operational))


# ------------------------------------------------------------
# Section 2: Operators
# ------------------------------------------------------------

air_temperature = 300.5
process_temperature = 310.2
tool_wear = 120

temperature_difference = process_temperature - air_temperature
total_temperature = air_temperature + process_temperature
wear_after_10_minutes = tool_wear + 10

print("\nOperators:")
print("Temperature Difference:", temperature_difference)
print("Total Temperature:", total_temperature)
print("Tool Wear After 10 Minutes:", wear_after_10_minutes)


# ------------------------------------------------------------
# Section 3: Lists
# ------------------------------------------------------------

machine_features = [
    "Type",
    "Air Temperature",
    "Process Temperature",
    "Rotational Speed",
    "Torque",
    "Tool Wear"
]

print("\nMachine Features:")
print(machine_features)

print("First Feature:", machine_features[0])
print("Last Feature:", machine_features[-1])


# ------------------------------------------------------------
# Section 4: Tuples
# ------------------------------------------------------------

machine_status = ("Normal", "Warning", "Failure")

print("\nMachine Status:")
print(machine_status)

print("First Status:", machine_status[0])


# ------------------------------------------------------------
# Section 5: Dictionary
# ------------------------------------------------------------

machine = {
    "type": "M",
    "air_temperature": 300.5,
    "process_temperature": 310.2,
    "rotational_speed": 1500,
    "torque": 42.0,
    "tool_wear": 120
}

print("\nMachine Information:")
print(machine)

print("Machine Type:", machine["type"])
print("Rotational Speed:", machine["rotational_speed"])


# ------------------------------------------------------------
# Section 6: Conditional Statement
# ------------------------------------------------------------

if machine["tool_wear"] > 100:
    print("\nWarning: Tool wear is high.")
else:
    print("\nTool wear is within normal range.")


# ------------------------------------------------------------
# Section 7: For Loop
# ------------------------------------------------------------

print("\nMachine Features Using Loop:")

for feature in machine_features:
    print("-", feature)


# ------------------------------------------------------------
# Section 8: Function
# ------------------------------------------------------------

def calculate_temperature_difference(air_temp, process_temp):
    return process_temp - air_temp


difference = calculate_temperature_difference(
    air_temperature,
    process_temperature
)

print("\nFunction Result:")
print("Temperature Difference:", difference)

# ------------------------------------------------------------
# Section 9: Reading CSV File Using Pandas
# ------------------------------------------------------------

import pandas as pd

# Load the Predictive Maintenance dataset
df = pd.read_csv("data/ai4i2020.csv")

# Display the first five rows
print("\nFirst Five Rows of the Dataset:")
print(df.head())

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display data types
print("\nDataset Information:")
print(df.info())

# ------------------------------------------------------------
# Section 10: Writing Data to a New CSV File
# ------------------------------------------------------------

# Save a copy of the dataset
df.to_csv("data/ai4i2020_lab1_copy.csv", index=False)

print("\nDataset successfully saved to:")
print("data/ai4i2020_lab1_copy.csv")