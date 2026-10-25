# ============================================================
# LAB 4: FREQUENT PATTERN MINING - APRIORI ALGORITHM
# Topic: Predictive Maintenance
# ============================================================

import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ============================================================
# PART 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")

print("=" * 70)
print("LAB 4: FREQUENT PATTERN MINING - APRIORI")
print("=" * 70)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# PART 2: CREATE MACHINE CONDITION CATEGORIES
# ============================================================

# Create a copy for Apriori preprocessing
apriori_df = df.copy()

# ------------------------------------------------------------
# Temperature conditions
# ------------------------------------------------------------

temperature_threshold = apriori_df["Air temperature [K]"].median()

apriori_df["High Air Temperature"] = (
    apriori_df["Air temperature [K]"] > temperature_threshold
)

# ------------------------------------------------------------
# Process temperature conditions
# ------------------------------------------------------------

process_temperature_threshold = (
    apriori_df["Process temperature [K]"].median()
)

apriori_df["High Process Temperature"] = (
    apriori_df["Process temperature [K]"]
    > process_temperature_threshold
)

# ------------------------------------------------------------
# Rotational speed conditions
# ------------------------------------------------------------

speed_threshold = apriori_df["Rotational speed [rpm]"].median()

apriori_df["High Rotational Speed"] = (
    apriori_df["Rotational speed [rpm]"] > speed_threshold
)

# ------------------------------------------------------------
# Torque conditions
# ------------------------------------------------------------

torque_threshold = apriori_df["Torque [Nm]"].median()

apriori_df["High Torque"] = (
    apriori_df["Torque [Nm]"] > torque_threshold
)

# ------------------------------------------------------------
# Tool wear conditions
# ------------------------------------------------------------

tool_wear_threshold = apriori_df["Tool wear [min]"].median()

apriori_df["High Tool Wear"] = (
    apriori_df["Tool wear [min]"] > tool_wear_threshold
)


# ============================================================
# PART 3: CREATE TRANSACTION DATA
# ============================================================

# Each machine becomes a transaction containing
# the conditions associated with that machine.

transactions = []

for _, row in apriori_df.iterrows():

    transaction = []

    # Machine type
    transaction.append("Type_" + str(row["Type"]))

    # Temperature
    if row["High Air Temperature"]:
        transaction.append("High_Air_Temperature")

    # Process temperature
    if row["High Process Temperature"]:
        transaction.append("High_Process_Temperature")

    # Rotational speed
    if row["High Rotational Speed"]:
        transaction.append("High_Rotational_Speed")

    # Torque
    if row["High Torque"]:
        transaction.append("High_Torque")

    # Tool wear
    if row["High Tool Wear"]:
        transaction.append("High_Tool_Wear")

    # Machine failure
    if row["Machine failure"] == 1:
        transaction.append("Machine_Failure")
    else:
        transaction.append("No_Machine_Failure")

    transactions.append(transaction)


print("\nTransactions created successfully!")

print("\nFirst 5 Transactions:")

for transaction in transactions[:5]:
    print(transaction)


# ============================================================
# PART 4: CONVERT TRANSACTIONS INTO ONE-HOT DATA
# ============================================================

# Get all unique items
all_items = sorted(
    set(item for transaction in transactions for item in transaction)
)

# Create an empty transaction matrix
transaction_matrix = pd.DataFrame(
    False,
    index=range(len(transactions)),
    columns=all_items
)

# Fill transaction matrix
for i, transaction in enumerate(transactions):

    for item in transaction:
        transaction_matrix.loc[i, item] = True


print("\nTransaction Matrix Shape:")
print(transaction_matrix.shape)

print("\nTransaction Matrix:")
print(transaction_matrix.head())


# ============================================================
# PART 5: APRIORI - FIND FREQUENT ITEMSETS
# ============================================================

print("\n" + "=" * 70)
print("PART 5: APRIORI FREQUENT ITEMSETS")
print("=" * 70)

# Minimum support
min_support = 0.10

frequent_itemsets = apriori(
    transaction_matrix,
    min_support=min_support,
    use_colnames=True
)

# Add itemset length
frequent_itemsets["length"] = (
    frequent_itemsets["itemsets"].apply(len)
)

# Sort by support
frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.to_string(index=False))

print("\nNumber of Frequent Itemsets:")
print(len(frequent_itemsets))


# ============================================================
# PART 6: GENERATE ASSOCIATION RULES
# ============================================================

print("\n" + "=" * 70)
print("PART 6: ASSOCIATION RULES")
print("=" * 70)

# Generate rules using confidence
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.30
)

# Sort rules by lift
rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("\nAssociation Rules:")

if len(rules) > 0:

    display_columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]

    print(
        rules[display_columns].to_string(index=False)
    )

else:

    print("No association rules were found.")


# ============================================================
# PART 7: FILTER RULES RELATED TO MACHINE FAILURE
# ============================================================

print("\n" + "=" * 70)
print("PART 7: RULES RELATED TO MACHINE FAILURE")
print("=" * 70)

if len(rules) > 0:

    failure_rules = rules[
        rules["consequents"].apply(
            lambda x: "Machine_Failure" in x
        )
    ]

    failure_rules = failure_rules.sort_values(
        by="lift",
        ascending=False
    )

    if len(failure_rules) > 0:

        print("\nRules predicting Machine Failure:")

        print(
            failure_rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "\nNo rules with Machine_Failure as the "
            "consequent were found."
        )


# ============================================================
# PART 8: SAVE FREQUENT ITEMSETS
# ============================================================

# Convert itemsets to readable text
frequent_itemsets_output = frequent_itemsets.copy()

frequent_itemsets_output["itemsets"] = (
    frequent_itemsets_output["itemsets"]
    .apply(lambda x: ", ".join(sorted(x)))
)

frequent_itemsets_output.to_csv(
    "data/apriori_frequent_itemsets.csv",
    index=False
)

print(
    "\nFrequent itemsets saved to:"
    " data/apriori_frequent_itemsets.csv"
)


# ============================================================
# PART 9: SAVE ASSOCIATION RULES
# ============================================================

if len(rules) > 0:

    rules_output = rules.copy()

    rules_output["antecedents"] = (
        rules_output["antecedents"]
        .apply(lambda x: ", ".join(sorted(x)))
    )

    rules_output["consequents"] = (
        rules_output["consequents"]
        .apply(lambda x: ", ".join(sorted(x)))
    )

    rules_output.to_csv(
        "data/apriori_association_rules.csv",
        index=False
    )

    print(
        "Association rules saved to:"
        " data/apriori_association_rules.csv"
    )


# ============================================================
# PART 10: APRIORI SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("APRIORI SUMMARY")
print("=" * 70)

print("\nTotal Transactions:", len(transactions))

print("Total Possible Items:", len(all_items))

print("Minimum Support:", min_support)

print("Minimum Confidence:", 0.30)

print(
    "Frequent Itemsets Found:",
    len(frequent_itemsets)
)

print(
    "Association Rules Found:",
    len(rules)
)

if len(rules) > 0:

    print(
        "Highest Lift:",
        round(rules["lift"].max(), 4)
    )

    print(
        "Highest Confidence:",
        round(rules["confidence"].max(), 4)
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("LAB 4 - APRIORI COMPLETED SUCCESSFULLY!")
print("=" * 70)