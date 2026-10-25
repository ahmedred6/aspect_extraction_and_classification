import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------
FILE = r"C:\Users\ahmed\Downloads\monthly_electricity_consumption.csv"
df = pd.read_csv(FILE)

# -----------------------------------------------------
# Space IDs
# -----------------------------------------------------
SPACE_A = "3ec75bbc-73b9-4dcf-95e7-eecf53c33ae5"
SPACE_B = "0c2abaaf-e787-4538-8db0-4dc75e0f1fc7"

spaces = {
    "Space A": SPACE_A,
    "Space B": SPACE_B
}

plt.figure(figsize=(12, 6))

for label, space_id in spaces.items():
    space_df = df[df["space_uuid"] == space_id].copy()
    
    if space_df.empty:
        print(f"⚠ No data found for {space_id}")
        continue
    
    # Build date column
    space_df["date"] = pd.to_datetime(
        space_df["billing_year"].astype(str) + "-" +
        space_df["billing_month"].astype(str) + "-01"
    )
    
    # Sort chronologically
    space_df = space_df.sort_values("date")
    
    # Plot
    plt.plot(space_df["date"], space_df["total_kwh"],
             marker="o", linewidth=2, label=label)

# -----------------------------------------------------
# Final formatting
# -----------------------------------------------------
plt.title("Electricity Consumption Trends (Two Spaces)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Total kWh")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
