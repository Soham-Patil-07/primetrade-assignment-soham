import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load datasets
fear = pd.read_csv("fear_greed_index.csv")
trader = pd.read_csv("historical_data.csv")

print("Files Loaded Successfully")

# Convert dates
fear['date'] = pd.to_datetime(fear['date'])

trader['Date'] = pd.to_datetime(
    trader['Timestamp IST'],
    dayfirst=True
).dt.date

trader['Date'] = pd.to_datetime(trader['Date'])

# Merge datasets
merged = pd.merge(
    trader,
    fear,
    left_on='Date',
    right_on='date',
    how='left'
)

print("Merge Completed")
print(merged.head())

avg_pnl = merged.groupby(
    'classification'
)['Closed PnL'].mean()

print(avg_pnl)

avg_pnl.plot(kind='bar')

plt.title("Average Profit by Market Sentiment")
plt.ylabel("Average PnL")

plt.tight_layout()

plt.savefig("output/avg_profit.png")

plt.close()

total_pnl = merged.groupby(
    'classification'
)['Closed PnL'].sum()

total_pnl.plot(kind='bar')

plt.title("Total Profit by Market Sentiment")

plt.tight_layout()

plt.savefig("output/total_profit.png")

plt.close()

merged['Win'] = merged['Closed PnL'] > 0

win_rate = merged.groupby(
    'classification'
)['Win'].mean() * 100

print(win_rate)

win_rate.plot(kind='bar')

plt.title("Win Rate by Sentiment")
plt.ylabel("Percentage")

plt.tight_layout()

plt.savefig("output/win_rate.png")

plt.close()

side_analysis = merged.groupby(
    ['classification', 'Side']
)['Closed PnL'].mean().unstack()

side_analysis.plot(kind='bar')

plt.title("Buy vs Sell Performance")

plt.tight_layout()

plt.savefig("output/buy_sell.png")

plt.close()

coin_profit = merged.groupby(
    'Coin'
)['Closed PnL'].sum().sort_values(
    ascending=False
).head(10)

coin_profit.plot(kind='bar')

plt.title("Top 10 Profitable Coins")

plt.tight_layout()

plt.savefig("output/top_coins.png")

plt.close()

with open("output/insights.txt", "w") as f:
    f.write("PRIMETRADE ASSIGNMENT\n\n")

    f.write("Average Profit by Sentiment:\n")
    f.write(str(avg_pnl))
    f.write("\n\n")

    f.write("Win Rate by Sentiment:\n")
    f.write(str(win_rate))
    f.write("\n\n")

    f.write("Total Profit by Sentiment:\n")
    f.write(str(total_pnl))

print("Analysis Complete")

coin_profit = merged.groupby(
    'Coin'
)['Closed PnL'].sum().sort_values(
    ascending=False
)

print("\nTop 10 Coins:")
print(coin_profit.head(10))

sentiment_count = merged['classification'].value_counts()

print(sentiment_count)

side_pnl = merged.groupby(
    ['classification', 'Side']
)['Closed PnL'].mean()

print("\nAverage PnL by Sentiment and Side:")
print(side_pnl)