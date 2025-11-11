import csv
from random import choice, uniform, randint
from datetime import datetime, timedelta

coins = ["bitcoin", "ethereum", "litecoin", "ripple", "cardano", "solana"]
types = ["buy", "sell"]
start_date = datetime(2025, 7, 10)

filename = "transactions_sample.csv"

with open(filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "coin", "type", "amount", "price"])

    for i in range(70):
        date = start_date + timedelta(days=i)
        coin = choice(coins)
        tx_type = choice(types)
        
        if coin == "bitcoin":
            amount = round(uniform(0.1, 0.5), 3)
            price = round(uniform(34000, 36000), 2)
        elif coin == "ethereum":
            amount = round(uniform(0.5, 1.5), 3)
            price = round(uniform(1800, 1950), 2)
        elif coin == "litecoin":
            amount = round(uniform(3, 8), 2)
            price = round(uniform(120, 135), 2)
        elif coin == "ripple":
            amount = randint(350, 500)
            price = round(uniform(0.60, 0.68), 2)
        elif coin == "cardano":
            amount = randint(100, 150)
            price = round(uniform(0.55, 0.61), 2)
        elif coin == "solana":
            amount = randint(8, 15)
            price = round(uniform(22, 25), 2)
        
        writer.writerow([date.strftime("%Y-%m-%d"), coin, tx_type, amount, price])

print(f"Generated sample transactions CSV: {filename}")
