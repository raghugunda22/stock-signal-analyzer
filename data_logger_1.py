

mock_price = [100.5, 102.0, 101.5, 105.0, 108.5, 99.0]


moving_avg = 103.0

print("--- 📈 Price Data ---")

for price in mock_price:
    if price > moving_avg:
        print(f"Price {price} is above the moving average of {moving_avg}.")
    else:
        print(f"Price {price} is below the moving average of {moving_avg}.")


print("--- Check Complete ---")

