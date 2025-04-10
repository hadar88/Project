import requests

SERVER_URL = "http://127.0.0.1:5000"

data = {"weights": [62.0, 63.0, 80, 50, 120, 40], "bmis": [19.79, 20.11, 25.5, 16, 38.3, 12.8], "times": ["2025-04-09", "2025-04-10", "2025-04-11", "2025-04-12", "2025-04-13", "2025-04-14"]}  # Example data

response = requests.get(f"{SERVER_URL}/wgraph", json=data)

if response.status_code == 200:
    with open("plot.png", "wb") as file:
        file.write(response.content)
    print("Saved")
else:
    print(response.json())
