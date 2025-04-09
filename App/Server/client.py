import requests

SERVER_URL = "http://127.0.0.1:5000"

data = {
    "weights": [60, 61, 62, 70, 80],
    "bmis": [12, 20.2, 20.5, 27, 30],
    "times": ["2025-04-09", "2025-04-10", "2025-04-17", "2025-04-18", "2025-04-20"]
}

response = requests.get(f"{SERVER_URL}/wgraph", json=data)

if response.status_code == 200:
    with open("plot.png", "wb") as file:
        file.write(response.content)
    print("Saved")
else:
    print(response.json())
