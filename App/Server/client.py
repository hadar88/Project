import requests

SERVER_URL = "http://127.0.0.1:5000"

levels = [16, 18.5, 25, 30, 40]

data = {"weights": [30, 30], "bmis": [15, 15], "times": ["2025-04-01", "2025-04-02"]}

response = requests.get(f"{SERVER_URL}/wgraph", json=data)

if response.status_code == 200:
    with open("plot.png", "wb") as file:
        file.write(response.content)
    print("Saved")
else:
    print(response.json())
