from model_testing import InsideModel
import sqlite3
import asyncio
import websockets
import json
import socket

conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

# cursor.execute("DELETE FROM InsideValues")

# model = InsideModel(0, '/home/jbeni/Desktop/School/COP4934/AI-for-Safe-Driving/Latest Models/yolov9t/weights/best.pt', cursor)
# model.run_inferences(10, True, True)
# conn.commit()

cursor.execute("SELECT * FROM InsideValues")
results = cursor.fetchall()

results_arr = []

if len(results) != 0:
    for row in results:
        # print(row)
        results_arr.append(row)
else:
    print("table is empty")

send_json = {
    "results": results_arr
}

send_json = json.dumps(send_json)

# print(send_json)

async def send_data(json_data: str, uri):
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json_data)

            response = await websocket.recv()
            await websocket.close()
            return response
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed with code: {e.code}, reason: {e.reason}")

def valid_internet_connection() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:
        pass
    return False

if valid_internet_connection():
    res = asyncio.run(send_data(send_json, "wss://aifsd.xyz"))
else:
    print("Not connected to the internet.")


conn.close()