# websocket_server.py
import asyncio
import websockets
import json
import time
import socket

connected_clients = set()

async def handle_connection(websocket, path):
    """Handle WebSocket connections"""
    connected_clients.add(websocket)
    print(f"✅ Client connected. Total clients: {len(connected_clients)}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
                elif data.get('type') == 'progress_request':
                    await websocket.send(json.dumps({'type': 'progress', 'data': {'stage': 'ready', 'progress': 0}}))
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
                
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")

async def broadcast_progress(progress_data):
    """Broadcast progress updates to all connected clients"""
    if connected_clients:
        message = json.dumps({
            'type': 'progress',
            'data': progress_data
        })
        # Use asyncio.gather with return_exceptions to handle client errors
        results = await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )
        # Remove failed clients
        for client, result in zip(list(connected_clients), results):
            if isinstance(result, Exception):
                connected_clients.discard(client)

def is_port_available(port, host='0.0.0.0'):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

async def start_websocket_server(host='0.0.0.0', port=8765):
    """Start WebSocket server on specified port"""
    try:
        async with websockets.serve(handle_connection, host, port):
            print(f"✅ WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever
    except Exception as e:
        print(f"❌ WebSocket error on port {port}: {e}")
        raise
    
    try:
        async with websockets.serve(
            handle_connection, 
            host, 
            port,
            ping_interval=20,
            ping_timeout=20
        ):
            print(f"✅ WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever
    except Exception as e:
        print(f"❌ WebSocket server error: {e}")

if __name__ == "__main__":
    asyncio.run(start_websocket_server())