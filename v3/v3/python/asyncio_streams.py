import asyncio

async def echo_server():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    print(f"Server listening on port {port}")
    async def handle_client(reader, writer):
        data = await reader.read(100)
        writer.write(data)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    asyncio.create_task(server.serve_forever())
    return server, port

async def client(port):
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    writer.write(b'Hello asyncio')
    await writer.drain()
    data = await reader.read(100)
    writer.close()
    await writer.wait_closed()
    return data

async def main():
    server, port = await echo_server()
    try:
        response = await client(port)
        print(f"Client received: {response.decode()}")
    finally:
        server.close()
        await server.wait_closed()

asyncio.run(main())
