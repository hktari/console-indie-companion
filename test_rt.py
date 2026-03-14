import websockets, asyncio, os, base64, json
from dotenv import load_dotenv

load_dotenv()


async def test():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {"Authorization": "Bearer " + api_key, "OpenAI-Beta": "realtime=v1"}
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.recv()
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"modalities": ["audio", "text"], "voice": "sage"},
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create"}))
        for _ in range(50):
            msg = json.loads(await ws.recv())
            print(msg["type"])
            if msg["type"] == "response.done":
                print(json.dumps(msg, indent=2))
                break


asyncio.run(test())
