import logging
import azure.functions as func

print("🔥 Module loaded.")

def main(msg: func.QueueMessage) -> None:
    print("📥 Queue trigger main() called.")
    try:
        body = msg.get_body().decode("utf-8")
        print(f"📩 Raw message: {body}")
    except Exception as e:
        print(f"💥 Exception: {e}")