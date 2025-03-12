import sys
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from rasa.__main__ import main

if __name__ == "__main__":
    # Imposta sys.argv in modo che includa il comando "run"
    sys.argv = ["rasa", "run"]
    main()
