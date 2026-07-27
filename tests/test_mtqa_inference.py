import unittest

from inference.MTQA.inference import (
    OPEN_ENDED_SYSTEM_PROMPT,
    BaseClient,
    chat_with_memory,
)


class RecordingClient(BaseClient):
    def __init__(self):
        self.requests = []

    def chat_with_memory(self, image_paths, text, messages, timeout=120.0):
        self.requests.append(
            {
                "image_paths": image_paths,
                "text": text,
                "messages": list(messages),
            }
        )
        return "answer"


class OpenEndedInferenceTests(unittest.TestCase):
    def test_figure_10_system_prompt_and_memory_are_applied(self):
        client = RecordingClient()
        turns = [
            {"turn_id": "T1", "question": "first"},
            {"turn_id": "T2", "question": "second"},
        ]
        chat_with_memory(client, turns, ["image.png"])

        first_messages = client.requests[0]["messages"]
        self.assertEqual(first_messages[0]["role"], "system")
        self.assertIn(
            OPEN_ENDED_SYSTEM_PROMPT,
            first_messages[0]["content"][0]["text"],
        )
        second_messages = client.requests[1]["messages"]
        self.assertEqual(second_messages[-2]["role"], "user")
        self.assertEqual(second_messages[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
