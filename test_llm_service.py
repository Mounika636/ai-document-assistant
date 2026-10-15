
from unittest.mock import patch

from llm_service import generate_answer


def test_empty_context():
    answer = generate_answer(
        question="What is the remote work policy?",
        context="",
    )

    assert "couldn't find" in answer.lower()


@patch("llm_service.client.chat")
def test_generate_answer(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": "Employees can work remotely 3 days."
        }
    }

    answer = generate_answer(
        question="How many remote days are allowed?",
        context="Employees may work remotely 3 days.",
    )

    assert "3 days" in answer
    mock_chat.assert_called_once()
