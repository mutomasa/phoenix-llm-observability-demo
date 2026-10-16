import pytest


class FakeJudgeLLM:
    """Phoenix Evals の LLM の代わりに、用意した応答を順に返す。"""

    model = "fake-judge"

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.kwargs = []

    def generate_object(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def make_judge_llm():
    return FakeJudgeLLM
