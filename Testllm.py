import os

import pytest
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference


def test_contest_precision():
    os.environ["API_KEY"] = "AIzaSyCJ4tHwQuY1Bq5mdQpPpBQtblG1A2ZltLM"
    llm = ChatOpenAI(model= "gpt4",temperture = 0)
    langchainllm = LangchainLLMWrapper(llm)
    LLMContextPrecisionWithoutReference(llm =langchainllm )
    SingleTurnSample(
        user_res = 'how many python  tutorial avalable in youtube',
        retrive_data = ''

    )




