import pandas as pd
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from groqconnection  import GroqModel
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from deepeval.test_case import  LLMTestCase

def test_coherancy_metric():
    load_dotenv ()
    eval_model = GroqModel(model_name="llama-3.3-70b-versatile")
    criteria = """Coherence (1-5) - the collective quality of all sentences. We align this dimension with
    the DUC quality question of structure and coherence whereby the summary should be
    well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic."""

    coherence_metric = GEval(
        name="Coherence",
        criteria=criteria,
         model=eval_model, async_mode=False,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    )

    
    test_case = LLMTestCase(input="Hey how's the weather like today?", actual_output="Temp 40F Cloudy !")

    # Use G-Eval metric
    coherence_metric.measure(test_case)
    print(coherence_metric.score, coherence_metric.reason)
