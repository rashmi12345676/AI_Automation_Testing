import pandas as pd
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from groqconnection  import GroqModel
from deepeval.metrics  import    HallucinationMetric

def test_ans_hallucination ():
    load_dotenv()
    eval_model = GroqModel(model_name="llama-3.3-70b-versatile")
    
    Golden_data_set = ["Wire transfers to Germany have a $15 fee"," Transfers to the UK are $20"]
    User_query = "how much fee required for tranfer the money to franch/UK and Customer get any tax exempt"
    generated_output= eval_model.generate(User_query)
    print("Hi i am LLMgenerated_output Genrated OUT ",generated_output )
    # actual_output = "The fee for Germany is $100 according to the 2024 Global Partnership rules."

    metric = HallucinationMetric(threshold=0.9, model=eval_model, async_mode=False)
    test_case = LLMTestCase(
    input=User_query,
    actual_output=generated_output,
    context=Golden_data_set
)
    metric.measure(test_case)
    print(f"Relevancy Score: {metric.score}")
    print(f"Reasoning: {metric.reason}")
    assert metric.is_successful(), f"Score {metric.score} is below threshold!"
    


