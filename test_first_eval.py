import os
import json
import pandas as pd
from dotenv import load_dotenv
# from  groqwrapper import  GroqModel 
from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM

# Improved Groq wrapper to handle JSON requirements
class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3-70b-8192"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        use_json = "llama" in self.model_name.lower()
        chat_completion = self.client.chat.completions.create(
            messages= [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Respond strictly in JSON format."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
            model=self.model_name,
            # Force JSON mode if using a compatible model
            response_format={"type": "json_object"} if use_json else None
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# Rest of your script...
def test_first_eval():
    load_dotenv()
    eval_model = GroqModel(model_name="llama-3.3-70b-versatile")

    
    # Initialize metrics with the model
    # Note: Use async_mode=False for simple loops to avoid event loop issues
    arm = AnswerRelevancyMetric(threshold=0.7, model=eval_model, async_mode=False)
    cr = ContextualRelevancyMetric(threshold=0.7, model=eval_model, async_mode=False)
    cp = ContextualPrecisionMetric(threshold=0.7, model=eval_model, async_mode=False)
    hm = HallucinationMetric(threshold=0.7, model=eval_model, async_mode=False)

    # Load data
    df = pd.read_excel("deepeval_rag_test.xlsx")
    df['answer_relevancy_score'] = 0.0
    df['contextual_relevancy_score'] = 0.0

    for index, row in df.iterrows():
        # Clean contexts: Ensure they are lists of strings
        # DeepEval often expects multiple chunks in retrieval_context
        retrieval_context = [str(row['retrieval_context'])] if pd.notna(row['retrieval_context']) else []
        ground_truth_context = [str(row['context'])] if pd.notna(row['context']) else []
        user_input = str(row['input'])
        print("Generating live response from Groq...")
        generated_output = eval_model.generate(user_input)
        print("Hi i me genrted",generated_output)
    
    
  

        test_case = LLMTestCase(
            input=str(row['input']),
            actual_output= generated_output,
            expected_output=str(row['expected_output']),
            retrieval_context=retrieval_context,
            context=ground_truth_context
        )

        # Run metrics
        try:
            print(f"Checking Relevancy...")
            arm.measure(test_case)
            cr.measure(test_case)
            cp.measure(test_case)
            print(f"Checking Halluicanation...")
            hm.measure(test_case)
           
            
            df.at[index, 'answer_relevancy_score'] = arm.score
            df.at[index, 'contextual_relevancy_score'] = cr.score
            df.at[index, 'contextual_precision_score'] = cp.score
            df.at[index, 'hallucination_score'] = hm.score
            # Print scores to terminal to see progress
            print(f"--- Row {index} Results ---")
            print(f"Answer Relevancy: {arm.score}")
            print(f"Context Relevancy: {cr.score}")
            print(f"Context Precision: {cp.score}")
            print(f"Hallucination: {hm.score}")
            print("-" * 30) # Prints a separator line
        except Exception as metric_err:
            print(f"Failed to evaluate row {index}: {metric_err}")
        df.to_excel("deepeval_rag_test_results.xlsx", index=False)



def test_ans_relvancy():
    load_dotenv()
    
    # 1. Setup Model
    eval_model = GroqModel(model_name="llama-3.3-70b-versatile")
    
    # 2. Setup Metric
    arm1 = AnswerRelevancyMetric(threshold=0.7, model=eval_model, async_mode=False)
    faith_metric = FaithfulnessMetric(threshold=0.7, model=eval_model)
   
    # 3. Define your prompt string (DO NOT name this 'input')
    query_text = "Who is preident of America"
    retrieval_context = [
        "Donald J. Trump was inaugurated as the 47th President of the United States on January 20, 2025.",
        "The 2024 election resulted in a victory for the Republican ticket."
    ]
    

    generated_output = eval_model.generate(query_text)
    print(f"\nGenerated Output: {generated_output}")
    
    # 5. Create Test Case
    test_case = LLMTestCase(
        input=query_text, 
        actual_output=generated_output,
        retrieval_context=retrieval_context
    )
    
    # 6. Measure
    arm1.measure(test_case)
    faith_metric.a_measure(test_case)

    
    print(f"Relevancy Score: {arm1.score}")
    print(f"Reasoning: {arm1.reason}")
    print(f"Relevancy Score: {faith_metric.score}")
    print(f"Reasoning: {faith_metric.reason}")
    
    assert arm1.is_successful(), f"Score {arm1.score} is below threshold!"
    assert faith_metric.is_successful(), f"Score {faith_metric.score} is below threshold!"

    evaluate(test_case,metrics=arm1)



    
  
    
   

    





