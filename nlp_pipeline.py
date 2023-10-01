nlp_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

def generate_response(query, context):
    response = nlp_pipeline(question=query, context=context)
    return response["answer"]
