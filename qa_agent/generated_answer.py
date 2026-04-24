from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_ENGINE") 


def extract_medical_keywords(user_query):
    """Uses the LLM to pull specific medical terms for searching."""
    extraction_prompt = f"""
    Identify the main medical conditions, symptoms, or treatments in this sentence. 
    Return them as a comma-separated list. 
    Correct any obvious spelling errors (e.g., 'hearth' to 'heart').
    
    Sentence: "{user_query}"
    Keywords:"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": extraction_prompt}],
        max_tokens=50,
        temperature=0
    )
    # Returns a list: ["heart attack", "sexual interest"]
    terms = response.choices[0].message.content.split(',')
    return [t.strip() for t in terms]

def generated_answer(extracted_terms_list, diagnosis_text, original_query, history=[]):
    # 1. Determine the behavior based on the diagnosis_text flag
    if diagnosis_text == "EXTERNAL_KNOWLEDGE":
        context_instruction = f"I couldn't find a specific study for '{original_query}', so I'm using my general knowledge."
        input_content = f"Please explain '{original_query}' for a 6th grader."
    else:
        context_instruction = "Use the provided definitions and research text to answer."
        input_content = f"Definitions: {extracted_terms_list}\nResearch: {diagnosis_text}\nUser Query: {original_query}"

    # 2. Build the System Message with strict "Conversation" instructions
    system_content = f"""
    Role: You are a friendly, human-like medical assistant.
    Current Goal: Answer the user's latest message: "{original_query}"
    
    CRITICAL INSTRUCTION: 
    If the user says 'Yes', 'Go ahead', or 'Sure', look at the VERY LAST question you asked in the conversation history. 
    You MUST fulfill that specific offer (e.g., explain treatments, symptoms, etc.) using the provided Research Context.
    
    Tone Guidelines:
    - 6th-grade reading level (simple words, short sentences).
    - No headers like 'Word Bank'. 
    - Respond like a helpful peer.
    - ALWAYS end with a new, helpful follow-up question.
    
    {context_instruction}
    """

    # 3. Assemble the messages
    messages = [{"role": "system", "content": system_content}]
    
    # Add history so the AI remembers the question it just asked
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current turn
    messages.append({"role": "user", "content": input_content})

    # 4. Call Azure OpenAI
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages, 
        max_tokens=600,
        temperature=0.4 # Slightly higher temperature helps with conversational flow
    )
    
    return response.choices[0].message.content