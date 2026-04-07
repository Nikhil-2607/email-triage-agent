import os
import sys
from openai import OpenAI
from models import EmailTriageAction
from server.email_triage_environment import EmailTriageEnvironment

def get_llm_classification(email_text: str, token: str) -> str:
    """Uses Llama 3.2 via Groq API to classify the email text."""
    
    # Initialize the standard OpenAI client configured for Groq
    client = OpenAI(
        api_key=token,
        base_url="https://api.groq.com/openai/v1"
    )
    
    system_prompt = (
        "You are an AI assistant that triages emails. "
        "Your job is to read an email and respond with EXACTLY ONE of the following precise words: "
        "'spam', 'urgent', or 'standard'. Do not include any punctuation, conversational text, or prefixes."
    )
    
    user_prompt = f"Here is the email to classify:\n\n{email_text}\n\nCategory:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print("  [LLM] Analyzing email with Groq...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )
        category = response.choices[0].message.content.strip().lower()
        
        # Cleanup response just in case the LLM outputs a sentence
        for valid in ["spam", "urgent", "standard"]:
            if valid in category:
                return valid
                
        return "standard" # Fallback if no matching keyword is found
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return "standard"

def main():
    # Require the user to set their Groq token
    token = os.environ.get("GROQ_API_KEY")
    if not token:
        print("Error: Groq API token is missing!")
        print("Please set your GROQ_API_KEY environment variable in the terminal first:")
        print("  Windows: $env:GROQ_API_KEY='gsk_your_groq_key'")
        print("  Mac/Linux: export GROQ_API_KEY='gsk_your_groq_key'")
        sys.exit(1)

    print("Initializing Smart Agent Environment...")
    env = EmailTriageEnvironment()
    
    total_score = 0
    episodes = 10
    
    for i in range(episodes):
        print(f"\n--- Episode {i+1} ---")
        obs = env.reset()
        email_text = obs.email_text
        print(f"Received Email: '{email_text}'")
        
        # Send the email to the Groq Llama 3.2 Brain
        guess = get_llm_classification(email_text, token)
        print(f"Agent guesses: {guess}")
        
        # Execute the step in the environment
        action = EmailTriageAction(category=guess)
        obs = env.step(action)
        
        print(f"Feedback: {obs.feedback}")
        print(f"Reward: {obs.reward}")
        
        total_score += obs.reward

    print(f"\n======================================")
    print(f"Final Agent Score: {total_score} / {episodes}")
    print(f"======================================")

if __name__ == "__main__":
    main()
