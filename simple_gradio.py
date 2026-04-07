import os
import gradio as gr
from models import EmailTriageAction
from server.email_triage_environment import EmailTriageEnvironment
from smart_agent import get_llm_classification

env = EmailTriageEnvironment()

def process_email(user_email_text):
    token = os.environ.get("GROQ_API_KEY")
    if not token or not token.startswith("gsk_"):
        return "Error: Please set the GROQ_API_KEY environment variable in your terminal first."
    
    if not user_email_text.strip():
        return "Please type an email first."
    
    # Start a new episode in the environment
    env.reset() 
    
    # Feed user's custom email text
    env._current_email_text = user_email_text
    
    # Since you requested a simple interface without a "ground truth" selector,
    # the environment will default to evaluating everything against "standard".
    env._current_email_category = "standard"
    
    # The Llama Brain analyzes the email
    guess = get_llm_classification(user_email_text, token)
    
    # The environment evaluates the guess and assigns a reward
    action = EmailTriageAction(category=guess)
    obs = env.step(action)
    
    # Return formatted result string showing category and reward
    return f"Category: {guess.upper()}\nReward Score: {obs.reward}"

# Build the simple custom Gradio User Interface
with gr.Blocks(title="Simple Email Triage") as demo:
    gr.Markdown("## Simple Email Triage")
    
    email_in = gr.Textbox(label="Type Email Here", lines=4, placeholder="e.g. Claim your sweepstakes prize now!")
    submit = gr.Button("Analyze", variant="primary")
    result_out = gr.Textbox(label="Result", lines=2)
    
    submit.click(
        fn=process_email,
        inputs=[email_in],
        outputs=[result_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
