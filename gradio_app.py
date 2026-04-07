import os
import gradio as gr
from models import EmailTriageAction
from server.email_triage_environment import EmailTriageEnvironment
from smart_agent import get_llm_classification

# Initialize a global environment instance
env = EmailTriageEnvironment()

def process_email(user_email_text, actual_category, token):
    if not token or not token.startswith("gsk_"):
        return "N/A", "Please enter a valid Groq API Key.", "0.0"
    
    if not user_email_text:
        return "N/A", "Please type an email first.", "0.0"
    
    # Reset starts a new episode and picks a random predefined email
    env.reset() 
    
    # We override the environment's selected email so it evaluates your custom email!
    env._current_email_text = user_email_text
    env._current_email_category = actual_category.lower()
    
    # The Llama Brain analyzes the email
    guess = get_llm_classification(user_email_text, token)
    
    # The environment evaluates the guess and assigns a reward
    action = EmailTriageAction(category=guess)
    obs = env.step(action)
    
    return guess, obs.feedback, str(obs.reward)

# Build the custom Gradio User Interface
with gr.Blocks(title="Email Triage Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 Email Triage AI Agent")
    gr.Markdown("Type a custom email below, define its *Ground Truth Target*, and watch the Llama agent attempt to classify it for a reward score!")
    
    with gr.Row():
        # Pre-fill with the API Key if it's currently set in the environment
        groq_token = gr.Textbox(
            label="Groq API Key (gsk_...)", 
            type="password", 
            value=os.environ.get("GROQ_API_KEY", "")
        )
    
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(
                label="Custom Email Text", 
                lines=5, 
                placeholder="e.g. You have 1 unseen notification. Click here to log in..."
            )
            actual_cat_input = gr.Radio(
                choices=["spam", "urgent", "standard"], 
                label="Ground Truth Category (For Environment Scoring)", 
                value="spam"
            )
            submit_btn = gr.Button("Analyze Email", variant="primary")
            
        with gr.Column():
            agent_guess = gr.Textbox(label="Agent's Final Classification")
            env_feedback = gr.Textbox(label="Environment Feedback")
            env_reward = gr.Textbox(label="Environment Reward (Score)")
            
    submit_btn.click(
        fn=process_email,
        inputs=[email_input, actual_cat_input, groq_token],
        outputs=[agent_guess, env_feedback, env_reward]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
