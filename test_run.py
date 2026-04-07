import sys
from models import EmailTriageAction
from server.email_triage_environment import EmailTriageEnvironment

def main():
    print("Initializing Email Triage Environment...")
    env = EmailTriageEnvironment()
    
    for i in range(3):
        print(f"\n--- Episode {i+1} ---")
        obs = env.reset()
        print(f"Received Email: '{obs.email_text}'")
        
        # We will guess randomly or hardcode a guess for testing purposes
        import random
        guess = random.choice(["spam", "urgent", "standard"])
        print(f"Agent guesses: {guess}")
        
        action = EmailTriageAction(category=guess)
        obs = env.step(action)
        print(f"Feedback: {obs.feedback}")
        print(f"Reward: {obs.reward}")

if __name__ == "__main__":
    main()
