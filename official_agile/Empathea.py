import gradio as gr
import os
from openai import OpenAI
from datetime import datetime
import json
import random

# HARDCODED API KEY - Replace with your actual API key
# WARNING: Hardcoding API keys is not secure and should only be done for testing
API_KEY = "your-api-key-here"  # Replace with your actual OpenAI API key

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=API_KEY)

# Configure conversation history storage
HISTORY_DIR = "conversation_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)

class EmpatheaAI:
    def __init__(self):
        self.system_prompt = """You are EmpatheaAI, a compassionate and empathetic AI companion with a diverse conversational style. When responding:

1. VARY YOUR APPROACH: Don't fall into patterns. Sometimes be brief and direct, other times more reflective or analytical.

2. MATCH THE USER'S ENERGY: If they're brief, be concise. If detailed, respond in kind.

3. USE DIFFERENT CONVERSATION TECHNIQUES:
   - Ask thoughtful follow-up questions when appropriate
   - Share relevant metaphors or gentle perspectives
   - Reflect feelings without repeating the same phrases
   - Occasionally use appropriate humor for lighter moments

4. AVOID REPETITIVE LANGUAGE PATTERNS: Don't rely on the same phrases, transitions, or structures.

5. BE AUTHENTIC: Respond as a thoughtful companion rather than a template-following assistant.

Always prioritize empathy and support, but express it in varied, natural ways. If someone expresses thoughts of self-harm or harm to others, kindly encourage them to seek professional help."""
        
        self.conversation_history = []
        self.session_id = None
        self.response_count = 0
        
        # Library of opening phrases to vary responses
        self.opening_phrases = [
            "I understand what you're saying about", 
            "It sounds like", 
            "I'm hearing that", 
            "From what you've shared,", 
            "That's really interesting -", 
            "I appreciate you sharing that.",
            "Thank you for opening up about this.",
            "I'm following what you're saying about",
            "What strikes me about your situation is",
            "I'm noticing that",
            "Let me reflect on what you've shared.",
            "I'm considering your words carefully.",
            "That's a thoughtful perspective.",
            "Your experience with this is important.",
            "I see what you mean about",
        ]
    
    def start_new_session(self):
        """Initialize a new conversation session with unique ID"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = []
        self.response_count = 0
        return "Hello, I'm here for you. Feel free to share whatever is on your mind."
    
    def save_conversation(self):
        """Save the current conversation to a file"""
        if not self.session_id:
            return
        
        filename = f"{HISTORY_DIR}/empathea_session_{self.session_id}.json"
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def analyze_message_length(self, message):
        """Analyze message length to adapt response style"""
        words = len(message.split())
        if words < 15:
            return "brief"
        elif words < 50:
            return "moderate"
        else:
            return "detailed"
    
    def get_response(self, user_input):
        """Get response from AI based on user input with enhanced variety"""
        if not user_input.strip():
            return "I'm here when you're ready to talk."
        
        if not self.session_id:
            self.start_new_session()
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Analyze message to adapt response style
        message_length = self.analyze_message_length(user_input)
        self.response_count += 1
        
        # Create diversity prompt based on conversation state
        diversity_prompt = self.system_prompt
        
        # Every few messages, add a specific style suggestion to increase variety
        if self.response_count % 3 == 0:
            styles = [
                "For this response, be more concise and direct than usual.",
                "For this response, use a more reflective, philosophical approach.",
                "For this response, use a gentle metaphor or analogy if appropriate.",
                "For this response, focus on asking thoughtful questions to explore the topic.",
                "For this response, be warm and encouraging with a touch of optimism.",
                "For this response, acknowledge emotion first before offering perspective."
            ]
            diversity_prompt += "\n\n" + random.choice(styles)
        
        # Adapt to message length
        if message_length == "brief":
            diversity_prompt += "\n\nThe user's message is brief. Keep your response concise and focused."
        elif message_length == "detailed":
            diversity_prompt += "\n\nThe user has shared in detail. You can be more thorough in your response."
        
        try:
            # Calculate dynamic temperature based on response count
            # Oscillate between 0.7 and 0.9 to vary creativity
            temperature = 0.7 + (self.response_count % 4) * 0.05
            
            # Prepare messages for API call with diversity enhancement
            messages = [
                {"role": "system", "content": diversity_prompt}
            ] + self.conversation_history
            
            # Call OpenAI API using the client interface
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=500
            )
            
            ai_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_message})
            self.save_conversation()
            
            return ai_message
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            return error_msg

# Initialize EmpatheaAI
empathea = EmpatheaAI()

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple")) as interface:
    gr.Markdown("## EmpatheaAI: A Compassionate Listener")
    gr.Markdown("Share your thoughts and feelings in a safe, judgment-free space.")
    
    # Keep track of chat history in the interface
    chat_history = gr.State([])
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=400,
        label="Conversation"
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="What's on your mind today?",
            container=True,
            scale=9
        )
        submit_btn = gr.Button("Send", scale=1)
    
    with gr.Row():
        clear_btn = gr.Button("Start New Conversation")
    
    gr.Markdown("""
    ### About EmpatheaAI
    
    EmpatheaAI is designed to provide a supportive space where you can express your emotions and thoughts.
    Sometimes we all need someone who will simply listen without judgment.
    
    **Important Note:** While EmpatheaAI aims to provide support, it is not a substitute for professional mental health care.
    If you're experiencing a crisis or need immediate help, please contact a mental health professional or a crisis helpline.
    
    **Crisis Resources:**
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    """)
    
    def user_submit(user_message, history):
        # Get AI response
        response = empathea.get_response(user_message)
        
        # Update chat history
        history.append([user_message, response])
        
        return gr.update(value=""), history, history
    
    def start_new_chat():
        welcome = empathea.start_new_session()
        new_history = [[None, welcome]]
        return new_history, [], gr.update(value="")
    
    submit_btn.click(
        user_submit, 
        inputs=[user_input, chat_history], 
        outputs=[user_input, chatbot, chat_history]
    )
    
    user_input.submit(
        user_submit, 
        inputs=[user_input, chat_history], 
        outputs=[user_input, chatbot, chat_history]
    )
    
    clear_btn.click(
        start_new_chat, 
        inputs=None, 
        outputs=[chatbot, chat_history, user_input]
    )

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
