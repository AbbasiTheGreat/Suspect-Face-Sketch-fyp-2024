import gradio as gr

# Define function to display image with prompt
def display_image_with_prompt(prompt_text):
    image_path = "usama_pic.jpeg"
    return image_path, prompt_text

# Define Gradio interface
inputs = gr.Textbox(label="Enter a prompt:")
outputs = [gr.Image(type="pil", label="Image"), gr.Textbox(label="Prompt")]

iface = gr.Interface(fn=display_image_with_prompt, 
                     inputs=inputs, 
                     outputs=outputs, 
                     title="Display Image with Prompt",
                     description="Enter a prompt and see the image 'usama_pic.jpeg'.")

# Launch the interface
iface.launch()
