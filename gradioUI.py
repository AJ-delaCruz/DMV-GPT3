import openai
import gradio as gr

openai.api_key = "key"
fine_tuned_model_id = "davinci:ft-personal-2023-05-01-03-34-01"


def chatbot(question):
    # Use a fine-tuned model
    response = openai.Completion.create(
        engine=fine_tuned_model_id,
        prompt=f"{question} ->",
        # temperature=0.5,  # 0,1 higher makes output more random
        max_tokens=150,  # aximum number of tokens in the generated output
        # top_p=1,  # value of 1 means all tokens have equal probability
        # frequency_penalty=0,
        # presence_penalty=0,
        stop=["\n"],  # Add an end-of-sequence token to stop the response
    )

    answer = response.choices[0].text.strip()
    return answer


interface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(label="Question"),
    outputs=gr.components.Textbox(label="Answer"),
    title="GPT-3 fine-tuned on California Drivers' Handbook",
)

interface.launch()
