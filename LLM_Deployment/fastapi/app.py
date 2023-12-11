from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
import uvicorn
from pydantic import BaseModel

# FASTAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)


# Request Body
class Query(BaseModel):
    query: str
    max_tokens: int


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: [str]


def format_prompt(user_prompt: str):
    return f""" You are an expert in asking questions. Generate short and concise questions based on the user prompt.
    User Prompt: {user_prompt}
"""


def generate(
        llm: AutoModelForCausalLM,
        generation_config: GenerationConfig,
        user_input: str,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(user_input),
        **asdict(generation_config), )

#
# model_path = "models/llama-2-7b.ggmlv3.q4_1.bin"
# config_path = "models"

config = AutoConfig.from_pretrained(
    os.path.abspath("models"),
    context_length=2048,
)
llm = AutoModelForCausalLM.from_pretrained(
    os.path.abspath("models/llama-2-7b.ggmlv3.q4_1.bin"),
    model_type="llama",
    config=config,
)


@app.get("/")
def root_fun():
    return {"Status": "Alive"}


@app.post('/generate')
async def question_generator(q: Query):
    generation_config = GenerationConfig(
        temperature=0.6,
        top_k=25,
        top_p=0.5,
        repetition_penalty=1.1,
        max_new_tokens=q.max_tokens,  # adjust as needed
        seed=42,
        reset=True,  # reset history (cache)
        stream=True,  # streaming per word/token
        threads=int(os.cpu_count() / 6),  # adjust for your CPU
        stop=["<|endoftext|>"],
    )

    questions = generate(llm, generation_config, q.query)
    return {'generated_text': questions}


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8080, reload=True)

