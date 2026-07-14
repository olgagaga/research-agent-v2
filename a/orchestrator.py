import os
from typing import List
from schemas import ExperimentPlan
from openai import OpenAI

def build_system_message():
    messages: List[str] = []
    task_path = os.environ.get("TASK_MD")
    wiki_path = os.environ.get("WIKI_MD")
    logs_path = os.environ.get("LOGS_MD")
    code_files = ...

    task = ...
    wiki = ...
    logs = ...

    message = f"Task: {task}. Logs: {}. Wiki: {}."
    return message


def run_iteration(client, model):
    message = build_system_message()
    response = client.responses.parse(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a coding agent. "
                    "Return ONLY structured edits."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        text_format=ExperimentPlan,
    )
    result = response.output_parsed




def run_loop():
    api_key = os.environ.get("API_KEY")
    model = os.environ.get('MAIN_MODEL')

    client = OpenAI(api_key=api_key)



    r



