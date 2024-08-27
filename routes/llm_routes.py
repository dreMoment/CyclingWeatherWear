import os
from pathlib import Path
from flask import Blueprint, request
from llm_service.llm_factory import LLMServiceFactory
from prompts import SYSTEM_PROMPT, USER_PROMPT

from dotenv import load_dotenv

load_dotenv(".env", override=True)

print(os.getenv("DEFAULT_MODEL_PROVIDER"))


# load dotenv in the base root
llm_routes_bp = Blueprint("llm_routes", __name__)


### General routes ###


# Prompt LLM to generate something
@llm_routes_bp.route("/suggestion", methods=["POST"])
def promptLLM():
    service = LLMServiceFactory.get_service(
        provider=os.getenv("DEFAULT_MODEL_PROVIDER"),
        model_name=os.getenv("DEFAULT_MODEL_NAME"),
        temperature=float(os.getenv("DEFAULT_TEMPERATURE")),
        max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS")),
    )

    clothes = request.json.get("clothes")
    temperature = request.json.get("temperature")
    precipitation = request.json.get("precipitation")
    wind_speed = request.json.get("wind_speed")

    prompt = USER_PROMPT.format(
        temperature=temperature,
        precipitation=precipitation,
        wind_speed=wind_speed,
        clothes=clothes,
    )

    print(prompt)
    res = service.make_request(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    # Parse only the first sentence
    res = res.split(".")[0] + "."

    return res
