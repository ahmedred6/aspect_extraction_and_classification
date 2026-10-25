from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import os
from pydantic import BaseModel, Field

class SQLResult(BaseModel):
    sql: str = Field(description="A valid MySQL SELECT query. No comments, no markdown.")


llm_raw = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-04-28",
    temperature=0,
    max_tokens=1500
)

#this is the agent, this is not rag, it is just prompt and thats it
#properly they have something better?
SQL_PROMPT = """
You are a MySQL expert. Convert the user question into ONE valid MySQL SELECT query.

RULES:
- Output ONLY a SELECT query.
- Do NOT include explanations.
- Do NOT include markdown.
- Query must be syntactically valid.
- Use the following schema:

TABLE community(community_id, community_name, lat, lon)
TABLE space(space_uuid, space_name, community_id, area, created_at)
TABLE subspace(subspace_uuid, subspace_name, space_uuid, created_at)
TABLE category(category_code, category_name)
TABLE product(product_uuid, product_name, cat_name, prod_type, provider_type, created_at)
TABLE device_tag(tag_uuid, tag_name, created_at, updated_at)
TABLE device(device_uuid, tag_uuid, device_name, serial_number, is_online, time_zone,
             space_uuid, subspace_uuid, product_uuid, ip, created_at, updated_at)
TABLE device_status(status_id, device_uuid, status_code, status_value, recorded_at)
TABLE automation(automation_uuid, automation_name, automation_status, automation_type,
                 space_uuid, community_id)
TABLE monthly_electricity_consumption(id, space_uuid, billing_year, billing_month, total_kwh, estimated_cost_aed)

Return ONLY the SQL as required by the SQLResult schema.
"""


def generate_sql(question: str):
    llm_tool = llm_raw.bind_tools([SQLResult])

    messages = [
        SystemMessage(content=SQL_PROMPT),
        HumanMessage(content=question)
    ]

    response = llm_tool.invoke(messages)
    print(response)
    if hasattr(response, "tool_calls") and response.tool_calls:
        args = response.tool_calls[0]["args"]
        sql = args["sql"]
        return sql.strip().rstrip(";")

    # --- CASE 2: Model returned content directly ---
    if response.content:
        return response.content.strip().rstrip(";")

    # If both empty → failure
    return ""
