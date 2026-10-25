from flask import Flask, request
import mysql.connector
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(".env")

# =========================
#  LLM (Qwen)
# =========================
llm = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-04-28",
    temperature=0,
    max_tokens=3000
)

# =========================
#  Flask App
# =========================
app = Flask(__name__)

# =========================
#  MySQL Connection
# =========================
def run_sql(sql):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="iot"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# =========================
#  Generate SQL from Question
# =========================
def generate_sql(question):
    prompt = f"""
    You are an expert MySQL query generator.
    Convert the user question into a SQL query
    using the following schema:

    TABLES:
    - community(community_id, community_name, lat, lon)
    - space(space_uuid, space_name, community_id, area, created_at)
    - subspace(subspace_uuid, subspace_name, space_uuid, created_at)
    - category(category_code, category_name)
    - product(product_uuid, product_name, cat_name, prod_type, provider_type, created_at)
    - device_tag(tag_uuid, tag_name, created_at, updated_at)
    - device(device_uuid, tag_uuid, device_name, serial_number, is_online, time_zone,
             space_uuid, subspace_uuid, product_uuid, ip, created_at, updated_at)
    - device_status(status_id, device_uuid, status_code, status_value, recorded_at)
    - automation(automation_uuid, automation_name, automation_status, automation_type,
                 space_uuid, community_id)
    - monthly_electricity_consumption(id, space_uuid, billing_year, billing_month,
                                      total_kwh, estimated_cost_aed)

    RULES:
    - Return ONLY raw SQL.
    - Do NOT include backticks.
    - Do NOT add explanation.
    - The query must be safe and syntactically valid.

    USER QUESTION:
    {question}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# =========================
#  Summarize SQL Results
# =========================
def summarize(rows, question):
    prompt = f"""
    Summarize the SQL query results in clear conversational English.
    User question: "{question}"

    SQL returned these rows:
    {rows}

    Provide an easy-to-understand answer.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# =========================
#  API Endpoint
# =========================
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "")

    # 1. Convert Question → SQL
    sql = generate_sql(user_question)

    # Prevent DELETE, DROP, UPDATE for safety
    forbidden = ["drop", "delete", "update", "insert", "alter"]
    if any(word in sql.lower() for word in forbidden):
        return {"error": "Unsafe SQL detected", "sql": sql}

    # 2. Execute SQL
    try:
        rows = run_sql(sql)
    except Exception as e:
        return {"error": str(e), "sql": sql}

    # 3. Summarize
    answer = summarize(rows, user_question)

    return {
        "question": user_question,
        "sql": sql,
        "answer": answer,
        "data": rows
    }

# =========================
#  Run Server
# =========================
if __name__ == "__main__":
    app.run(port=5000, debug=True)
