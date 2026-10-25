from database.connection import get_connection
from .sql_generator import generate_sql
from .safety import is_safe_select

def run_sql(sql: str):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def answer_question(question: str):
    sql = generate_sql(question)
    sql_clean = sql.strip().rstrip(";")

    if not is_safe_select(sql_clean):
        return {"error": "Unsafe SQL generated.", "sql": sql_clean}

    try:
        rows = run_sql(sql_clean)
        return {"sql": sql_clean, "rows": rows}
    except Exception as e:
        return {"error": str(e), "sql": sql_clean}
