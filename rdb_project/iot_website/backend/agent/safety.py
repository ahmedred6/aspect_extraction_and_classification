def is_safe_select(sql: str) -> bool:
    lowered = sql.strip().lower()

    if not lowered.startswith("select"):
        return False

    forbidden = ["drop ", "delete ", "update ", "insert ", "alter ", "truncate ", "create "]
    if ";" in lowered.strip().rstrip(";"):
        return False

    for word in forbidden:
        if word in lowered:
            return False

    return True
