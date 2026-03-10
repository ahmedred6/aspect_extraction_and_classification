from flask import Flask, request, jsonify, send_from_directory
from agent.llm_agent import answer_question
import os
from agent.llm_agent import run_sql
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../iot_website/backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # .../iot_website
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")       # .../iot_website/frontend
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")           # .../iot_website/static

print("Frontend directory:", FRONTEND_DIR)
print("Static directory:", STATIC_DIR)

app = Flask(
    __name__,
    static_folder=STATIC_DIR,       # serve /static from here
    static_url_path="/static"
)

# -------- API: LLM Agent --------
@app.route("/api/ask", methods=["POST"])
def ask_agent():
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    return jsonify(answer_question(question)), 200


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

# -------- API: Get communities ----------
@app.route("/api/communities")
def get_communities():
    sql = "SELECT community_id, community_name FROM community;"
    return jsonify(run_sql(sql))

@app.route("/api/community/<comm_id>/spaces")
def get_spaces(comm_id):
    print("Received comm_id:", comm_id)

    sql = f"""
        SELECT space_uuid, space_name
        FROM space
        WHERE community_id = '{comm_id}';
    """

    rows = run_sql(sql)
    return jsonify(rows)


@app.route("/api/space/<space_id>/devices")
def get_devices_in_space(space_id):

    sql_dev = f"""
        SELECT device_uuid, device_name, is_online
        FROM device
        WHERE space_uuid = '{space_id}';
    """
    devices = run_sql(sql_dev)

    if not devices:
        return jsonify([])

    ids = [d["device_uuid"] for d in devices]
    id_list = ",".join([f"'{i}'" for i in ids])

    sql_status = f"""
        SELECT ds.device_uuid, ds.status_code, ds.status_value, ds.recorded_at
        FROM device_status ds
        JOIN (
            SELECT device_uuid, MAX(recorded_at) AS latest_time
            FROM device_status
            WHERE device_uuid IN ({id_list})
            GROUP BY device_uuid
        ) latest
        ON ds.device_uuid = latest.device_uuid
        AND ds.recorded_at = latest.latest_time;
    """

    statuses = run_sql(sql_status)
    status_map = {s["device_uuid"]: s for s in statuses}

    results = []
    for d in devices:
        s = status_map.get(d["device_uuid"], {})
        results.append({
            "device_uuid": d["device_uuid"],
            "device_name": d["device_name"],
            "is_online": d["is_online"],
            "last_status_code": s.get("status_code"),
            "last_status_value": s.get("status_value"),
            "last_seen": s.get("recorded_at"),
        })

    return jsonify(results)


# -------- API: Community Overview Dashboard ----------
@app.route("/api/community/<comm_id>")
def community_overview(comm_id):

    result = {}

    # Total devices in community
    q1 = f"""
        SELECT COUNT(*) AS total
        FROM device d
        JOIN space s ON d.space_uuid = s.space_uuid
        WHERE s.community_id = '{comm_id}';
    """
    result["total_devices"] = run_sql(q1)[0]["total"]

    # Total spaces
    q2 = f"""
        SELECT COUNT(*) AS total
        FROM space
        WHERE community_id = '{comm_id}';
    """
    result["total_spaces"] = run_sql(q2)[0]["total"]

    # Total subspaces
    q3 = f"""
        SELECT COUNT(*) AS total
        FROM subspace ss
        JOIN space s ON ss.space_uuid = s.space_uuid
        WHERE s.community_id = '{comm_id}';
    """
    result["total_subspaces"] = run_sql(q3)[0]["total"]

    # Devices per space
    q4 = f"""
        SELECT s.space_name, COUNT(*) AS device_count
        FROM device d
        JOIN space s ON d.space_uuid = s.space_uuid
        WHERE s.community_id = '{comm_id}'
        GROUP BY s.space_uuid
        ORDER BY device_count DESC;
    """
    result["devices_per_space"] = run_sql(q4)

    # Subspaces per space
    q5 = f"""
        SELECT s.space_name, COUNT(ss.subspace_uuid) AS subspace_count
        FROM space s
        LEFT JOIN subspace ss ON ss.space_uuid = s.space_uuid
        WHERE s.community_id = '{comm_id}'
        GROUP BY s.space_uuid;
    """
    result["subspaces_per_space"] = run_sql(q5)

    # Electricity consumption
    q6 = f"""
        SELECT SUM(m.total_kwh) AS total_kwh
        FROM monthly_electricity_consumption m
        JOIN space s ON m.space_uuid = s.space_uuid
        WHERE s.community_id = '{comm_id}';
    """
    result["total_kwh"] = run_sql(q6)[0]["total_kwh"]

    return jsonify(result)
# -------- Pages (HTML) --------
@app.route("/")
def index_page():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_page(filename):
    # e.g. /devices.html, /dashboard.html, etc.
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
