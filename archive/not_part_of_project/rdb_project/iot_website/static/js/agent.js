async function askAgent() {
    const question = document.getElementById("question").value;

    const response = await fetch("/api/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ question })
    });

    const data = await response.json();

    document.getElementById("sql_output").innerText = data.sql || "No SQL";
    document.getElementById("result_output").innerText = JSON.stringify(data.rows, null, 2) || data.error;
}
