   const inputEl = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const messagesEl = document.getElementById("messages");

    function addMessage({ text, role, sql, rows }) {
      const row = document.createElement("div");
      row.classList.add("message-row");
      row.classList.add(role === "user" ? "user" : "bot");

      const avatar = document.createElement("div");
      avatar.classList.add("avatar");
      avatar.classList.add(role === "user" ? "user" : "bot");
      avatar.textContent = role === "user" ? "U" : "AI";

      const bubble = document.createElement("div");
      bubble.classList.add("bubble");
      bubble.classList.add(role === "user" ? "user" : "bot");

      if (role === "user") {
        bubble.textContent = text;
      } else {
        let html = "";

        if (text) {
          html += `<div>${text}</div>`;
        }

        if (sql) {
          html += `<div class="label">Generated SQL</div>
                   <div class="sql-box">${escapeHtml(sql)}</div>`;
        }

        if (rows && Array.isArray(rows)) {
          if (rows.length > 0) {
            html += `<div class="label" style="margin-top:6px;">Result Preview (${rows.length} row${rows.length !== 1 ? "s" : ""})</div>`;
            html += `<div class="table-box"><table>`;

            const cols = Object.keys(rows[0]);
            html += "<tr>";
            cols.forEach(c => {
              html += `<th>${escapeHtml(c)}</th>`;
            });
            html += "</tr>";

            rows.forEach(r => {
              html += "<tr>";
              cols.forEach(c => {
                const v = r[c];
                html += `<td>${escapeHtml(v !== null && v !== undefined ? String(v) : "")}</td>`;
              });
              html += "</tr>";
            });

            html += "</table></div>";
          } else {
            html += `<div style="margin-top:6px;color:#9ca3af;">No rows returned.</div>`;
          }
        }

        bubble.innerHTML = html || "Done.";
      }

      row.appendChild(role === "user" ? bubble : avatar);
      row.appendChild(role === "user" ? avatar : bubble);

      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addBotLoading() {
      const row = document.createElement("div");
      row.classList.add("message-row", "bot");

      const avatar = document.createElement("div");
      avatar.classList.add("avatar", "bot");
      avatar.textContent = "AI";

      const bubble = document.createElement("div");
      bubble.classList.add("bubble", "bot");
      bubble.innerHTML = `
        Thinking
        <span class="loading-dots">
          <span></span><span></span><span></span>
        </span>
      `;

      row.appendChild(avatar);
      row.appendChild(bubble);

      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;

      return { row, bubble };
    }

    function escapeHtml(str) {
      if (str === null || str === undefined) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    async function sendMessage() {
      const q = inputEl.value.trim();
      if (!q) return;

      addMessage({ text: q, role: "user" });
      inputEl.value = "";

      const loading = addBotLoading();

      try {
        console.log("[UI] Sending to /api/ask:", q);
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
        });

        const data = await res.json();
        console.log("[UI] Response from /api/ask:", data);

        loading.row.remove();

        if (data.error) {
          addMessage({
            role: "bot",
            text: `⚠️ Error: ${data.error}`,
            sql: data.sql || null,
            rows: null,
          });
        } else {
          addMessage({
            role: "bot",
            text: "",
            sql: data.sql || "",
            rows: data.rows || [],
          });
        }
      } catch (err) {
        console.error("[UI] Failed to call /api/ask:", err);
        loading.row.remove();
        addMessage({
          role: "bot",
          text: "⚠️ Failed to reach backend /api/ask. Check console & Flask logs.",
        });
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
      }
    });

    // Example pill click
    document.querySelectorAll(".example-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        const q = pill.getAttribute("data-q");
        inputEl.value = q;
        inputEl.focus();
      });
    });