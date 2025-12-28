// ---------------------- LIVE CLOCK ----------------------
function updateClock() {
    const clockEl = document.getElementById('clock');
    if (clockEl) clockEl.innerText = new Date().toLocaleString();
}
setInterval(updateClock, 1000);
updateClock();

// ---------------------- DARK MODE ----------------------
const toggleBtn = document.getElementById('toggleTheme');
if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
    });
}

// ---------------------- SEARCH FILTER ----------------------
const searchInput = document.getElementById('searchInput');
if (searchInput) {
    searchInput.addEventListener('input', () => {
        const q = searchInput.value.toLowerCase();
        document.querySelectorAll('#attendanceBody tr').forEach(tr => {
            tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
        });
    });
}

// ---------------------- LIVE UPDATE ----------------------
let lastLength = document.querySelectorAll('#attendanceBody tr').length;

async function fetchAttendance() {
    try {
        const res = await fetch('/attendance_data');
        const data = await res.json();

        if (data.length > lastLength) {
            const tbody = document.getElementById('attendanceBody');
            const newRows = data.slice(lastLength);

            newRows.forEach(log => {
                const tr = document.createElement('tr');
                tr.classList.add('highlight');
                tr.innerHTML = `
                    <td><img class="face-thumb" src="/static/uploads/${log.image || ''}"></td>
                    <td>${log.username}</td>
                    <td>${log.email}</td>
                    <td>${log.date}</td>
                    <td>${log.time}</td>
                    <td></td>
                `;
                tbody.appendChild(tr);
            });

            lastLength = data.length;
        }
    } catch (e) {
        console.error("Fetch error:", e);
    }
}

setInterval(fetchAttendance, 2500);
