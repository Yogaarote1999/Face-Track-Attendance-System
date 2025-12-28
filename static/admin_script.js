// ------------------------------
// Live Clock
// ------------------------------
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    const dateStr = now.toLocaleDateString();
    const clockElement = document.getElementById('clock');
    if (clockElement) {
        clockElement.innerText = `${dateStr} ${timeStr}`;
    }
}

// Update clock every second
setInterval(updateClock, 1000);
updateClock();

// ------------------------------
// Dark/Light Mode Toggle
// ------------------------------
const toggleBtn = document.getElementById('toggleTheme');
if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
    });
}

// ------------------------------
// Search/Filter Table
// ------------------------------
const searchInput = document.getElementById('searchInput');
if (searchInput) {
    searchInput.addEventListener('input', () => {
        const q = searchInput.value.toLowerCase();
        document.querySelectorAll('#attendanceBody tr').forEach(tr => {
            tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
        });
    });
}

// ------------------------------
// Dynamic Attendance Table Refresh
// ------------------------------
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
                    <td><img class="face-thumb" src="/static/uploads/${log.image || ''}" alt="face"></td>
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
    } catch (err) {
        console.error("Failed to fetch attendance data:", err);
    }
}

// Refresh table every 2.5 seconds
setInterval(fetchAttendance, 2500);
