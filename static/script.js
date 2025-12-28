<script>
// ------------------------------
// Webcam & Face Capture
// ------------------------------
const video = document.getElementById('video');
const status = document.getElementById('status');
let stream = null;

const expectedUser = "{{ username }}";

// Start webcam
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
    } catch (err) {
        console.error(err);
        status.innerText = "Cannot access webcam.";
    }
}
startWebcam();

// Stop camera
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// ------------------------------
// Face attendance capture
// ------------------------------
let processing = false;
let stopLoop = false;

async function captureAndRecognize() {
    if (processing || stopLoop || !stream) return;
    processing = true;

    if (video.videoWidth === 0) {
        processing = false;
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('face_image', blob, 'face.jpg');
        formData.append('expected_name', expectedUser);

        try {
            const response = await fetch('/capture', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            switch(data.status) {

                case "marked":
                    status.innerText = `Attendance marked for ${data.username} at ${data.date} ${data.time}`;

                    // Voice notification
                    const msg = new SpeechSynthesisUtterance(`${data.username}, attendance saved!`);
                    window.speechSynthesis.speak(msg);

                    // Stop camera + loop
                    stopLoop = true;
                    stopWebcam();

                    // Redirect to login after 2 seconds
                    setTimeout(() => {
                    window.location.href = data.redirect;  // /logout → goes to login
                    }, 2000);

                    return;


                case "already":
                    status.innerText = `Attendance already marked`;
                    stopWebcam();
                    stopLoop = true;
                    setTimeout(() => {
                        window.location.href = "/admin";
                    }, 2000);
                    return;

                case "user_not_registered":
                    status.innerText = "User not registered! Redirecting...";
                    stopWebcam();
                    stopLoop = true;
                    setTimeout(() => {
                        window.location.href = "/register";
                    }, 2000);
                    return;

                case "face_mismatch":
                    status.innerText = "⚠ Face not matching!";
                    break;

                default:
                    status.innerText = "Face not recognized!";
            }

        } catch (e) {
            status.innerText = "Server error";
        }

        setTimeout(() => { processing = false; }, 2000);
    }, 'image/jpeg');
}

// Auto scan loop
async function autoCaptureLoop() {
    if (!stopLoop) {
        await captureAndRecognize();
        setTimeout(autoCaptureLoop, 3000);
    }
}
autoCaptureLoop();

</script>
