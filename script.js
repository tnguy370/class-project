// ── DOM Elements ──
var form = document.getElementById("prediction-form");
var resultBox = document.getElementById("result-box");
var resultText = document.getElementById("result-text");

var attendanceInput = document.getElementById("attendance");
var attendanceValue = document.getElementById("attendance-value");
var hoursInput = document.getElementById("hours-studied");
var scoresInput = document.getElementById("previous-scores");
var sleepInput = document.getElementById("sleep-hours");

// ── Slider live update ──
attendanceInput.addEventListener("input", function () {
    attendanceValue.textContent = attendanceInput.value;
});

// ── Form submission ──
form.addEventListener("submit", function (event) {
    event.preventDefault();

    // Clear previous error highlights
    hoursInput.classList.remove("input-error");
    scoresInput.classList.remove("input-error");
    sleepInput.classList.remove("input-error");

    // Validate text inputs (slider always has a value)
    var valid = true;
    var numberInputs = [hoursInput, scoresInput, sleepInput];

    for (var i = 0; i < numberInputs.length; i++) {
        if (numberInputs[i].value.trim() === "") {
            numberInputs[i].classList.add("input-error");
            valid = false;
        }
    }

    if (!valid) {
        hideResult();
        alert("Please fill in all fields before predicting.");
        return;
    }

    // Parse values
    var attendance = parseFloat(attendanceInput.value);
    var hoursStudied = parseFloat(hoursInput.value);
    var previousScores = parseFloat(scoresInput.value);
    var sleepHours = parseFloat(sleepInput.value);

    // Range checks
    if (hoursStudied < 0 || hoursStudied > 168) {
        hoursInput.classList.add("input-error");
        alert("Hours Studied must be between 0 and 168.");
        return;
    }
    if (previousScores < 0 || previousScores > 100) {
        scoresInput.classList.add("input-error");
        alert("Previous Scores must be between 0 and 100.");
        return;
    }
    if (sleepHours < 0 || sleepHours > 24) {
        sleepInput.classList.add("input-error");
        alert("Sleep Hours must be between 0 and 24.");
        return;
    }

    // Run prediction
    var prediction = predict(attendance, hoursStudied, previousScores, sleepHours);
    showResult(prediction);
});

// ── Simple rule-based prediction (placeholder for real model) ──
function predict(attendance, hoursStudied, previousScores, sleepHours) {
    var riskScore = 0;

    // Attendance weighs the most
    if (attendance < 70) {
        riskScore += 2;
    } else if (attendance < 80) {
        riskScore += 1;
    }

    if (hoursStudied < 10) {
        riskScore += 1;
    }

    if (previousScores < 50) {
        riskScore += 2;
    } else if (previousScores < 65) {
        riskScore += 1;
    }

    if (sleepHours < 5) {
        riskScore += 1;
    }

    return riskScore >= 3 ? "At-Risk" : "Not At-Risk";
}

// ── Show / hide result ──
function showResult(prediction) {
    resultBox.classList.remove("hidden", "at-risk", "not-at-risk");

    if (prediction === "At-Risk") {
        resultBox.classList.add("at-risk");
    } else {
        resultBox.classList.add("not-at-risk");
    }

    resultText.textContent = "Prediction: " + prediction;
}

function hideResult() {
    resultBox.classList.add("hidden");
    resultBox.classList.remove("at-risk", "not-at-risk");
}
