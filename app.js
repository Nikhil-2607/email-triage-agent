const API_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
    fetchMetrics();
    
    document.getElementById("evaluateBtn").addEventListener("click", runEvaluation);
});

async function fetchMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        if (!response.ok) return;
        
        const data = await response.json();
        
        const totalEp = data.total_episodes || 0;
        const totalSc = data.total_score || 0.0;
        const avgRev = totalEp > 0 ? (totalSc / totalEp).toFixed(2) : "0.0";
        
        // Animate count up logic could go here, for now just innerText
        document.getElementById("totalEpisodes").innerText = totalEp;
        document.getElementById("totalScore").innerText = totalSc.toFixed(1);
        document.getElementById("avgReward").innerText = avgRev;
        
    } catch (err) {
        console.error("Failed to fetch metrics", err);
    }
}

async function runEvaluation() {
    const emailText = document.getElementById("emailText").value;
    const idealCategory = document.getElementById("idealCategory").value;
    const idealAction = document.getElementById("idealAction").value;
    const idealResolution = document.getElementById("idealResolution").value;
    
    if (!emailText.trim()) {
        alert("Please enter an email text.");
        return;
    }
    
    // UI state
    const btnText = document.querySelector(".btn-text");
    const spinner = document.querySelector(".spinner");
    const placeholder = document.getElementById("resultsPlaceholder");
    const resultsContent = document.getElementById("resultsContent");
    
    btnText.innerText = "Evaluating...";
    spinner.classList.remove("hidden");
    
    const groqKey = document.getElementById("groqKey").value;
    
    // Construct payload
    const payload = {
        email_text: emailText,
        ground_truth_category: idealCategory,
        ground_truth_action: idealAction,
        ground_truth_resolution: idealResolution,
        groq_api_key: groqKey
    };
    
    try {
        const res = await fetch(`${API_URL}/evaluate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        placeholder.classList.add("hidden");
        resultsContent.classList.remove("hidden");
        
        if (data.error) {
            document.getElementById("resCategory").innerText = "ERROR";
            document.getElementById("resAction").innerText = "ERROR";
            document.getElementById("resResolution").innerText = "ERROR";
            
            const rewardEl = document.getElementById("resReward");
            rewardEl.innerText = "N/A";
            rewardEl.classList.remove("negative");
            rewardEl.style.color = "var(--danger)";
            
            document.getElementById("resFeedback").innerText = data.error;
            return;
        }
        
        document.getElementById("resCategory").innerText = data.guess_category || "N/A";
        document.getElementById("resAction").innerText = data.guess_action || "N/A";
        document.getElementById("resResolution").innerText = data.guess_resolution || "N/A";
        
        const rewardEl = document.getElementById("resReward");
        const rewVal = parseFloat(data.reward || 0);
        rewardEl.innerText = rewVal.toFixed(1);
        if (rewVal < 0) {
            rewardEl.classList.add("negative");
            rewardEl.style.color = "var(--danger)";
        } else {
            rewardEl.classList.remove("negative");
            rewardEl.style.color = "var(--success)";
        }
        
        document.getElementById("resFeedback").innerText = data.feedback || "Evaluation complete.";
        
        // Refresh metrics
        fetchMetrics();
        
    } catch (err) {
        console.error("Evaluation error:", err);
        alert("Failed to reach API. Ensure uvicorn server is running on port 8000.");
    } finally {
        btnText.innerText = "Run Evaluation";
        spinner.classList.add("hidden");
    }
}
