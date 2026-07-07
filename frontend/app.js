/* =====================================================================
   DoomRL Tactical AI Console JavaScript Controller
   Provides scenario selection state, dual-video synchronization,
   and dynamic Plotly chart switches.
   ===================================================================== */

// Keep track of active scenario
let currentScenario = 'basic';
let traceActive = false;
let traceInterval = null;

// Select scenario card and update all details on dashboard
function selectScenario(scenario) {
    currentScenario = scenario;
    
    // 1. Update Card CSS States
    document.querySelectorAll('.level-card').forEach(card => {
        card.classList.remove('active');
    });
    
    let activeCardId = 'card-basic';
    let outputDimsText = '5 Q-Values';
    
    if (scenario === 'defend_the_center') {
        activeCardId = 'card-defend';
        outputDimsText = '5 Q-Values';
    } else if (scenario === 'deadly_corridor') {
        activeCardId = 'card-corridor';
        outputDimsText = '11 Q-Values';
    }
    
    document.getElementById(activeCardId).classList.add('active');
    
    // Update DQN output layers representation
    document.getElementById('net-out-dims').innerText = outputDimsText;
    
    // 2. Switch Plotly Charts source URL
    document.getElementById('iframe-reward').src = `/charts/${scenario}_reward.html`;
    document.getElementById('iframe-loss').src = `/charts/${scenario}_loss.html`;
    document.getElementById('iframe-kills').src = `/charts/${scenario}_kills.html`;
    document.getElementById('iframe-epsilon').src = `/charts/${scenario}_epsilon.html`;
    
    // 3. Switch Video Player sources
    const vidRandom = document.getElementById('vid-random');
    const vidTrained = document.getElementById('vid-trained');
    
    let baseNameRandom = 'basic_random.mp4';
    let baseNameTrained = 'basic_trained.mp4';
    
    if (scenario === 'defend_the_center') {
        baseNameRandom = 'defend_random.mp4';
        baseNameTrained = 'defend_trained.mp4';
    } else if (scenario === 'deadly_corridor') {
        baseNameRandom = 'corridor_random.mp4';
        baseNameTrained = 'corridor_trained.mp4';
    }
    
    vidRandom.src = `/videos/${baseNameRandom}`;
    vidTrained.src = `/videos/${baseNameTrained}`;
    
    // Stop tracing animation if active
    if (traceActive) {
        toggleTraceAnimation();
    }
}

// Synchronize and play both comparison videos at the exact same moment
function syncCompareVideos() {
    const vidRandom = document.getElementById('vid-random');
    const vidTrained = document.getElementById('vid-trained');
    
    // Reset time and play
    vidRandom.currentTime = 0;
    vidTrained.currentTime = 0;
    
    vidRandom.play();
    vidTrained.play();
}

// Play/Pause Trace Animation (animates loading of Plotly iframes as if training is live)
function toggleTraceAnimation() {
    const btn = document.getElementById('btn-trace');
    traceActive = !traceActive;
    
    if (traceActive) {
        btn.innerText = "PAUSE_TRACE";
        btn.style.borderColor = "var(--doom-red)";
        btn.style.color = "var(--doom-red)";
        
        // Dynamic reloading to create scan trace visual effect
        traceInterval = setInterval(() => {
            const iframes = document.querySelectorAll('.chart-iframe');
            iframes.forEach(iframe => {
                // Slightly reload source to flash the plot trace
                const src = iframe.src;
                iframe.src = '';
                iframe.src = src;
            });
        }, 8000);
    } else {
        btn.innerText = "PLAY_TRACE";
        btn.style.borderColor = "var(--terminal-green)";
        btn.style.color = "var(--terminal-green)";
        
        clearInterval(traceInterval);
        traceInterval = null;
    }
}
