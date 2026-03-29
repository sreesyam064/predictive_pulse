/*  Functions to build  ( from html file )
    1. showSection()    :-|
    2. fillDemo()
    3. runPredict()
    4. resetForm()

    Functions to build ()
    1. collectFormData()    -> json conversion
    2. runPrediction()      -> to run prediction
    3. displayResult()      -> to display the result after prediction
    4. loadModelInfo()      -> to load model info / overfitting table
    5. showToast()          -> to create popup message
    6. setText()            -> to find element and update text

*/ 
// print(f"hi {s}") -> console.log('hi ${s}');
// .trim(); -> str.strip()
function showSection(name) {
    ['predict', 'models', 'about'].forEach((s, i) => {
        const el = document.getElementById(`section-${s}`);
        if (el) el.style.display = s === name ? '': 'none';
        document.querySelectorAll('.btn-nav')[i]?.classList.toggle('active', s===name);
    });
    document.getElementById('hero-section').style.display = name ==='predict' ? '': 'none';
    if (name==='models') loadModelInfo();
    window.scrollTo({ top:0, behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', () => {
    fetch('/model-info').then(r => r.json()).then(d => {
        const acc = d.model_results?.['Logistic Regression']?.accuracy;
        if (acc) document.getElementById('stat-acc').textContent = acc + '%';
    }).catch(err => console.error(err));
});

// Demo patient (Stage 2 case)
function fillDemo() {
    const demo = {
    gender:'male', age:'2', history:'1', patient:'1',
    when_diagnosed:'1', severity:'1', breath_shortness:'1',
    visual_changes:'0', nose_bleeding:'0',
    systolic:'2', diastolic:'2',
    take_medication:'1', controlled_diet:'0'
    };
    Object.entries(demo).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    });
    showToast('Demo patient loaded — 51-64 male, Stage 2 case', 'info');
}

// Collect & Validate Data Form
function collectFormData() {
    const fields = [
    'gender','age','history','patient','when_diagnosed',
    'severity','breath_shortness','visual_changes','nose_bleeding',
    'systolic','diastolic','take_medication','controlled_diet'
    ];
    const data = {}, missing = [];
    fields.forEach(f => {
        const el = document.getElementById(f);
        if (!el) return;
        const val = el.value.trim();
        if (val === '') missing.push(el.closest('.field')?.querySelector('label')?.childNodes[0]?.textContent?.trim() || f);
        data[f] = val;
    });
    return { data, missing };
}

// Run Prediction
async function runPrediction() {
    const { data, missing } = collectFormData();
    if (missing.length > 0) {
        showToast(
            `Please fill: ${missing.slice(0, 3).join(', ')}${missing.length > 3 ?` + ${missing.length -3} more` : ''}`, 'error');
        return;
    }
    const btn = document.getElementById('predict-btn');
    btn.classList.add('loading');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Analyzing...';
    try {
        console.log("Sending data:", data);

        const resp = await fetch('/predict', {
            method:'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!resp.ok) {
            throw new Error(`Server error: ${resp.status}`);
        }

        const result = await resp.json();
        console.log("Response:", result);

        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }
        displayResult(result);

    } catch(err) {
        console.log(err);
        showToast('Error:' + err.message, 'error');

    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
        btn.innerHTML = `
            <span class="btn-icon">
                <svg viewBox="0 0 24 24" width="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </span> Analyze Blood Pressure`;
    }
}

// Display Result
// setText() -> el.textContent = value
function displayResult(result) {
    console.log("Result:", result);

    const panel = document.getElementById('result-panel');
    const header = document.getElementById('result-header');
    const colors = { 0:'#10b981', 1:'#f97316', 2:'#ef4444', 3:'#dc2626' };
    const color = colors[result.prediction] || result.stage_color;

    header.style.borderLeft = `5px solid ${color}`;
    setText('result-icon', result.stage_icon);
    setText('result-stage', result.stage_name);
    setText('result-urgency', result.urgency);
    setText('result-description', result.description);
    setText('res-systolic', result.bp_range?.systolic);
    setText('res-diastolic', result.bp_range?.diastolic);

    const badge = document.getElementById('result-risk');
    badge.textContent = result.risk_level + 'Risk';
    badge.style.background = color;
    document.getElementById('result-model-tag').textContent = 
    `${result.model_used} · ${result.model_accuracy}% accuracy`;

    // Probability bars
    const stageColors = {
    'Normal':'#10b981',
    'Hypertension Stage 1':'#f97316',
    'Hypertension Stage 2':'#ef4444',
    'Hypertensive Crisis':'#dc2626'
  };
  const pc = document.getElementById('prob-bars');
  pc.innerHTML = '';
  Object.entries(result.probabilities).forEach(([name, pct]) => {
    const fc = stageColors[name] || '@c084fc';
    pc.innerHTML += `
    <div class="prob-row>
        <span class="prob-name">${name}</span>
        <div class="prob-track">
            <div class="prob-fill" style="width:0%; background:${fc}" data-width="${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
    </div>`;
  });
  setTimeout(() => {
    document.querySelectorAll('.prob-fill').forEach(el => {
        el.style.width = el.dataset.width;
    });
  }, 50);


  // Risk factors
  const rs = document.getElementById('risk-section');
  const rl = document.getElementById('risk-factors-list');
  rl.innerHTML= '';
  if (result.risk_factors?.length) {
    rs.style.display = '';
    result.risk_factors.forEach(({factor, impact}) => {
      rl.innerHTML += `<span class="risk-tag ${impact}">${impact === 'High'? '▲':impact === 'Moderate'?'◆': '▸'} ${factor}</span>`  
    });
  } else {
    rs.style.display = 'none';
  }

  // Recommendations
  const recoList = document.getElementById('recommendations-list');
  recoList.innerHTML = '';
  result.recommendations.forEach((rec, i) => {
    const li = document.createElement('li');
    li.textContent = rec;
    li.style.animationDelay =  `${i * 55}ms`;
    recoList.appendChild(li);
  });

  panel.style.display = '';
  setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function resetForm() {
    document.getElementById('result-panel').style.display = 'none';
    document.querySelectorAll('select').forEach(el => {
        el.value = '';
    });
    document.querySelectorAll('input').forEach(el => {
        el.value = '';
    });
    document.querySelectorAll('.field input, .field select').forEach(el => {
        el.blur();
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
    showToast('Form reset successfully', 'info');
}

// Load Model info / Overfitting Table
async function loadModelInfo() {
    const grid = document.getElementById('model-grid');
    if (grid.dataset.loaded === '1') return;
    // grid.dataset.loaded = '1';
    try {
        const resp = await fetch('/model-info');
        const data = await resp.json();
        grid.dataset.loaded = '1';
        if (!resp.ok) {
            throw new Error(`Server error: ${resp.status}`);
        }
        console.log("Model Data:", data);
        console.log('Response:', resp);
        // Overfitting Table
        const tbl = document.getElementById('overfitting-table');
        if (tbl && data.model_results) {
            let rows = '';
            Object.entries(data.model_results).forEach(([name, r]) => {
                const isBest = r.status === 'Selected';
                const isRej = r.status === 'Rejected';
                const sc = isBest ? 'status-selected' : isRej ? 'status-rejected' : 'status-considered';
                const gc = r.generalization === 'Excellent' ? 'gen-excellent' : r.generalization === 'Overfitted' ? 'gen-overfitted' : 'gen-good';
                const si = isBest ? '✅' : isRej ? '❌' : '⚠️';
                rows += `<tr class="${isBest ? 'row-best' : isRej ? 'row-rejected' : '' }">
                <td class="col-algo">${isBest ? '⭐ ' : ''}${name}</td>
                <td class="col-acc">${r.accuracy}%</td>
                <td><span class="gen-badge ${gc}">${r.generalization}</span></td>
                <td><span class="status-badge ${sc}">${si} ${r.status}</span></td>
                </tr>`;
            });
            
            tbl.innerHTML = `
            <h3 class="section-heading" style="margin-bottom:1.25rem;">Model Comparison & Overfitting Analysis</h3>
            <div class="table-wrapper">
                <table class="model-table">
                    <thead><tr><th>Algorithm</th><th>Accuracy</th><th>Generalization Assessment</th><th>Selection Status</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
            <div class="overfit-explainer">
                <div class="overfit-box rejected-box">
                    <div class="overfit-box-title">❌ Perfect Accuracy Models (100%) — Overfitting Indicators</div>
                    <ul>
                        <li>Decision Tree, Random Forest, and SVM achieved perfect test accuracy</li>
                        <li>This is a classic sign of overfitting in medical datasets</li>
                        <li>Perfect performance rarely translates to real-world clinical scenario</li>
                        <li>Models likely memorized training patterns rather than learning generalizable features</li>
                    </ul>
                </div>
                <div class="overfit-box selected-box">
                    <div class="overfit-box-title">✅ Why Logistic Regression Was Selected</div>
                    <ul>
                        <li>97.3% accuracy with excellent generalization assessment</li>
                        <li>No overfitting —consistent train and test accuracy</li>
                        <li>Low CV standard deviation (0.45%) = stable on unseen data</li>
                        <li>Interpretable coefficients support c;inical transparency</li>
                        <li>Proven reliability in medical classification literature</li>
                    </ul>
                </div>
            </div>
            <div class="consequences-box">
                <div class="overfit-box-title">⚠️ Overfitting Consequences in Medical AI</div>
                <div class="consequences-grid">
                    <div class="consequence-item"><span class="ci-icon">🏥</span><strong>Poor real-world performance</strong><p>Fails on new unseen patient data outside training distribution</p></div>
                    <div class="consequence-item"><span class="ci-icon">⚕️</span><strong>Clinical variation failure</strong><p>Cannot adapt to diverse patient demographics and presentations</p></div>
                    <div class="consequence-item"><span class="ci-icon">⚠️</span><strong>False clinical confidence</strong><p>100% accuracy misleads staff but fails on real patients</p></div>
                    <div class="consequence-item"><span class="ci-icon">🚨</span><strong>Patient safety risk</strong><p>Incorrect predictions in clinical settings can directly harm patients</p></div>
                </div>
            </div>`;

        }

        // Model Cards
        grid.innerHTML = '';
        Object.entries(data.model_results).forEach(([name, r]) => {
            const isBest = r.status === 'Selected';
            const isRej = r.status === 'Rejected';
            const barC = isBest ? 'var(--green)' : isRej ? 'var(--red)' : 'var(--amber)';
            grid.innerHTML += `
                <div class="model-card ${isBest ? 'best' : '' } ${isRej ? 'rejected' : '' }">
                     <div class="model-name">${name}</div>
                    <div class="model-metrics">
                        <div class="metric"><span class="metric-val" style="color:${barC}">${r.accuracy}%</span><span class="metric-label">Test Accuracy</span></div>
                        <div class="metric"><span class="metric-val" style="color:${barC}">${r.cv_mean}%</span><span class="metric-label">CV Mean ± ${r.cv_std}%</span></div>
                    </div>
                    <div class="model-bar-track">
                        <div class="model-bar-fill" style="width:0%;background:${barC}" data-width="${r.accuracy}%"></div>
                    </div>
                    <div class="model-status-row">
                        <span class="model-gen ${r.generalization.toLowerCase()}">${r.generalization}</span>
                        <span class="model-sel ${r.status.toLowerCase()}">${isBest ? '✅' : isRej ? '❌' : '⚠️'} ${r.status}</span>
                    </div>
                </div>`;
        });

        setTimeout(() => {
            document.querySelectorAll('.model-bar-fill').forEach(el => {
                el.style.width = el.dataset.width; });
        }, 100);

        // Feature Importance
        const fi = data.feature_importance;
        if (fi && Object.keys(fi).length) {
            const maxV = Math.max(...Object.values(fi));
            const barHtml = Object.entries(fi).map(([feat,val]) => {
                const pct = ((val / maxV) * 100).toFixed(1);
                const label = feat.replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase());
                return `<div class="feat-row">
                <span class="feat-name">${label}</span>
                <div class="feat-track"><div class="feat-fill" style="width:0%" data-width="${pct}%"></div></div>
                <span class="feat-pct">${(val * 100).toFixed(1)}%</span>
                </div>`;
            }).join('');
            document.getElementById('feature-imp-section').innerHTML = `
                <h3 class="section-heading" style="margin:2rem 0 1rem">
                    FEature Importances — Logistic Regression Coefficients
                </h3>
                <div class="feat-bars">${barHtml}</div>`;
            setTimeout(() => {
                document.querySelectorAll('.feat-fill').forEach( el => {
                    el.style.width = el.dataset.width;
                });
            }, 200);
        }

    } catch(err) {
        console.log("Model load error:", err);
        grid.innerHTML = `<p style="color:var(--text3"); padding: 1rem;">Failed to load model data</p>`;
    }
} 

function showToast(msg, type='info') {
    const existing = document.getElementById('toast');
    if (existing) existing.remove();
    const colors = {error: 'dc2626', info: '7c3aed', success: '10b981'};
    const t = document.createElement('div');
    t.id = 'toast';
    t.style.cssText = `position:fixed;bottom:2rem;right:2rem;z-index:9999;
    padding:.8rem 1.25rem;background:${colors[type]||colors.info};color:white;
    border-radius:10px;font-family:'DM Sans',sans-serif;font-size:.88rem;
    font-weight:500;box-shadow:0 8px 32px rgba(0,0,0,.4);max-width:340px;
    animation:fadeUp .3s ease;`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}