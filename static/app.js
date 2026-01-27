import { CodeJar } from 'https://medv.io/codejar/codejar.js';

const jar = CodeJar(document.getElementById('editor'), (e) => {
    e.textContent = e.textContent;
    hljs.highlightElement(e);
}, { tab: '  ' });

let currentMode = 'json'; 
let store = { json: null, prompt: null, tags: null, file: null, url: null };
let allPersonas = [];
let scanFile = null;
let wardrobeFile = null;  // FIX: Added missing variable declaration
let inputMode = 'image'; 
let rescanTargetId = null;
let wardrobeItems = [];

const els = {
    modeImage: document.getElementById('mode-image'),
    modeText: document.getElementById('mode-text'),
    inputImgCont: document.getElementById('input-container-image'),
    inputTextCont: document.getElementById('input-container-text'),
    textPromptInput: document.getElementById('text-prompt-input'),
    visionModel: document.getElementById('model-vision'),
    writerModel: document.getElementById('model-writer'),
    taggerModel: document.getElementById('model-tagger'),
    tabJson: document.getElementById('tab-json'),
    tabPrompt: document.getElementById('tab-prompt'),
    tabTags: document.getElementById('tab-tags'),
    actionBtn: document.getElementById('action-btn'),
    actionText: document.getElementById('action-btn-text'),
    editor: document.getElementById('editor'),
    loader: document.getElementById('loader'),
    loaderText: document.getElementById('loader-text'),
    loaderSub: document.getElementById('loader-sub'),
    refinerBar: document.getElementById('refiner-bar'),
    refineInput: document.getElementById('refine-input'),
    refineBtn: document.getElementById('refine-btn'),
    file: document.getElementById('file-input'),
    drop: document.getElementById('drop-zone'),
    preview: document.getElementById('preview-container'),
    empty: document.getElementById('empty-state'),
    clearImg: document.getElementById('clear-image'),
    thumbBlur: document.getElementById('thumb-blur'),
    thumbMain: document.getElementById('thumb-main'),
    urlInput: document.getElementById('url-input'),
    urlBtn: document.getElementById('url-load-btn'),
    persona: document.getElementById('persona-select'),
    
    // WARDROBE
    wardrobeTrigger: document.getElementById('wardrobe-trigger'),
    wardrobeBrowser: document.getElementById('wardrobe-browser'),
    wardrobeGrid: document.getElementById('wardrobe-grid'),
    wardrobeIdHidden: document.getElementById('wardrobe-id-hidden'),
    activeWardrobeName: document.getElementById('active-wardrobe-name'),
    activeWardrobeImg: document.getElementById('active-wardrobe-img'),
    browserAddBtn: document.getElementById('browser-add-btn'),
    wardrobeSearch: document.getElementById('wardrobe-search'),
    wardrobeEmpty: document.getElementById('wardrobe-empty'),
    wardrobeCount: document.getElementById('wardrobe-count'),
    wardrobeModal: document.getElementById('wardrobe-modal'),
    closeWardrobe: document.getElementById('close-wardrobe'),
    wName: document.getElementById('w-name'),
    wFile: document.getElementById('w-file'),
    wUrl: document.getElementById('w-url'),
    wUpload: document.getElementById('w-upload-zone'),
    wPreview: document.getElementById('w-preview'),
    saveWardrobeBtn: document.getElementById('save-wardrobe-btn'),
    closeBrowser: document.getElementById('close-browser'),

    time: document.getElementById('time-select'),
    ratio: document.getElementById('ratio-select'),
    styleNarrative: document.getElementById('style-select'), 
    styleVisual: document.getElementById('style-manual-select'),
    quality: document.getElementById('quality-select'),
    refMode: document.getElementById('reference-mode'),
    hairStyle: document.getElementById('hair-style-select'),
    hairColor: document.getElementById('hair-color-select'),
    makeup: document.getElementById('makeup-select'),
    glasses: document.getElementById('glasses-select'),
    expr: document.getElementById('expr-select'),
	hairSource: document.getElementById('hair-source-select'),
    addBtn: document.getElementById('add-persona-btn'),
    manageBtn: document.getElementById('manage-personas-btn'),
    modal: document.getElementById('persona-modal'),
    closeModal: document.getElementById('close-modal'),
    scanBtn: document.getElementById('scan-btn'),
    pName: document.getElementById('p-name'),
    scanInput: document.getElementById('scan-input'),
    scanDrop: document.getElementById('scan-drop'),
    manualFields: document.getElementById('manual-fields'),
    editPid: document.getElementById('edit-pid'),
    modalTitle: document.getElementById('modal-title-text'),
    updateManualBtn: document.getElementById('update-manual-btn'),
    managerModal: document.getElementById('manager-modal'),
    closeManager: document.getElementById('close-manager'),
    managerList: document.getElementById('manager-list'),
    historyList: document.getElementById('history-list'),
    clearHistory: document.getElementById('clear-history-btn'),
    downloadBtn: document.getElementById('download-btn'),
    copyBtn: document.getElementById('copy-btn'),
    flushBtn: document.getElementById('flush-btn'),
    pAge: document.getElementById('p-age'),
    pEthnicity: document.getElementById('p-ethnicity'),
    pBuild: document.getElementById('p-build'),
    pChest: document.getElementById('p-chest'),
    pShoulders: document.getElementById('p-shoulders'),
    pWaist: document.getElementById('p-waist'),
    pFace: document.getElementById('p-face'),
    pSkin: document.getElementById('p-skin'),
    pEyes: document.getElementById('p-eyes'),
    pNose: document.getElementById('p-nose'),
    pLips: document.getElementById('p-lips'),
    pHairColor: document.getElementById('p-hair-color'),
    pHairStyle: document.getElementById('p-hair-style'),
    pTattoos: document.getElementById('p-tattoos'),
    pEyewear: document.getElementById('p-eyewear'),
    pMakeup: document.getElementById('p-makeup'), 
    scanMsg: document.getElementById('scan-msg'),
    scanPreviewCont: document.getElementById('scan-preview-container'),
    scanBlur: document.getElementById('scan-blur'),
    scanMain: document.getElementById('scan-main'),
    triggerRescan: document.getElementById('trigger-rescan'),
    triggerNewPhoto: document.getElementById('trigger-new-photo'),
    editPhotoButtons: document.getElementById('edit-photo-buttons'),
    settingsBtn: document.getElementById('settings-btn'),
    settingsModal: document.getElementById('settings-modal'),
    saveKeysBtn: document.getElementById('save-keys-btn'),
    closeSettings: document.getElementById('close-settings'),
    keyGoogle: document.getElementById('key-google'),
    keyFal: document.getElementById('key-fal'),
    keyXai: document.getElementById('key-xai'),
    genCloudBtn: document.getElementById('generate-cloud-btn'),
    cloudModel: document.getElementById('cloud-model'),
    imgModal: document.getElementById('img-result-modal'),
    closeImgModal: document.getElementById('close-img-modal'),
    cloudLoader: document.getElementById('cloud-loader'),
    cloudResult: document.getElementById('cloud-result-img'),
    cloudError: document.getElementById('cloud-error'),
    openStyleBtn: document.getElementById('open-style-btn'),
    deleteStyleBtn: document.getElementById('delete-style-btn'),
    styleModal: document.getElementById('style-modal'),
    styleInputRaw: document.getElementById('style-input-raw'),
    analyzeStyleBtn: document.getElementById('analyze-style-btn'),
    styleAnalysisResult: document.getElementById('style-analysis-result'),
    styleInstruction: document.getElementById('style-instruction'),
    styleName: document.getElementById('style-name'),
    saveStyleBtn: document.getElementById('save-style-btn'),
    closeStyleModal: document.getElementById('close-style-modal')
};

// --- WARDROBE LOGIC ---

async function loadWardrobe() {
    try {
        const res = await fetch('/wardrobe');
        if (!res.ok) throw new Error("Backend offline");
        const rawData = await res.json();
        
        // Handle both Array and Dict returns safely
        if (Array.isArray(rawData)) {
            wardrobeItems = rawData;
        } else if (typeof rawData === 'object' && rawData !== null) {
            wardrobeItems = Object.values(rawData); 
        } else {
            wardrobeItems = [];
        }
        
        renderWardrobeGrid(wardrobeItems);
        
        const savedId = localStorage.getItem('pref_wardrobe_id');
        if(savedId && savedId !== 'none') {
            const item = wardrobeItems.find(i => i.id === savedId);
            if(item) selectOutfit(item.id, item.name, item.image, false); 
        }
    } catch(e) { console.error("Wardrobe error", e); renderWardrobeGrid([]); }
}

function renderWardrobeGrid(items) {
    els.wardrobeGrid.innerHTML = '';
    
    const noneCard = document.createElement('div');
    noneCard.className = `group relative bg-[#161b22] rounded-xl border border-white/10 overflow-hidden cursor-pointer hover:border-slate-500 transition-all flex flex-col items-center justify-center h-56 ${els.wardrobeIdHidden.value === 'none' ? 'ring-2 ring-pink-500 border-pink-500' : ''}`;
    noneCard.innerHTML = `<i class="fas fa-ban text-3xl text-slate-600 group-hover:text-white mb-3 transition-colors"></i><span class="text-xs font-bold text-slate-500 group-hover:text-white">No Outfit</span>`;
    noneCard.onclick = () => selectOutfit('none', 'None (Default)', null);
    els.wardrobeGrid.appendChild(noneCard);

    if(items.length === 0) els.wardrobeEmpty.classList.remove('hidden');
    else els.wardrobeEmpty.classList.add('hidden');

    items.forEach(item => {
        const isSelected = els.wardrobeIdHidden.value === item.id;
        const card = document.createElement('div');
        card.className = `group relative bg-[#161b22] rounded-xl border border-white/10 overflow-hidden cursor-pointer hover:border-pink-500/50 hover:shadow-lg hover:shadow-pink-900/10 transition-all flex flex-col h-56 ${isSelected ? 'ring-2 ring-pink-500 border-pink-500' : ''}`;
        card.innerHTML = `
            <div class="h-40 bg-slate-800 relative overflow-hidden">
                <img src="${item.image}" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" loading="lazy">
                <div class="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors"></div>
                ${isSelected ? '<div class="absolute top-2 right-2 bg-pink-500 text-white text-[10px] font-bold px-2 py-1 rounded shadow-lg"><i class="fas fa-check"></i> WORN</div>' : ''}
            </div>
            <div class="flex-1 p-3 flex flex-col justify-between bg-[#161b22]">
                <h4 class="text-xs font-bold text-slate-200 truncate group-hover:text-pink-400 transition-colors">${item.name}</h4>
                <div class="flex justify-between items-center mt-2">
                    <span class="text-[9px] text-slate-600 uppercase tracking-wider">OUTFIT</span>
                    <button class="delete-btn text-slate-600 hover:text-red-500 transition-colors p-1" title="Delete"><i class="fas fa-trash text-xs"></i></button>
                </div>
            </div>
        `;
        card.addEventListener('click', (e) => {
            if(e.target.closest('.delete-btn')) return;
            selectOutfit(item.id, item.name, item.image);
        });
        const delBtn = card.querySelector('.delete-btn');
        delBtn.onclick = async (e) => {
            e.stopPropagation();
            if(!confirm(`Delete "${item.name}"?`)) return;
            await fetch(`/wardrobe/${item.id}`, { method: 'DELETE' });
            if(isSelected) selectOutfit('none', 'None (Default)', null, false);
            await loadWardrobe();
        };
        els.wardrobeGrid.appendChild(card);
    });
    els.wardrobeCount.innerText = `${items.length} items`;
}

function selectOutfit(id, name, image, close = true) {
    els.wardrobeIdHidden.value = id;
    localStorage.setItem('pref_wardrobe_id', id);
    els.activeWardrobeName.innerText = name;
    if (image) {
        els.activeWardrobeImg.innerHTML = '';
        els.activeWardrobeImg.style.backgroundImage = `url('${image}')`;
        els.activeWardrobeName.classList.add('text-pink-400');
    } else {
        els.activeWardrobeImg.style.backgroundImage = 'none';
        els.activeWardrobeImg.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-slate-600 text-xs bg-[#0d1117]"><i class="fas fa-ban"></i></div>';
        els.activeWardrobeName.classList.remove('text-pink-400');
    }
    renderWardrobeGrid(wardrobeItems);
    
    // --- FIX: Centering logic ---
    if(close) {
        els.wardrobeBrowser.classList.add('hidden');
        els.wardrobeBrowser.classList.remove('flex'); // remove flex to hide properly
    }
}

// Handlers
els.wardrobeTrigger.onclick = () => {
    els.wardrobeBrowser.classList.remove('hidden');
    els.wardrobeBrowser.classList.add('flex'); // add flex to center properly
    loadWardrobe(); 
};
els.closeBrowser.onclick = () => {
    els.wardrobeBrowser.classList.add('hidden');
    els.wardrobeBrowser.classList.remove('flex');
};
els.wardrobeBrowser.onclick = (e) => { 
    if(e.target === els.wardrobeBrowser) {
        els.wardrobeBrowser.classList.add('hidden');
        els.wardrobeBrowser.classList.remove('flex');
    } 
};
els.browserAddBtn.onclick = () => {
    els.wardrobeBrowser.classList.add('hidden'); 
    els.wardrobeBrowser.classList.remove('flex');
    els.wardrobeModal.classList.remove('hidden');
};
els.wardrobeSearch.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = wardrobeItems.filter(i => i.name.toLowerCase().includes(term));
    renderWardrobeGrid(filtered);
});
els.closeWardrobe.onclick = () => els.wardrobeModal.classList.add('hidden');

// Wardrobe Image
els.wUpload.onclick = () => els.wFile.click();
els.wFile.onchange = (e) => {
    if(e.target.files[0]) {
        wardrobeFile = e.target.files[0];
        els.wPreview.style.backgroundImage = `url(${URL.createObjectURL(wardrobeFile)})`;
        els.wUrl.value = "";
    }
};
els.wUrl.addEventListener('input', (e) => {
    if(e.target.value) {
        els.wPreview.style.backgroundImage = `url(${e.target.value})`;
        wardrobeFile = null;
    }
});

// Wardrobe Save
els.saveWardrobeBtn.onclick = async () => {
    const name = els.wName.value.trim();
    if (!name) return showNotification("⚠️ Name required", "warning");
    els.saveWardrobeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ANALYZING...';
    els.saveWardrobeBtn.disabled = true;
    const fd = new FormData();
    fd.append('name', name);
    if(wardrobeFile) fd.append('file', wardrobeFile);
    if(els.wUrl.value) fd.append('image_url', els.wUrl.value);
    fd.append('model', els.visionModel.value);
    try {
        const res = await fetch('/wardrobe/create', { method: 'POST', body: fd });
        const data = await res.json();
        if(data.status === 'success') {
            showNotification("✨ Outfit Added!", "success");
            els.wName.value = ""; els.wUrl.value = ""; els.wPreview.style.backgroundImage = ""; wardrobeFile = null;
            els.wardrobeModal.classList.add('hidden'); 
            await loadWardrobe();
            els.wardrobeBrowser.classList.remove('hidden'); 
            els.wardrobeBrowser.classList.add('flex');
        } else throw new Error(data.message);
    } catch(e) { showNotification(`❌ ${e.message}`, "error"); } 
    finally { els.saveWardrobeBtn.innerHTML = 'ANALYZE & SAVE OUTFIT'; els.saveWardrobeBtn.disabled = false; }
};

// --- GENERIC HANDLERS ---
function bindSticky(element, key, defaultValue = "") {
    if (!element) return;
    const saved = localStorage.getItem('pref_' + key);
    if (saved !== null) {
        if (element.type === 'checkbox') element.checked = (saved === 'true');
        else element.value = saved;
    } else if (defaultValue) element.value = defaultValue;
    element.addEventListener('change', () => {
        const val = (element.type === 'checkbox') ? element.checked : element.value;
        localStorage.setItem('pref_' + key, val);
    });
}

els.tabJson.onclick = () => switchTab('json');
els.tabPrompt.onclick = () => switchTab('prompt');
els.tabTags.onclick = () => switchTab('tags');
els.refineInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') els.refineBtn.click(); });

function handleFile(f) {
    if(!f) return;
    store.file = f; store.url = null; els.urlInput.value = "";
    const r = new FileReader();
    r.onload = (ev) => {
        els.thumbBlur.src = ev.target.result;
        els.thumbMain.src = ev.target.result;
        els.preview.classList.remove('hidden');
        els.empty.classList.add('hidden');
    };
    r.readAsDataURL(f);
}

els.file.onchange = (e) => handleFile(e.target.files[0]);
els.drop.onclick = () => els.file.click();
els.drop.addEventListener('dragover', (e) => { e.preventDefault(); els.drop.classList.add('drop-zone-active'); });
els.drop.addEventListener('dragleave', () => els.drop.classList.remove('drop-zone-active'));
els.drop.addEventListener('drop', (e) => { e.preventDefault(); els.drop.classList.remove('drop-zone-active'); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
els.clearImg.onclick = (e) => { e.stopPropagation(); store.file = null; store.url = null; els.urlInput.value = ""; els.preview.classList.add('hidden'); els.empty.classList.remove('hidden'); };
els.urlBtn.onclick = () => { const url = els.urlInput.value.trim(); if(!url) return; store.url = url; store.file = null; els.thumbBlur.src = url; els.thumbMain.src = url; els.preview.classList.remove('hidden'); els.empty.classList.add('hidden'); };
els.urlInput.addEventListener('keypress', (e) => { if(e.key === 'Enter') els.urlBtn.click(); });

els.copyBtn.addEventListener('click', () => {
    let text = "";
    const editorContent = document.getElementById('editor').innerText;
    if (currentMode === 'json') text = jar.toString() || editorContent; else text = editorContent;
    if (!text || text.trim().startsWith("//") || text.trim().length === 0) return showNotification("⚠️ Nothing to copy", "warning");
    navigator.clipboard.writeText(text);
    const orig = els.copyBtn.innerHTML;
    els.copyBtn.innerHTML = '<i class="fas fa-check"></i> COPIED';
    els.copyBtn.classList.add('text-green-400', 'border-green-400');
    setTimeout(() => { els.copyBtn.innerHTML = orig; els.copyBtn.classList.remove('text-green-400', 'border-green-400'); }, 1500);
});

els.downloadBtn.onclick = () => {
    let content = "", ext = "txt";
    if (currentMode === 'json') { content = jar.toString(); ext = "json"; } 
    else { content = document.getElementById('editor').innerText; ext = "txt"; }
    if(!content || content.startsWith("//")) return showNotification("⚠️ Nothing to download", "warning");
    const blob = new Blob([content], {type: 'text/plain'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `prompt_director_${Date.now()}.${ext}`; a.click();
    showNotification("📥 Download Started", "success");
};

if(els.flushBtn) {
    els.flushBtn.onclick = async () => {
        showNotification("🧹 Cleaning VRAM...", "info");
        try { await fetch('/system/free-vram', {method: 'POST'}); showNotification("✨ Memory Freed", "success"); } catch(e) { showNotification("❌ Cleanup Failed", "error"); }
    };
}

els.scanInput.onchange = (e) => {
    if(e.target.files[0]) {
        scanFile = e.target.files[0];
        const r = new FileReader();
        r.onload = (ev) => { 
            els.scanBlur.src = ev.target.result; 
            els.scanMain.src = ev.target.result; 
            els.scanPreviewCont.classList.remove('hidden'); 
            els.scanMsg.classList.add('hidden'); 
            els.scanBtn.classList.remove('hidden'); 
            els.scanDrop.classList.add('hidden');
            els.editPhotoButtons.classList.add('hidden');
        };
        r.readAsDataURL(scanFile);
    }
};

function setInputMode(mode) {
    inputMode = mode;
    if (mode === 'image') {
        els.modeImage.className = "flex-1 py-1.5 text-[10px] font-bold rounded bg-cyan-600 text-white shadow-sm transition-all";
        els.modeText.className = "flex-1 py-1.5 text-[10px] font-bold rounded text-slate-400 hover:text-white transition-all";
        els.inputImgCont.classList.remove('hidden'); els.inputTextCont.classList.add('hidden'); els.actionText.innerText = "ANALYZE IMAGE";
    } else {
        els.modeText.className = "flex-1 py-1.5 text-[10px] font-bold rounded bg-cyan-600 text-white shadow-sm transition-all";
        els.modeImage.className = "flex-1 py-1.5 text-[10px] font-bold rounded text-slate-400 hover:text-white transition-all";
        els.inputTextCont.classList.remove('hidden'); els.inputImgCont.classList.add('hidden'); els.actionText.innerText = "PARSE & APPLY";
    }
}
els.modeImage.onclick = () => setInputMode('image');
els.modeText.onclick = () => setInputMode('text');

els.textPromptInput.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            const file = item.getAsFile();
            if (file) {
                setInputMode('image');
                store.file = file;
                store.url = null;
                const reader = new FileReader();
                reader.onload = (ev) => {
                    els.thumbBlur.src = ev.target.result;
                    els.thumbMain.src = ev.target.result;
                    els.preview.classList.remove('hidden');
                    els.empty.classList.add('hidden');
                };
                reader.readAsDataURL(file);
                showNotification('📋 Image pasted! Switched to IMAGE mode', 'success');
            }
            return;
        }
    }
});

function showNotification(message, type = 'info', duration = 3000) {
    const n = document.createElement('div');
    n.className = `fixed top-4 right-4 z-[9999] px-6 py-4 rounded-lg border shadow-lg transform transition-all duration-300 translate-x-[400px] bg-[#161b22] ${type === 'success' ? 'border-green-500/30 text-green-400' : type === 'error' ? 'border-red-500/30 text-red-400' : type === 'warning' ? 'border-yellow-500/30 text-yellow-400' : 'border-cyan-500/30 text-cyan-400'}`;
    n.innerHTML = `<div class="flex items-center gap-3"><div class="text-lg">${type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}</div><span class="text-sm font-medium">${message}</span></div>`;
    document.body.appendChild(n);
    setTimeout(() => n.style.transform = 'translateX(0)', 10);
    setTimeout(() => { n.style.transform = 'translateX(400px)'; setTimeout(() => n.remove(), 300); }, duration);
}

function showLoading(title, sub) { els.loaderText.innerText = title; els.loaderSub.innerText = sub; els.loader.classList.remove('hidden'); els.actionBtn.disabled = true; }
function hideLoading() { els.loader.classList.add('hidden'); els.actionBtn.disabled = false; }

function renderManager() {
    els.managerList.innerHTML = allPersonas.map(p => `
        <div class="flex justify-between items-center bg-[#0d1117] p-3 rounded border border-white/5 hover:border-purple-500/30 group transition-all mb-2">
            <div class="flex items-center gap-3">
                <div class="relative w-8 h-8"><img src="/persona-image/${p.id}?t=${Date.now()}" class="w-8 h-8 rounded-full object-cover bg-slate-700" onerror="this.style.display='none'"></div>
                <span class="text-sm text-slate-200 font-medium">${p.name}</span>
            </div>
            <div class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button onclick="window.editPersona('${p.id}')" class="text-xs bg-slate-800 hover:bg-cyan-600 text-slate-400 hover:text-white px-2 py-1 rounded"><i class="fas fa-pen"></i></button>
                <button onclick="window.deletePersona('${p.id}')" class="text-xs bg-slate-800 hover:bg-red-600 text-slate-400 hover:text-white px-2 py-1 rounded"><i class="fas fa-trash"></i></button>
            </div>
        </div>
    `).join('');
}

async function loadPersonas() {
    try {
        const res = await fetch('/personas'); allPersonas = await res.json();
        els.persona.innerHTML = `<option value="none">No Persona (Auto-detect)</option>` + allPersonas.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
        const savedPersona = localStorage.getItem('pref_persona');
        if(savedPersona && allPersonas.some(p => p.id === savedPersona)) els.persona.value = savedPersona;
        els.persona.addEventListener('change', () => localStorage.setItem('pref_persona', els.persona.value));
        renderManager();
    } catch(e) { console.error(e); }
}

async function updateHistoryUI() {
    const res = await fetch('/history'); const h = await res.json();
    els.historyList.innerHTML = h.map((item, i) => `<div onclick="window.loadH(${i})" class="p-2 mb-1 rounded bg-[#0d1117] border border-white/5 hover:border-cyan-500/50 cursor-pointer text-[10px] text-slate-300 truncate transition-colors"><span class="text-cyan-500 mr-2">${item.timestamp}</span> ${item.filename}</div>`).join('');
    window.histData = h;
}

function switchTab(mode) {
    currentMode = mode;
    [els.tabJson, els.tabPrompt, els.tabTags].forEach(t => t.className = "px-4 py-1.5 text-xs font-bold rounded transition-all flex items-center gap-2 text-slate-400 hover:text-white border border-transparent");
    if (mode === 'json') { els.tabJson.className = "px-4 py-1.5 text-xs font-bold rounded transition-all flex items-center gap-2 tab-active-json border border-cyan-500/50"; els.actionText.innerText = "ANALYZE IMAGE"; els.actionBtn.className = "w-full py-4 font-bold text-xs uppercase tracking-widest rounded-lg shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white"; els.refinerBar.classList.remove('hidden'); if (store.json) jar.updateCode(JSON.stringify(store.json, null, 2)); else document.getElementById('editor').textContent = "// Ready."; } 
    else if (mode === 'prompt') { els.tabPrompt.className = "px-4 py-1.5 text-xs font-bold rounded transition-all flex items-center gap-2 tab-active-prompt border border-green-500/50"; els.actionText.innerText = "GENERATE PROMPT"; els.actionBtn.className = "w-full py-4 font-bold text-xs uppercase tracking-widest rounded-lg shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-500 text-white"; els.refinerBar.classList.add('hidden'); if (store.prompt) document.getElementById('editor').textContent = store.prompt; else document.getElementById('editor').textContent = "// Ready."; } 
    else if (mode === 'tags') { els.tabTags.className = "px-4 py-1.5 text-xs font-bold rounded transition-all flex items-center gap-2 tab-active-tags border border-orange-500/50"; els.actionText.innerText = "GENERATE TAGS"; els.actionBtn.className = "w-full py-4 font-bold text-xs uppercase tracking-widest rounded-lg shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2 bg-orange-600 hover:bg-orange-500 text-white"; els.refinerBar.classList.add('hidden'); if (store.tags) document.getElementById('editor').textContent = store.tags; else document.getElementById('editor').textContent = "// Ready."; }
}

function openModal(mode, pid=null) {
    els.modal.classList.remove('hidden'); scanFile = null;
    if (mode === 'create') { 
        els.modalTitle.innerText = "New Persona"; 
        els.pName.value = ""; 
        els.editPid.value = ""; 
        els.scanMsg.classList.add('hidden'); 
        els.scanPreviewCont.classList.add('hidden'); 
        els.editPhotoButtons.classList.add('hidden');
        els.manualFields.classList.add('hidden'); 
        els.scanBtn.innerText = "SCAN & AUTO-FILL"; 
        els.scanBtn.classList.remove('hidden'); 
        els.scanDrop.classList.remove('hidden'); 
    } else { 
        els.modalTitle.innerText = "Edit Persona"; 
        const p = allPersonas.find(x => String(x.id) === String(pid)); 
        if (!p) return; 
        els.pName.value = p.name; 
        els.editPid.value = pid; 
        els.scanDrop.classList.add('hidden'); 
        els.scanBtn.classList.add('hidden'); 
        els.scanMsg.classList.add('hidden'); 
        els.scanPreviewCont.classList.remove('hidden'); 
        els.editPhotoButtons.classList.remove('hidden');
        els.scanMain.src = `/persona-image/${pid}?t=${Date.now()}`; 
        els.scanBlur.src = els.scanMain.src; 
        els.manualFields.classList.remove('hidden');
        const s = p.subject || {}; const bp = s.body_proportions || {}; const h = s.hair || {};
        els.pAge.value = s.age || ""; els.pEthnicity.value = s.ethnicity || ""; els.pBuild.value = bp.build || s.body_type || ""; els.pChest.value = bp.chest || ""; els.pShoulders.value = bp.shoulders || ""; els.pWaist.value = bp.waist_to_chest_ratio || ""; els.pFace.value = s.face_structure || ""; els.pSkin.value = s.skin || ""; els.pEyes.value = s.eyes || ""; els.pNose.value = s.nose || ""; els.pLips.value = s.lips || ""; els.pTattoos.value = s.tattoos || ""; els.pEyewear.value = s.eyewear || ""; els.pMakeup.value = s.makeup || ""; els.pHairColor.value = h.color || ""; els.pHairStyle.value = h.style || "";
    }
}

els.triggerRescan.onclick = async () => {
    const pid = els.editPid.value;
    if (!pid) return;
    els.scanMsg.classList.remove('hidden');
    els.scanMsg.innerText = "Re-scanning...";
    const fd = new FormData();
    fd.append('name', els.pName.value);
    fd.append('mode', 'edit');
    fd.append('pid', pid);
    fd.append('rescan', 'true');
    fd.append('model', els.visionModel.value);
    try {
        const res = await fetch('/personas/create', { method: 'POST', body: fd });
        if (res.ok) {
            const data = await res.json();
            await loadPersonas();
            openModal('edit', pid);
            showNotification('✨ Re-scanned', 'success');
        } else {
            const err = await res.json();
            showNotification(`❌ ${err.detail || 'Failed'}`, 'error');
        }
    } catch(e) {
        showNotification(`❌ Error: ${e.message}`, 'error');
    } finally {
        els.scanMsg.classList.add('hidden');
    }
};

els.triggerNewPhoto.onclick = () => {
    els.scanPreviewCont.classList.add('hidden');
    els.editPhotoButtons.classList.add('hidden');
    els.scanDrop.classList.remove('hidden');
    els.scanBtn.classList.remove('hidden');
    els.scanBtn.innerText = "UPLOAD & RE-SCAN";
};

els.scanBtn.onclick = async () => {
    if(!els.pName.value || !scanFile) return showNotification('⚠️ Name/Photo required!', 'warning');
    els.scanBtn.disabled = true; els.scanBtn.innerText = "SCANNING..."; els.scanMsg.classList.remove('hidden');
    const fd = new FormData(); fd.append('name', els.pName.value); fd.append('file', scanFile); fd.append('model', els.visionModel.value);
    if(els.editPid.value) { fd.append('mode', 'edit'); fd.append('pid', els.editPid.value); }
    try { const res = await fetch('/personas/create', { method: 'POST', body: fd }); if(res.ok) { const data = await res.json(); await loadPersonas(); els.editPid.value = data.id; openModal('edit', data.id); showNotification(`✨ Persona Scanned`, 'success'); } } 
    catch(e) { showNotification(`❌ Error: ${e.message}`, 'error'); } finally { els.scanBtn.disabled = false; els.scanMsg.classList.add('hidden'); }
};

els.updateManualBtn.onclick = async () => {
    const pid = els.editPid.value; if(!pid) return;
    const sub = { age: els.pAge.value, ethnicity: els.pEthnicity.value, body_proportions: { build: els.pBuild.value, chest: els.pChest.value, shoulders: els.pShoulders.value, waist_to_chest_ratio: els.pWaist.value }, body_type: els.pBuild.value, face_structure: els.pFace.value, skin: els.pSkin.value, eyes: els.pEyes.value, nose: els.pNose.value, lips: els.pLips.value, tattoos: els.pTattoos.value, eyewear: els.pEyewear.value, makeup: els.pMakeup.value, hair: { color: els.pHairColor.value, style: els.pHairStyle.value } };
    await fetch(`/personas/${pid}`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name: els.pName.value, subject: sub }) });
    await loadPersonas(); els.modal.classList.add('hidden'); showNotification(`✅ Updated`, 'success');
};

els.actionBtn.onclick = async () => {
    if (currentMode === 'json') {
        if (inputMode === 'image' && !store.file && !store.url) return showNotification('⚠️ No Image', 'warning');
        if (inputMode === 'text' && !els.textPromptInput.value.trim()) return showNotification('⚠️ No Text', 'warning');
        showLoading(inputMode === 'image' ? "ANALYZING" : "PARSING", "Processing...");
        const fd = new FormData();
        if (inputMode === 'image') { if (store.file) fd.append('file', store.file); if (store.url) fd.append('image_url', store.url); fd.append('text_prompt', ''); } else { fd.append('text_prompt', els.textPromptInput.value.trim()); fd.append('image_url', ''); }
        fd.append('model', els.visionModel.value); 
		fd.append('persona_id', els.persona.value); 
		fd.append('wardrobe_id', els.wardrobeIdHidden.value); 
		fd.append('time_override', els.time.value); 
		fd.append('ratio_override', els.ratio.value); 
		fd.append('style_override', els.styleVisual.value); 
		fd.append('quality_override', els.quality.value); 
		fd.append('hair_style_override', els.hairStyle.value); 
		fd.append('hair_color_override', els.hairColor.value); 
		fd.append('makeup_override', els.makeup.value); 
		fd.append('glasses_override', els.glasses.value); 
		fd.append('expr_override', els.expr.value);
        fd.append('reference_mode', els.refMode.checked);		
		fd.append('hair_source', els.hairSource.value);
        try { const res = await fetch('/analyze', { method: 'POST', body: fd }); const data = await res.json(); if (data.error) throw new Error(data.error); store.json = data; store.prompt = null; store.tags = null; jar.updateCode(JSON.stringify(data, null, 2)); updateHistoryUI(); if (currentMode !== 'json') switchTab('json'); showNotification('✨ Done', 'success'); } catch(e) { jar.updateCode(`// ERROR: ${e.message}`); showNotification('❌ Failed', 'error'); } finally { hideLoading(); }
    } else if (currentMode === 'prompt') {
        if (!store.json) return showNotification('⚠️ Generate JSON first', 'warning');
        showLoading("WRITING", "Generating...");
        
        // USE NARRATIVE STYLE (CORRECTED REFERENCE)
        const styleName = els.styleNarrative.value;
        const instruction = (styleName === 'default') ? "Write a natural, detailed description." : window.allStyles[styleName];
        
        try { const res = await fetch('/generate-prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ json: store.json, model: els.writerModel.value, style_instruction: instruction }) }); const data = await res.json(); let output = `POSITIVE PROMPT:\n${data.prompt}\n\n`; if (store.json.reference_image_instruction) output = `// REFERENCE:\n${store.json.reference_image_instruction}\n\n` + output; if (store.json.negative_prompt) output += `---\n\nNEGATIVE PROMPT:\n${store.json.negative_prompt.join(', ')}`; store.prompt = output; document.getElementById('editor').textContent = output; showNotification('✨ Generated', 'success'); } catch(e) { showNotification('❌ Failed', 'error'); } finally { hideLoading(); }
    } else if (currentMode === 'tags') {
        if (!store.file && !store.url) return showNotification('⚠️ Upload Image', 'warning');
        showLoading("TAGGING", "Generating...");
        const fd = new FormData(); if(store.file) fd.append('file', store.file); if(store.url) fd.append('image_url', store.url); fd.append('model', els.taggerModel.value); fd.append('persona_id', els.persona.value); fd.append('reference_mode', els.refMode.checked);
        try { const res = await fetch('/generate-tags', { method: 'POST', body: fd }); const data = await res.json(); store.tags = `POSITIVE TAGS:\n${data.positive_tags}\n\n---\n\nNEGATIVE:\n${data.negative_prompt}`; document.getElementById('editor').textContent = store.tags; showNotification('✨ Tags Generated', 'success'); } catch(e) { showNotification('❌ Failed', 'error'); } finally { hideLoading(); }
    }
};

els.refineBtn.onclick = async () => {
    const txt = els.refineInput.value.trim(); if(!txt || currentMode !== 'json' || !store.json) return;
    showLoading("REFINING", "...");
    try { const res = await fetch('/refine', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ current_json: store.json, instruction: txt, model: els.writerModel.value }) }); const data = await res.json(); if(data.status === 'success') { store.json = data.json; jar.updateCode(JSON.stringify(data.json, null, 2)); els.refineInput.value = ''; showNotification('✨ Refined', 'success'); } } catch(e) { showNotification('❌ Failed', 'error'); } finally { hideLoading(); }
};

// --- STYLE MANAGER LOGIC (FIXED) ---
let allStyles = {};
async function loadStyles() {
    try {
        const res = await fetch('/styles'); allStyles = await res.json(); window.allStyles = allStyles;
        let html = `<option value="default">Standard</option>`;
        for (const [name, instruction] of Object.entries(allStyles)) html += `<option value="${name}">${name}</option>`;
        
        // USE CORRECT VARIABLE: styleNarrative
        els.styleNarrative.innerHTML = html;
        bindSticky(els.styleNarrative, 'promptStyle', 'default');
    } catch(e) {}
}
els.openStyleBtn.onclick = () => { els.styleModal.classList.remove('hidden'); els.styleInputRaw.value = ""; els.styleAnalysisResult.classList.add('hidden'); };
els.closeStyleModal.onclick = () => els.styleModal.classList.add('hidden');
els.analyzeStyleBtn.onclick = async () => {
    const raw = els.styleInputRaw.value.trim(); 
    if(!raw) return showNotification("⚠️ Please paste a prompt first", "warning");
    
    const currentModel = els.writerModel.value;
    if (!currentModel || currentModel === "" || currentModel === "loading") {
        showNotification("⚠️ Ollama is not running! Please start it and refresh.", "error");
        return; 
    }
    
    els.analyzeStyleBtn.innerText = "⏳ ANALYZING..."; 
    els.analyzeStyleBtn.disabled = true;
    
    try { 
        const res = await fetch('/styles/analyze', { 
            method: 'POST', 
            headers: {'Content-Type': 'application/json'}, 
            body: JSON.stringify({ prompt: raw, model: currentModel }) 
        }); 
        const data = await res.json(); 

        if (data.status === "error" || !data.instruction) {
            throw new Error(data.message || "No instruction returned");
        }

        els.styleInstruction.value = data.instruction; 
        els.styleAnalysisResult.classList.remove('hidden'); 
    } catch(e) { 
        showNotification(`❌ ${e.message}`, "error"); 
    } finally { 
        els.analyzeStyleBtn.innerText = "ANALYZE STRUCTURE"; 
        els.analyzeStyleBtn.disabled = false; 
    }
};

els.saveStyleBtn.onclick = async () => {
    const name = els.styleName.value.trim(); const instr = els.styleInstruction.value.trim(); if(!name) return;
    await fetch('/styles/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name, instruction: instr }) });
    await loadStyles(); 
    els.styleNarrative.value = name; 
    els.styleModal.classList.add('hidden'); 
    showNotification("✨ Saved", "success");
};

els.deleteStyleBtn.onclick = async () => {
    const name = els.styleNarrative.value;
    if (name === 'default' || name === 'Standard') {
        showNotification("⚠️ Cannot delete default style", "warning");
        return;
    }
    if (!confirm(`Delete style "${name}"?`)) return;
    try {
        await fetch(`/styles/${encodeURIComponent(name)}`, { method: 'DELETE' });
        await loadStyles();
        showNotification("🗑️ Style deleted", "success");
    } catch(e) {
        showNotification("❌ Failed to delete", "error");
    }
};

// --- INIT ---
window.editPersona = (pid) => { els.managerModal.classList.add('hidden'); openModal('edit', pid); };
window.deletePersona = async (pid) => { if(confirm("Delete?")) { await fetch(`/personas/${pid}`, { method: 'DELETE' }); await loadPersonas(); }};
window.loadH = (i) => { store.json = window.histData[i].json; switchTab('json'); };
window.deleteWardrobe = async (id) => { if(!confirm("Delete?")) return; await fetch(`/wardrobe/${id}`, { method: 'DELETE' }); await loadWardrobe(); };

els.addBtn.onclick = () => openModal('create'); els.closeModal.onclick = () => els.modal.classList.add('hidden');
els.manageBtn.onclick = () => els.managerModal.classList.remove('hidden'); els.closeManager.onclick = () => els.managerModal.classList.add('hidden');
els.scanDrop.onclick = () => els.scanInput.click();
els.clearHistory.onclick = async () => { await fetch('/history', {method:'DELETE'}); updateHistoryUI(); };
document.getElementById('quit-btn').onclick = async () => { await fetch('/shutdown', {method:'POST'}); window.close(); };
els.settingsBtn.onclick = () => els.settingsModal.classList.remove('hidden'); els.closeSettings.onclick = () => els.settingsModal.classList.add('hidden'); els.closeImgModal.onclick = () => els.imgModal.classList.add('hidden');
els.settingsBtn.onclick = () => els.settingsModal.classList.remove('hidden');
els.keyGoogle.value = localStorage.getItem('google_key') || ""; els.keyFal.value = localStorage.getItem('fal_key') || ""; els.keyXai.value = localStorage.getItem('xai_key') || "";
els.saveKeysBtn.onclick = () => { localStorage.setItem('google_key', els.keyGoogle.value); localStorage.setItem('fal_key', els.keyFal.value); localStorage.setItem('xai_key', els.keyXai.value); els.settingsModal.classList.add('hidden'); showNotification("🔑 Keys Saved", "success"); };

els.genCloudBtn.onclick = async () => {
    let prompt = "", negative = "";
    if (currentMode === 'prompt' && store.prompt) { const p = store.prompt.split('NEGATIVE PROMPT:'); prompt = p[0].replace('POSITIVE PROMPT:', '').replace('// REFERENCE:', '').trim(); if(p[1]) negative = p[1].trim(); }
    else if (currentMode === 'tags' && store.tags) { const p = store.tags.split('NEGATIVE:'); prompt = p[0].replace('POSITIVE TAGS:', '').trim(); if(p[1]) negative = p[1].trim(); }
    else return showNotification("⚠️ Generate Prompt/Tags first", "warning");
    const provider = els.cloudModel.value, gKey = localStorage.getItem('google_key'), fKey = localStorage.getItem('fal_key'), xKey = localStorage.getItem('xai_key');
    let activeKey = (provider === 'nanobana') ? gKey : (provider === 'grok') ? xKey : fKey;
    els.imgModal.classList.remove('hidden'); els.cloudLoader.classList.remove('hidden'); els.cloudResult.classList.add('hidden'); els.cloudError.classList.add('hidden');
    try { const res = await fetch('/generate-image-cloud', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ prompt, negative_prompt: negative, width: provider === 'grok'?1024:832, height: provider === 'grok'?1024:1216, model_provider: provider, api_key: activeKey }) }); const data = await res.json(); if(data.status === 'success') { els.cloudResult.src = "data:image/png;base64," + data.image; els.cloudResult.classList.remove('hidden'); } else throw new Error(data.message); } catch(e) { els.cloudError.innerText = e.message; els.cloudError.classList.remove('hidden'); } finally { els.cloudLoader.classList.add('hidden'); }
};

els.persona.addEventListener('change', async () => {
    if (!store.json || Object.keys(store.json).length === 0) return;
    showLoading("SWAPPING", "Injecting new persona...");
    try { const res = await fetch('/inject-persona', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ json: store.json, persona_id: els.persona.value, reference_mode: els.refMode.checked, hair_source: els.hairSource.value }) }); const data = await res.json(); if(data.status === 'success') { store.json = data.json; if (currentMode === 'json') jar.updateCode(JSON.stringify(data.json, null, 2)); showNotification(`✨ Switched`, "success"); if (currentMode === 'prompt') els.actionBtn.click(); } } catch(e) { showNotification("❌ Failed", "error"); } finally { hideLoading(); }
});

(async () => {
    const res = await fetch('/models'); const data = await res.json();
    
    if (data.ollama_running === false) {
        showNotification("⚠️ Ollama is not running! Start it with 'ollama serve'", "error", 10000);
        const banner = document.createElement('div');
        banner.id = 'ollama-warning';
        banner.className = 'fixed top-0 left-0 right-0 bg-red-600 text-white text-center py-2 text-sm font-bold z-[100] flex items-center justify-center gap-2';
        banner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Ollama is not running. Start it with: <code class="bg-red-800 px-2 py-0.5 rounded ml-1">ollama serve</code> <button onclick="location.reload()" class="ml-4 bg-white text-red-600 px-3 py-1 rounded text-xs font-bold hover:bg-red-100">Retry</button>';
        document.body.prepend(banner);
    }
    
    const visionModels = data.vision_models || [];
    const textModels = data.text_models || data.all_model_names || [];
    const allModels = data.all_model_names || [...visionModels, ...textModels];
    
    if (visionModels.length === 0 && data.ollama_running !== false) {
        showNotification("⚠️ No vision models found. Install one with 'ollama pull qwen2.5-vl'", "warning", 8000);
    }
    
    const populateVision = (sel, def, key) => { 
        if (visionModels.length === 0) {
            sel.innerHTML = '<option value="">No vision models</option>';
            return;
        }
        sel.innerHTML = visionModels.map(m => `<option value="${m}">${m}</option>`).join(''); 
        bindSticky(sel, key, visionModels.find(m => m.includes(def)) || visionModels[0]); 
    };
    const populateText = (sel, def, key) => { 
        if (allModels.length === 0) {
            sel.innerHTML = '<option value="">No models</option>';
            return;
        }
        sel.innerHTML = allModels.map(m => `<option value="${m}">${m}</option>`).join(''); 
        bindSticky(sel, key, allModels.find(m => m.includes(def)) || textModels[0] || allModels[0]); 
    };
    populateVision(els.visionModel, "qwen", "mem_vision"); 
    populateText(els.writerModel, "llama", "mem_writer"); 
    populateVision(els.taggerModel, "qwen", "mem_tagger");
    
    await loadPersonas(); 
    await loadWardrobe(); 
    await loadStyles(); 
    updateHistoryUI();
    
    bindSticky(els.time, 'time'); bindSticky(els.ratio, 'ratio'); bindSticky(els.styleVisual, 'styleVisual'); bindSticky(els.quality, 'quality'); bindSticky(els.refMode, 'refMode', false); bindSticky(els.cloudModel, 'cloudModel', 'grok'); bindSticky(els.hairStyle, 'hairStyle'); bindSticky(els.hairColor, 'hairColor'); bindSticky(els.makeup, 'makeup'); bindSticky(els.glasses, 'glasses'); bindSticky(els.hairSource, 'hairSource', 'persona');
	
	els.hairSource.addEventListener('change', () => {
    const isManual = els.hairSource.value === 'manual';
    els.hairStyle.closest('div').parentElement.style.opacity = isManual ? '1' : '0.5';
    els.hairColor.closest('div').parentElement.style.opacity = isManual ? '1' : '0.5';
});

    try { 
        const vRes = await fetch('/version'); 
        const vData = await vRes.json(); 
        document.title = `AI Prompt Director v${vData.local}`; 
        document.getElementById('app-version').textContent = `v${vData.local}`;
        
        if (vData.update_available) {
            document.getElementById('app-version').innerHTML = `v${vData.local} <span class="text-yellow-400" title="Update available: v${vData.remote}">⬆</span>`;
        }
    } catch(e) {}
    
    try {
        const batchRes = await fetch('/batch-modal');
        const batchHtml = await batchRes.text();
        const container = document.getElementById('batch-modal-container');
        container.innerHTML = batchHtml;
        
        const scripts = container.querySelectorAll('script');
        scripts.forEach(oldScript => {
            const newScript = document.createElement('script');
            newScript.textContent = oldScript.textContent;
            oldScript.parentNode.replaceChild(newScript, oldScript);
        });
        console.log('✓ Batch modal loaded');
    } catch(e) { console.error('Failed to load batch modal:', e); }
    
    startSystemMonitor();
})();

function startSystemMonitor() {
    const monCpu = document.getElementById('mon-cpu');
    const monGpu = document.getElementById('mon-gpu');
    const monVram = document.getElementById('mon-vram');
    const monRam = document.getElementById('mon-ram');
    const monTemp = document.getElementById('mon-temp');
    
    function getColor(percent) {
        if (percent >= 90) return 'text-red-400';
        if (percent >= 75) return 'text-yellow-400';
        return 'text-green-400';
    }
    
    async function updateStats() {
        try {
            const res = await fetch('/system/stats');
            const data = await res.json();
            
            if (data.cpu) {
                monCpu.textContent = `${data.cpu.percent}%`;
                monCpu.className = getColor(data.cpu.percent);
            }
            
            if (data.ram) {
                monRam.textContent = `${data.ram.percent}%`;
                monRam.className = getColor(data.ram.percent);
            }
            
            if (data.gpu && data.gpu.available) {
                monGpu.textContent = `${data.gpu.utilization || 0}%`;
                monGpu.className = getColor(data.gpu.utilization || 0);
                
                const vramUsed = (data.gpu.vram_used / 1024).toFixed(1);
                const vramTotal = (data.gpu.vram_total / 1024).toFixed(1);
                monVram.textContent = `${vramUsed}/${vramTotal}G`;
                monVram.className = getColor(data.gpu.vram_percent || 0);
                
                if (data.gpu.temperature) {
                    monTemp.textContent = `${data.gpu.temperature}°C`;
                    monTemp.className = data.gpu.temperature >= 80 ? 'text-red-400' : data.gpu.temperature >= 65 ? 'text-yellow-400' : 'text-green-400';
                } else {
                    monTemp.textContent = '--°C';
                    monTemp.className = 'text-slate-500';
                }
            } else {
                monGpu.textContent = 'N/A';
                monGpu.className = 'text-slate-500';
                monVram.textContent = 'N/A';
                monVram.className = 'text-slate-500';
                monTemp.textContent = '--°C';
                monTemp.className = 'text-slate-500';
            }
        } catch(e) {
            console.error('System monitor error:', e);
        }
    }
    
    updateStats();
    setInterval(updateStats, 3000);
}

document.getElementById('batch-btn').onclick = () => {
    if (typeof window.openBatchModal === 'function') {
        window.openBatchModal();
    } else {
        showNotification('⚠️ Batch module not loaded', 'warning');
    }
};