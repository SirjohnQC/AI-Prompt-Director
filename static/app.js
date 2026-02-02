import { CodeJar } from 'https://medv.io/codejar/codejar.js';

// Initialize CodeJar editor
const jar = CodeJar(document.getElementById('editor'), (e) => {
    e.textContent = e.textContent;
    hljs.highlightElement(e);
}, { tab: '  ' });

// State
let currentMode = 'json'; 
let store = { json: null, prompt: null, tags: null, file: null, url: null };
let allPersonas = [];
let scanFile = null;
let wardrobeFile = null;
let inputMode = 'image'; 
let wardrobeItems = [];
let sysMonitorOpen = true;

// DOM Elements
const els = {
    // Input
    modeImage: document.getElementById('mode-image'),
    modeText: document.getElementById('mode-text'),
    inputImgCont: document.getElementById('input-container-image'),
    inputTextCont: document.getElementById('input-container-text'),
    textPromptInput: document.getElementById('text-prompt-input'),
    file: document.getElementById('file-input'),
    drop: document.getElementById('drop-zone'),
    preview: document.getElementById('preview-container'),
    empty: document.getElementById('empty-state'),
    clearImg: document.getElementById('clear-image'),
    thumbBlur: document.getElementById('thumb-blur'),
    thumbMain: document.getElementById('thumb-main'),
    urlInput: document.getElementById('url-input'),
    urlBtn: document.getElementById('url-load-btn'),
    
    // Temperature
    tempSlider: document.getElementById('ai-temperature'),
    tempValue: document.getElementById('temp-value'),
    
    // Tabs & Action
    tabJson: document.getElementById('tab-json'),
    tabPrompt: document.getElementById('tab-prompt'),
    actionBtn: document.getElementById('action-btn'),
    actionText: document.getElementById('action-btn-text'),
    
    // Editor
    editor: document.getElementById('editor'),
    editorPlaceholder: document.getElementById('editor-placeholder'),
    loader: document.getElementById('loader'),
    loaderText: document.getElementById('loader-text'),
    loaderSub: document.getElementById('loader-sub'),
    
    // Bottom bar
    refineInput: document.getElementById('refine-input'),
    refineBtn: document.getElementById('refine-btn'),
    copyBtn: document.getElementById('copy-btn'),
    copyBtnContainer: document.getElementById('copy-btn-container'),
    
    // Dropdowns
    promptType: document.getElementById('prompt-type-select'),
    persona: document.getElementById('persona-select'),
    styleNarrative: document.getElementById('style-select'),
    cloudModel: document.getElementById('cloud-model'),
    
    // Negative prompt toggle
    includeNegative: document.getElementById('include-negative'),
    
    // Image generation
    generateImageBtn: document.getElementById('generate-image-btn'),
    imgGenBackend: document.getElementById('img-gen-backend'),
    imgGenUrl: document.getElementById('img-gen-url'),
    imgGenWidth: document.getElementById('img-gen-width'),
    imgGenHeight: document.getElementById('img-gen-height'),
    imgGenSteps: document.getElementById('img-gen-steps'),
    
    // Models (in settings)
    visionModel: document.getElementById('model-vision'),
    writerModel: document.getElementById('model-writer'),
    taggerModel: document.getElementById('model-tagger'),
    
    // Scene Controls
    time: document.getElementById('time-select'),
    ratio: document.getElementById('ratio-select'),
    styleVisual: document.getElementById('style-manual-select'),
    quality: document.getElementById('quality-select'),
    
    // Style Overrides
    hairStyle: document.getElementById('hair-style-select'),
    hairColor: document.getElementById('hair-color-select'),
    makeup: document.getElementById('makeup-select'),
    glasses: document.getElementById('glasses-select'),
    expr: document.getElementById('expr-select'),
    hairSource: document.getElementById('hair-source-select'),
    refMode: document.getElementById('reference-mode'),
    
    // Wardrobe
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
    
    // Buttons
    addBtn: document.getElementById('add-persona-btn'),
    manageBtn: document.getElementById('manage-personas-btn'),
    settingsBtn: document.getElementById('settings-btn'),
    batchBtn: document.getElementById('batch-btn'),
    flushBtn: document.getElementById('flush-btn'),
    openStyleBtn: document.getElementById('open-style-btn'),
    
    // Modals
    modal: document.getElementById('persona-modal'),
    closeModal: document.getElementById('close-modal'),
    managerModal: document.getElementById('manager-modal'),
    closeManager: document.getElementById('close-manager'),
    managerList: document.getElementById('manager-list'),
    settingsModal: document.getElementById('settings-modal'),
    closeSettings: document.getElementById('close-settings'),
    styleModal: document.getElementById('style-modal'),
    closeStyleModal: document.getElementById('close-style-modal'),
    imgModal: document.getElementById('img-result-modal'),
    closeImgModal: document.getElementById('close-img-modal'),
    historyModal: document.getElementById('history-modal'),
    closeHistoryModal: document.getElementById('close-history-modal'),
    
    // History
    historyList: document.getElementById('history-list'),
    historyListFull: document.getElementById('history-list-full'),
    historySearch: document.getElementById('history-search'),
    historySort: document.getElementById('history-sort'),
    openHistoryBtn: document.getElementById('open-history-modal-btn'),
    
    // Persona form
    scanBtn: document.getElementById('scan-btn'),
    pName: document.getElementById('p-name'),
    scanInput: document.getElementById('scan-input'),
    scanDrop: document.getElementById('scan-drop'),
    manualFields: document.getElementById('manual-fields'),
    editPid: document.getElementById('edit-pid'),
    modalTitle: document.getElementById('modal-title-text'),
    updateManualBtn: document.getElementById('update-manual-btn'),
    scanMsg: document.getElementById('scan-msg'),
    scanPreviewCont: document.getElementById('scan-preview-container'),
    scanBlur: document.getElementById('scan-blur'),
    scanMain: document.getElementById('scan-main'),
    triggerRescan: document.getElementById('trigger-rescan'),
    triggerNewPhoto: document.getElementById('trigger-new-photo'),
    editPhotoButtons: document.getElementById('edit-photo-buttons'),
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
    
    // NSFW body fields
    nsfwBodyFields: document.getElementById('nsfw-body-fields'),
    pHips: document.getElementById('p-hips'),
    pButt: document.getElementById('p-butt'),
    pThighs: document.getElementById('p-thighs'),
    pLegs: document.getElementById('p-legs'),
    pNipples: document.getElementById('p-nipples'),
    pPubic: document.getElementById('p-pubic'),
    
    // Settings
    saveKeysBtn: document.getElementById('save-keys-btn'),
    keyGoogle: document.getElementById('key-google'),
    keyFal: document.getElementById('key-fal'),
    keyXai: document.getElementById('key-xai'),
    nsfwMode: document.getElementById('nsfw-mode'),
    quitBtn: document.getElementById('quit-btn'),
    
    // Style learner
    styleInputRaw: document.getElementById('style-input-raw'),
    analyzeStyleBtn: document.getElementById('analyze-style-btn'),
    styleAnalysisResult: document.getElementById('style-analysis-result'),
    styleInstruction: document.getElementById('style-instruction'),
    styleName: document.getElementById('style-name'),
    saveStyleBtn: document.getElementById('save-style-btn'),
    
    // Cloud generation
    cloudLoader: document.getElementById('cloud-loader'),
    cloudResult: document.getElementById('cloud-result-img'),
    cloudError: document.getElementById('cloud-error'),
    
    // System monitor
    sysMonitorToggle: document.getElementById('sys-monitor-toggle'),
    sysMonitorPanel: document.getElementById('sys-monitor-panel'),
    sysToggleIcon: document.getElementById('sys-toggle-icon'),
};

// ===================== UTILITIES =====================

function showNotification(message, type = 'info', duration = 3000) {
    const n = document.createElement('div');
    const colors = {
        success: 'border-green-500/30 text-green-400',
        error: 'border-red-500/30 text-red-400',
        warning: 'border-yellow-500/30 text-yellow-400',
        info: 'border-cyan-500/30 text-cyan-400'
    };
    const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
    n.className = `fixed top-4 right-4 z-[9999] px-5 py-3 rounded-lg border shadow-lg transform transition-all duration-300 translate-x-[400px] bg-[#161b22] ${colors[type]}`;
    n.innerHTML = `<div class="flex items-center gap-2"><span>${icons[type]}</span><span class="text-sm">${message}</span></div>`;
    document.body.appendChild(n);
    setTimeout(() => n.style.transform = 'translateX(0)', 10);
    setTimeout(() => { n.style.transform = 'translateX(400px)'; setTimeout(() => n.remove(), 300); }, duration);
}

function showLoading(title, sub) {
    if (els.loaderText) els.loaderText.innerText = title;
    if (els.loaderSub) els.loaderSub.innerText = sub;
    if (els.loader) els.loader.classList.remove('hidden');
    if (els.actionBtn) els.actionBtn.disabled = true;
    
    // Create progress bar if it doesn't exist
    let progressContainer = document.getElementById('analysis-progress-container');
    if (!progressContainer && els.loaderSub) {
        progressContainer = document.createElement('div');
        progressContainer.id = 'analysis-progress-container';
        progressContainer.className = 'w-64 mt-4';
        progressContainer.innerHTML = `
            <div class="bg-slate-800 rounded-full h-2 overflow-hidden">
                <div id="analysis-progress-bar" class="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-300 ease-out" style="width: 0%"></div>
            </div>
            <div id="analysis-percent" class="text-xs text-slate-500 mt-1 text-center">0%</div>
        `;
        els.loaderSub.after(progressContainer);
    }
    
    // Reset progress
    const pBar = document.getElementById('analysis-progress-bar');
    const pText = document.getElementById('analysis-percent');
    if (pBar) pBar.style.width = '0%';
    if (pText) pText.textContent = '0%';
}

function hideLoading() {
    if (els.loader) els.loader.classList.add('hidden');
    if (els.actionBtn) els.actionBtn.disabled = false;
    
    // Reset progress bar
    const pBar = document.getElementById('analysis-progress-bar');
    const pText = document.getElementById('analysis-percent');
    if (pBar) pBar.style.width = '0%';
    if (pText) pText.textContent = '';
}

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

function updateCopyButtonVisibility() {
    const content = els.editor?.innerText?.trim() || '';
    if (content && !content.startsWith('//') && content.length > 10) {
        els.copyBtnContainer?.classList.remove('hidden');
    } else {
        els.copyBtnContainer?.classList.add('hidden');
    }
}

function updatePlaceholderVisibility() {
    const content = els.editor?.innerText?.trim() || '';
    if (content && !content.startsWith('//') && content.length > 10) {
        els.editorPlaceholder?.classList.add('hidden');
    } else {
        els.editorPlaceholder?.classList.remove('hidden');
    }
}

// ===================== TEMPERATURE SLIDER =====================

if (els.tempSlider) {
    els.tempSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(1);
        if (els.tempValue) els.tempValue.textContent = val;
        localStorage.setItem('pref_temperature', val);
    });
    const savedTemp = localStorage.getItem('pref_temperature');
    if (savedTemp) {
        els.tempSlider.value = savedTemp;
        if (els.tempValue) els.tempValue.textContent = savedTemp;
    }
}

// ===================== SYSTEM MONITOR TOGGLE =====================

if (els.sysMonitorToggle) {
    els.sysMonitorToggle.addEventListener('click', () => {
        sysMonitorOpen = !sysMonitorOpen;
        if (sysMonitorOpen) {
            els.sysMonitorPanel?.classList.remove('collapsed');
            if (els.sysToggleIcon) els.sysToggleIcon.className = 'fas fa-chevron-right text-[10px]';
        } else {
            els.sysMonitorPanel?.classList.add('collapsed');
            if (els.sysToggleIcon) els.sysToggleIcon.className = 'fas fa-chevron-left text-[10px]';
        }
    });
}

// ===================== INPUT MODE TOGGLE =====================

function setInputMode(mode) {
    inputMode = mode;
    if (mode === 'image') {
        els.modeImage.className = "flex-1 py-2 text-[11px] font-medium rounded bg-cyan-600 text-white transition-all flex items-center justify-center gap-1.5";
        els.modeText.className = "flex-1 py-2 text-[11px] font-medium rounded text-slate-400 hover:text-white transition-all flex items-center justify-center gap-1.5";
        els.inputImgCont?.classList.remove('hidden');
        els.inputTextCont?.classList.add('hidden');
    } else {
        els.modeText.className = "flex-1 py-2 text-[11px] font-medium rounded bg-cyan-600 text-white transition-all flex items-center justify-center gap-1.5";
        els.modeImage.className = "flex-1 py-2 text-[11px] font-medium rounded text-slate-400 hover:text-white transition-all flex items-center justify-center gap-1.5";
        els.inputTextCont?.classList.remove('hidden');
        els.inputImgCont?.classList.add('hidden');
    }
    
    // Update editor hint based on new input mode (only if no content)
    const content = els.editor?.innerText?.trim() || '';
    if (!content || content.startsWith('//')) {
        const actionWord = currentMode === 'json' ? 'ANALYZE' : 'GENERATE PROMPT';
        const hint = mode === 'text' ? `// Enter text and click ${actionWord}` : `// Upload an image and click ${actionWord}`;
        els.editor.textContent = hint;
    }
}

if (els.modeImage) els.modeImage.onclick = () => setInputMode('image');
if (els.modeText) els.modeText.onclick = () => setInputMode('text');

// ===================== IMAGE HANDLING =====================

function handleFile(f) {
    if (!f) return;
    store.file = f;
    store.url = null;
    if (els.urlInput) els.urlInput.value = "";
    const r = new FileReader();
    r.onload = (ev) => {
        if (els.thumbBlur) els.thumbBlur.src = ev.target.result;
        if (els.thumbMain) els.thumbMain.src = ev.target.result;
        els.preview?.classList.remove('hidden');
        els.empty?.classList.add('hidden');
    };
    r.readAsDataURL(f);
}

if (els.file) els.file.onchange = (e) => handleFile(e.target.files[0]);
if (els.drop) {
    els.drop.onclick = () => els.file?.click();
    els.drop.addEventListener('dragover', (e) => { e.preventDefault(); els.drop.classList.add('drop-zone-active'); });
    els.drop.addEventListener('dragleave', () => els.drop.classList.remove('drop-zone-active'));
    els.drop.addEventListener('drop', (e) => { e.preventDefault(); els.drop.classList.remove('drop-zone-active'); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
}

if (els.clearImg) {
    els.clearImg.onclick = (e) => {
        e.stopPropagation();
        store.file = null;
        store.url = null;
        if (els.urlInput) els.urlInput.value = "";
        els.preview?.classList.add('hidden');
        els.empty?.classList.remove('hidden');
    };
}

if (els.urlBtn) {
    els.urlBtn.onclick = () => {
        const url = els.urlInput?.value.trim();
        if (!url) return;
        store.url = url;
        store.file = null;
        if (els.thumbBlur) els.thumbBlur.src = url;
        if (els.thumbMain) els.thumbMain.src = url;
        els.preview?.classList.remove('hidden');
        els.empty?.classList.add('hidden');
    };
}

if (els.urlInput) {
    els.urlInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') els.urlBtn?.click(); });
}

// Paste image support
if (els.textPromptInput) {
    els.textPromptInput.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) {
                    setInputMode('image');
                    handleFile(file);
                    showNotification('📋 Image pasted!', 'success');
                }
                return;
            }
        }
    });
}

// ===================== TAB SWITCHING =====================

function switchTab(mode) {
    currentMode = mode;
    
    // Reset tab styles
    if (els.tabJson) els.tabJson.className = "px-4 py-1.5 text-xs font-medium rounded transition-all text-slate-400 hover:text-white";
    if (els.tabPrompt) els.tabPrompt.className = "px-4 py-1.5 text-xs font-medium rounded transition-all text-slate-400 hover:text-white";
    
    if (mode === 'json') {
        if (els.tabJson) els.tabJson.className = "px-4 py-1.5 text-xs font-medium rounded transition-all text-cyan-400 bg-cyan-500/10";
        if (els.actionText) els.actionText.innerText = "ANALYZE";
        if (els.actionBtn) {
            els.actionBtn.className = "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white px-8 py-2 rounded-lg font-semibold text-sm shadow-lg shadow-cyan-900/30 transition-all active:scale-95 flex items-center gap-2";
        }
        if (store.json) {
            jar.updateCode(JSON.stringify(store.json, null, 2));
        } else {
            const hint = inputMode === 'text' ? "// Enter text and click ANALYZE" : "// Upload an image and click ANALYZE";
            els.editor.textContent = hint;
        }
    } else if (mode === 'prompt') {
        if (els.tabPrompt) els.tabPrompt.className = "px-4 py-1.5 text-xs font-medium rounded transition-all text-green-400 bg-green-500/10";
        if (els.actionText) els.actionText.innerText = "GENERATE PROMPT";
        if (els.actionBtn) {
            els.actionBtn.className = "bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white px-8 py-2 rounded-lg font-semibold text-sm shadow-lg shadow-green-900/30 transition-all active:scale-95 flex items-center gap-2";
        }
        if (store.prompt) {
            els.editor.textContent = store.prompt;
        } else {
            const hint = inputMode === 'text' ? "// Enter text and click GENERATE PROMPT" : "// Upload an image and click GENERATE PROMPT";
            els.editor.textContent = hint;
        }
    }
    
    updateCopyButtonVisibility();
    updatePlaceholderVisibility();
}

if (els.tabJson) els.tabJson.onclick = () => switchTab('json');
if (els.tabPrompt) els.tabPrompt.onclick = () => switchTab('prompt');

// ===================== MAIN GENERATE ACTION =====================

if (els.actionBtn) {
    els.actionBtn.onclick = async () => {
        console.log('[ACTION BTN] Clicked!', { currentMode, inputMode });
        console.log('[ACTION BTN] Vision model:', els.visionModel?.value, 'Options:', els.visionModel?.options?.length);
        console.log('[ACTION BTN] Store:', { hasFile: !!store.file, hasUrl: !!store.url });
        
        const temperature = parseFloat(els.tempSlider?.value || 0.7);
        
        // Check if vision model is available
        if (!els.visionModel?.value || els.visionModel.options.length === 0) {
            console.log('[ACTION BTN] No vision model - showing warning');
            showOllamaWarning("Ollama is not running! Start it with:");
            return;
        }
        
        // Common validation for both modes
        if (inputMode === 'image' && !store.file && !store.url) {
            return showNotification('⚠️ Upload an image first', 'warning');
        }
        if (inputMode === 'text' && !els.textPromptInput?.value.trim()) {
            return showNotification('⚠️ Enter some text', 'warning');
        }
        
        // =================== JSON TAB: ANALYZE ===================
        if (currentMode === 'json') {
            console.log('[ANALYZE] Starting...', { inputMode, file: store.file?.name, url: store.url });
            const loadingMsg = inputMode === 'text' ? "Processing text..." : "Processing image...";
            showLoading("ANALYZING", loadingMsg);
            
            const fd = new FormData();
            if (inputMode === 'image') {
                if (store.file) {
                    fd.append('file', store.file);
                    console.log('[ANALYZE] File:', store.file.name, store.file.size, 'bytes');
                }
                if (store.url) {
                    fd.append('image_url', store.url);
                    console.log('[ANALYZE] URL:', store.url);
                }
                fd.append('text_prompt', '');
            } else {
                fd.append('text_prompt', els.textPromptInput.value.trim());
                fd.append('image_url', '');
            }
            
            const model = els.visionModel?.value || 'qwen3-vl';
            console.log('[ANALYZE] Model:', model);
            fd.append('model', model);
            fd.append('persona_id', els.persona?.value || 'none');
            fd.append('wardrobe_id', els.wardrobeIdHidden?.value || 'none');
            fd.append('time_override', els.time?.value || 'auto');
            fd.append('ratio_override', els.ratio?.value || 'auto');
            fd.append('style_override', els.styleVisual?.value || 'auto');
            fd.append('quality_override', els.quality?.value || 'auto');
            fd.append('hair_style_override', els.hairStyle?.value || 'auto');
            fd.append('hair_color_override', els.hairColor?.value || 'auto');
            fd.append('makeup_override', els.makeup?.value || 'auto');
            fd.append('glasses_override', els.glasses?.value || 'auto');
            fd.append('expr_override', els.expr?.value || 'auto');
            fd.append('reference_mode', els.refMode?.checked || false);
            fd.append('hair_source', els.hairSource?.value || 'persona');
            fd.append('nsfw_mode', localStorage.getItem('nsfw_mode') === 'true' ? 'true' : 'false');
            
            try {
                console.log('[ANALYZE] Sending request to /analyze-stream...');
                
                // Use streaming endpoint for progress updates
                const res = await fetch('/analyze-stream', { method: 'POST', body: fd });
                
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}`);
                }
                
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let data = null;
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    
                    // Process complete events
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        
                        try {
                            const eventData = JSON.parse(line.substring(6));
                            
                            switch (eventData.type) {
                                case 'progress':
                                    if (els.loaderText) els.loaderText.textContent = 'ANALYZING';
                                    if (els.loaderSub) els.loaderSub.textContent = eventData.phase || 'Processing...';
                                    // Update progress bar if exists
                                    const pBar = document.getElementById('analysis-progress-bar');
                                    const pText = document.getElementById('analysis-percent');
                                    if (pBar) pBar.style.width = `${eventData.percent || 0}%`;
                                    if (pText) pText.textContent = `${eventData.percent || 0}%`;
                                    console.log(`[ANALYZE] ${eventData.percent}% - ${eventData.phase}`);
                                    break;
                                    
                                case 'complete':
                                    data = eventData.data;
                                    break;
                                    
                                case 'error':
                                    throw new Error(eventData.message || 'Analysis failed');
                            }
                        } catch (parseErr) {
                            if (parseErr.message !== 'Analysis failed') {
                                console.warn('[ANALYZE] Event parse error:', parseErr);
                            } else {
                                throw parseErr;
                            }
                        }
                    }
                }
                
                if (!data) {
                    throw new Error('No analysis result received');
                }
                
                console.log('[ANALYZE] Response data:', data);
                
                if (data.error) throw new Error(data.error);
                
                // Remove negative prompt if toggle is off
                if (!els.includeNegative?.checked && data.negative_prompt) {
                    delete data.negative_prompt;
                }
                
                store.json = data;
                store.prompt = null; // Clear old prompt
                jar.updateCode(JSON.stringify(data, null, 2));
                
                // Enable generate image button
                if (els.generateImageBtn) els.generateImageBtn.disabled = false;
                
                updateHistoryUI();
                updateCopyButtonVisibility();
                updatePlaceholderVisibility();
                showNotification('✨ Analysis complete!', 'success');
                
            } catch (e) {
                console.error(e);
                if (e.message?.includes('Ollama') || e.message?.includes('model') || e.message?.includes('connection')) {
                    showOllamaWarning("Ollama connection failed. Is it running?");
                } else {
                    showNotification('❌ ' + e.message, 'error');
                }
            } finally {
                hideLoading();
            }
        }
        
        // =================== PROMPT TAB: GENERATE PROMPT DIRECTLY ===================
        else if (currentMode === 'prompt') {
            console.log('[PROMPT TAB] Starting prompt generation...');
            console.log('[PROMPT TAB] Input mode:', inputMode);
            console.log('[PROMPT TAB] Store:', { hasFile: !!store.file, fileName: store.file?.name, hasUrl: !!store.url, url: store.url });
            
            const loadingMsg = inputMode === 'text' ? "Enhancing text prompt..." : "Creating prompt from image...";
            showLoading("GENERATING", loadingMsg);
            
            const promptStyle = els.promptType?.value || 'narrative';
            const styleName = els.styleNarrative?.value || 'default';
            const instruction = (styleName === 'default') 
                ? null 
                : (window.allStyles?.[styleName] || null);
            
            console.log('[PROMPT TAB] Prompt style:', promptStyle);
            console.log('[PROMPT TAB] Narrative style:', styleName);
            
            const fd = new FormData();
            if (inputMode === 'image') {
                if (store.file) {
                    fd.append('file', store.file);
                    console.log('[PROMPT TAB] Appending file:', store.file.name, store.file.size, 'bytes');
                }
                if (store.url) {
                    fd.append('image_url', store.url);
                    console.log('[PROMPT TAB] Appending URL:', store.url);
                }
            } else {
                fd.append('text_input', els.textPromptInput.value.trim());
            }
            
            fd.append('model', els.visionModel?.value || 'qwen2.5-vl');
            fd.append('persona_id', els.persona?.value || 'none');
            fd.append('wardrobe_id', els.wardrobeIdHidden?.value || 'none');
            fd.append('prompt_style', promptStyle);
            if (instruction) fd.append('style_instruction', instruction);
            fd.append('time_override', els.time?.value || 'auto');
            fd.append('aesthetic_override', els.styleVisual?.value || 'auto');
            fd.append('expression_override', els.expr?.value || 'auto');
            fd.append('makeup_override', els.makeup?.value || 'auto');
            fd.append('reference_mode', els.refMode?.checked || false);
            fd.append('nsfw_mode', localStorage.getItem('nsfw_mode') === 'true' ? 'true' : 'false');
            
            try {
                // Call direct prompt generation endpoint
                console.log('[PROMPT] Sending request to /generate-prompt-direct...');
                console.log('[PROMPT] Form data:', {
                    promptStyle,
                    styleName,
                    model: els.visionModel?.value,
                    persona: els.persona?.value
                });
                
                const res = await fetch('/generate-prompt-direct', { method: 'POST', body: fd });
                console.log('[PROMPT] Response status:', res.status);
                const data = await res.json();
                console.log('[PROMPT] Response data:', data);
                
                if (data.error) throw new Error(data.error);
                
                // Build output
                let output = data.prompt || '';
                
                // Add negative prompt if toggle is ON and negative prompt exists
                if (els.includeNegative?.checked) {
                    const negPrompt = data.negative_prompt || data.negative;
                    if (negPrompt) {
                        output += `\n\n---\n\nNEGATIVE PROMPT:\n${negPrompt}`;
                    }
                }
                
                // Add reference instruction if present
                if (data.reference_instruction) {
                    output = `// REFERENCE: ${data.reference_instruction}\n\n${output}`;
                }
                
                store.prompt = output;
                store.negativePrompt = data.negative_prompt || data.negative || '';
                els.editor.textContent = output;
                
                // Enable generate image button
                if (els.generateImageBtn) els.generateImageBtn.disabled = false;
                
                updateCopyButtonVisibility();
                updatePlaceholderVisibility();
                showNotification('✨ Prompt generated!', 'success');
                
            } catch (e) {
                console.error(e);
                if (e.message?.includes('Ollama') || e.message?.includes('model')) {
                    showOllamaWarning("Ollama connection failed. Is it running?");
                } else {
                    showNotification('❌ ' + e.message, 'error');
                }
            } finally {
                hideLoading();
            }
        }
    };
}

// ===================== REFINE =====================

if (els.refineBtn) {
    els.refineBtn.onclick = async () => {
        const txt = els.refineInput?.value.trim();
        if (!txt || !store.json) return;
        
        showLoading("REFINING", "...");
        try {
            const res = await fetch('/refine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current_json: store.json,
                    instruction: txt,
                    model: els.writerModel?.value || 'llama3.2'
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                store.json = data.json;
                jar.updateCode(JSON.stringify(data.json, null, 2));
                if (els.refineInput) els.refineInput.value = '';
                updateCopyButtonVisibility();
                updatePlaceholderVisibility();
                showNotification('✨ Refined!', 'success');
            }
        } catch (e) {
            showNotification('❌ Failed', 'error');
        } finally {
            hideLoading();
        }
    };
}

if (els.refineInput) {
    els.refineInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') els.refineBtn?.click(); });
}

// ===================== COPY =====================

if (els.copyBtn) {
    els.copyBtn.onclick = () => {
        const text = els.editor?.innerText || '';
        if (!text || text.startsWith('//')) return showNotification('⚠️ Nothing to copy', 'warning');
        
        navigator.clipboard.writeText(text);
        
        // Visual feedback on button
        const originalHTML = els.copyBtn.innerHTML;
        els.copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        els.copyBtn.classList.remove('bg-green-600', 'hover:bg-green-500');
        els.copyBtn.classList.add('bg-emerald-500');
        
        setTimeout(() => {
            els.copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            els.copyBtn.classList.add('bg-green-600', 'hover:bg-green-500');
            els.copyBtn.classList.remove('bg-emerald-500');
        }, 1500);
        
        showNotification('📋 Copied!', 'success');
    };
}

// ===================== WARDROBE =====================

async function loadWardrobe() {
    try {
        const res = await fetch('/wardrobe');
        const rawData = await res.json();
        wardrobeItems = Array.isArray(rawData) ? rawData : Object.values(rawData || {});
        renderWardrobeGrid(wardrobeItems);
        
        const savedId = localStorage.getItem('pref_wardrobe_id');
        if (savedId && savedId !== 'none') {
            const item = wardrobeItems.find(i => i.id === savedId);
            if (item) selectOutfit(item.id, item.name, item.image, false);
        }
    } catch (e) {
        console.error("Wardrobe error", e);
        renderWardrobeGrid([]);
    }
}

function renderWardrobeGrid(items) {
    if (!els.wardrobeGrid) return;
    els.wardrobeGrid.innerHTML = '';
    
    // None card
    const noneCard = document.createElement('div');
    noneCard.className = `group relative bg-[#161b22] rounded-xl border border-white/10 overflow-hidden cursor-pointer hover:border-slate-500 transition-all flex flex-col items-center justify-center h-48 ${els.wardrobeIdHidden?.value === 'none' ? 'ring-2 ring-pink-500' : ''}`;
    noneCard.innerHTML = `<i class="fas fa-ban text-2xl text-slate-600 mb-2"></i><span class="text-xs text-slate-500">No Outfit</span>`;
    noneCard.onclick = () => selectOutfit('none', 'None', null);
    els.wardrobeGrid.appendChild(noneCard);
    
    if (items.length === 0) {
        els.wardrobeEmpty?.classList.remove('hidden');
    } else {
        els.wardrobeEmpty?.classList.add('hidden');
    }
    
    items.forEach(item => {
        const isSelected = els.wardrobeIdHidden?.value === item.id;
        const card = document.createElement('div');
        card.className = `group relative bg-[#161b22] rounded-xl border border-white/10 overflow-hidden cursor-pointer hover:border-pink-500/50 transition-all flex flex-col h-48 ${isSelected ? 'ring-2 ring-pink-500' : ''}`;
        card.innerHTML = `
            <div class="h-32 bg-slate-800 relative overflow-hidden">
                <img src="${item.image}" class="w-full h-full object-cover" loading="lazy">
            </div>
            <div class="flex-1 p-2 flex flex-col justify-between">
                <h4 class="text-xs font-medium text-slate-200 truncate">${item.name}</h4>
                <button class="delete-btn text-slate-600 hover:text-red-500 text-xs self-end"><i class="fas fa-trash"></i></button>
            </div>
        `;
        card.addEventListener('click', (e) => {
            if (e.target.closest('.delete-btn')) return;
            selectOutfit(item.id, item.name, item.image);
        });
        card.querySelector('.delete-btn').onclick = async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete "${item.name}"?`)) return;
            await fetch(`/wardrobe/${item.id}`, { method: 'DELETE' });
            if (isSelected) selectOutfit('none', 'None', null, false);
            await loadWardrobe();
        };
        els.wardrobeGrid.appendChild(card);
    });
    
    if (els.wardrobeCount) els.wardrobeCount.innerText = `${items.length} items`;
}

function selectOutfit(id, name, image, close = true) {
    if (els.wardrobeIdHidden) els.wardrobeIdHidden.value = id;
    localStorage.setItem('pref_wardrobe_id', id);
    if (els.activeWardrobeName) els.activeWardrobeName.innerText = name;
    if (els.activeWardrobeImg) {
        if (image) {
            els.activeWardrobeImg.innerHTML = '';
            els.activeWardrobeImg.style.backgroundImage = `url('${image}')`;
        } else {
            els.activeWardrobeImg.style.backgroundImage = 'none';
            els.activeWardrobeImg.innerHTML = '<div class="absolute inset-0 flex items-center justify-center text-slate-600 text-[10px]"><i class="fas fa-ban"></i></div>';
        }
    }
    renderWardrobeGrid(wardrobeItems);
    if (close) {
        els.wardrobeBrowser?.classList.add('hidden');
        els.wardrobeBrowser?.classList.remove('flex');
    }
}

// Wardrobe event handlers
if (els.wardrobeTrigger) {
    els.wardrobeTrigger.onclick = () => {
        els.wardrobeBrowser?.classList.remove('hidden');
        els.wardrobeBrowser?.classList.add('flex');
        loadWardrobe();
    };
}

if (els.closeBrowser) {
    els.closeBrowser.onclick = () => {
        els.wardrobeBrowser?.classList.add('hidden');
        els.wardrobeBrowser?.classList.remove('flex');
    };
}

if (els.wardrobeBrowser) {
    els.wardrobeBrowser.onclick = (e) => {
        if (e.target === els.wardrobeBrowser) {
            els.wardrobeBrowser.classList.add('hidden');
            els.wardrobeBrowser.classList.remove('flex');
        }
    };
}

if (els.browserAddBtn) {
    els.browserAddBtn.onclick = () => {
        els.wardrobeBrowser?.classList.add('hidden');
        els.wardrobeBrowser?.classList.remove('flex');
        els.wardrobeModal?.classList.remove('hidden');
    };
}

if (els.wardrobeSearch) {
    els.wardrobeSearch.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();
        const filtered = wardrobeItems.filter(i => i.name.toLowerCase().includes(term));
        renderWardrobeGrid(filtered);
    });
}

if (els.closeWardrobe) els.closeWardrobe.onclick = () => els.wardrobeModal?.classList.add('hidden');

if (els.wUpload) els.wUpload.onclick = () => els.wFile?.click();

if (els.wFile) {
    els.wFile.onchange = (e) => {
        if (e.target.files[0]) {
            wardrobeFile = e.target.files[0];
            if (els.wPreview) els.wPreview.style.backgroundImage = `url(${URL.createObjectURL(wardrobeFile)})`;
            if (els.wUrl) els.wUrl.value = "";
        }
    };
}

if (els.wUrl) {
    els.wUrl.addEventListener('input', (e) => {
        if (e.target.value && els.wPreview) {
            els.wPreview.style.backgroundImage = `url(${e.target.value})`;
            wardrobeFile = null;
        }
    });
}

if (els.saveWardrobeBtn) {
    els.saveWardrobeBtn.onclick = async () => {
        const name = els.wName?.value.trim();
        if (!name) return showNotification("⚠️ Name required", "warning");
        
        els.saveWardrobeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ANALYZING...';
        els.saveWardrobeBtn.disabled = true;
        
        const fd = new FormData();
        fd.append('name', name);
        if (wardrobeFile) fd.append('file', wardrobeFile);
        if (els.wUrl?.value) fd.append('image_url', els.wUrl.value);
        fd.append('model', els.visionModel?.value || 'qwen3-vl');
        
        try {
            const res = await fetch('/wardrobe/create', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.status === 'success') {
                showNotification("✨ Outfit Added!", "success");
                if (els.wName) els.wName.value = "";
                if (els.wUrl) els.wUrl.value = "";
                if (els.wPreview) els.wPreview.style.backgroundImage = "";
                wardrobeFile = null;
                els.wardrobeModal?.classList.add('hidden');
                await loadWardrobe();
                els.wardrobeBrowser?.classList.remove('hidden');
                els.wardrobeBrowser?.classList.add('flex');
            } else {
                throw new Error(data.message);
            }
        } catch (e) {
            showNotification(`❌ ${e.message}`, "error");
        } finally {
            els.saveWardrobeBtn.innerHTML = 'ANALYZE & SAVE';
            els.saveWardrobeBtn.disabled = false;
        }
    };
}

// ===================== PERSONAS =====================

async function loadPersonas() {
    try {
        const res = await fetch('/personas');
        allPersonas = await res.json();
        if (els.persona) {
            els.persona.innerHTML = `<option value="none">No Persona</option>` + allPersonas.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
        }
        const savedPersona = localStorage.getItem('pref_persona');
        if (savedPersona && allPersonas.some(p => p.id === savedPersona) && els.persona) {
            els.persona.value = savedPersona;
        }
        renderManager();
    } catch (e) {
        console.error(e);
    }
}

function renderManager() {
    if (!els.managerList) return;
    els.managerList.innerHTML = allPersonas.map(p => `
        <div class="flex justify-between items-center bg-[#0d1117] p-3 rounded border border-white/5 hover:border-purple-500/30 group transition-all mb-2">
            <div class="flex items-center gap-3">
                <img src="/persona-image/${p.id}?t=${Date.now()}" class="w-8 h-8 rounded-full object-cover bg-slate-700" onerror="this.style.display='none'">
                <span class="text-sm text-slate-200">${p.name}</span>
            </div>
            <div class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button onclick="window.editPersona('${p.id}')" class="text-xs bg-slate-800 hover:bg-cyan-600 text-slate-400 hover:text-white px-2 py-1 rounded"><i class="fas fa-pen"></i></button>
                <button onclick="window.deletePersona('${p.id}')" class="text-xs bg-slate-800 hover:bg-red-600 text-slate-400 hover:text-white px-2 py-1 rounded"><i class="fas fa-trash"></i></button>
            </div>
        </div>
    `).join('');
}

function openModal(mode, pid = null) {
    els.modal?.classList.remove('hidden');
    scanFile = null;
    
    // Check if NSFW mode is enabled
    const isNsfwEnabled = localStorage.getItem('nsfw_mode') === 'true';
    if (els.nsfwBodyFields) {
        if (isNsfwEnabled) {
            els.nsfwBodyFields.classList.remove('hidden');
        } else {
            els.nsfwBodyFields.classList.add('hidden');
        }
    }
    
    if (mode === 'create') {
        if (els.modalTitle) els.modalTitle.innerText = "New Persona";
        if (els.pName) els.pName.value = "";
        if (els.editPid) els.editPid.value = "";
        els.scanMsg?.classList.add('hidden');
        els.scanPreviewCont?.classList.add('hidden');
        els.editPhotoButtons?.classList.add('hidden');
        els.manualFields?.classList.add('hidden');
        els.scanBtn?.classList.remove('hidden');
        els.scanDrop?.classList.remove('hidden');
        
        // Clear NSFW fields
        if (els.pHips) els.pHips.value = "";
        if (els.pButt) els.pButt.value = "";
        if (els.pThighs) els.pThighs.value = "";
        if (els.pLegs) els.pLegs.value = "";
        if (els.pNipples) els.pNipples.value = "";
        if (els.pPubic) els.pPubic.value = "";
    } else {
        if (els.modalTitle) els.modalTitle.innerText = "Edit Persona";
        const p = allPersonas.find(x => String(x.id) === String(pid));
        if (!p) return;
        if (els.pName) els.pName.value = p.name;
        if (els.editPid) els.editPid.value = pid;
        els.scanDrop?.classList.add('hidden');
        els.scanBtn?.classList.add('hidden');
        els.scanMsg?.classList.add('hidden');
        els.scanPreviewCont?.classList.remove('hidden');
        els.editPhotoButtons?.classList.remove('hidden');
        if (els.scanMain) els.scanMain.src = `/persona-image/${pid}?t=${Date.now()}`;
        if (els.scanBlur) els.scanBlur.src = els.scanMain?.src || '';
        els.manualFields?.classList.remove('hidden');
        
        const s = p.subject || {};
        const bp = s.body_proportions || {};
        const h = s.hair || {};
        const intimate = s.intimate_details || {};
        
        if (els.pAge) els.pAge.value = s.age || "";
        if (els.pEthnicity) els.pEthnicity.value = s.ethnicity || "";
        if (els.pBuild) els.pBuild.value = bp.build || s.body_type || "";
        if (els.pChest) els.pChest.value = bp.chest || "";
        if (els.pShoulders) els.pShoulders.value = bp.shoulders || "";
        if (els.pWaist) els.pWaist.value = bp.waist_to_chest_ratio || bp.waist || "";
        if (els.pFace) els.pFace.value = s.face_structure || "";
        if (els.pSkin) els.pSkin.value = s.skin || "";
        if (els.pEyes) els.pEyes.value = s.eyes || "";
        if (els.pNose) els.pNose.value = s.nose || "";
        if (els.pLips) els.pLips.value = s.lips || "";
        if (els.pTattoos) els.pTattoos.value = s.tattoos || "";
        if (els.pEyewear) els.pEyewear.value = s.eyewear || "";
        if (els.pMakeup) els.pMakeup.value = s.makeup || "";
        if (els.pHairColor) els.pHairColor.value = h.color || "";
        if (els.pHairStyle) els.pHairStyle.value = h.style || "";
        
        // NSFW fields
        if (els.pHips) els.pHips.value = bp.hips || "";
        if (els.pButt) els.pButt.value = bp.butt || "";
        if (els.pThighs) els.pThighs.value = bp.thighs || "";
        if (els.pLegs) els.pLegs.value = bp.legs || "";
        if (els.pNipples) els.pNipples.value = intimate.nipples || "";
        if (els.pPubic) els.pPubic.value = h.pubic || "";
    }
}

window.editPersona = (pid) => { els.managerModal?.classList.add('hidden'); openModal('edit', pid); };
window.deletePersona = async (pid) => { if (confirm("Delete?")) { await fetch(`/personas/${pid}`, { method: 'DELETE' }); await loadPersonas(); } };

if (els.addBtn) els.addBtn.onclick = () => openModal('create');
if (els.closeModal) els.closeModal.onclick = () => els.modal?.classList.add('hidden');
if (els.manageBtn) els.manageBtn.onclick = () => els.managerModal?.classList.remove('hidden');
if (els.closeManager) els.closeManager.onclick = () => els.managerModal?.classList.add('hidden');

if (els.scanDrop) els.scanDrop.onclick = () => els.scanInput?.click();

if (els.scanInput) {
    els.scanInput.onchange = (e) => {
        if (e.target.files[0]) {
            scanFile = e.target.files[0];
            const r = new FileReader();
            r.onload = (ev) => {
                if (els.scanBlur) els.scanBlur.src = ev.target.result;
                if (els.scanMain) els.scanMain.src = ev.target.result;
                els.scanPreviewCont?.classList.remove('hidden');
                els.scanMsg?.classList.add('hidden');
                els.scanBtn?.classList.remove('hidden');
                els.scanDrop?.classList.add('hidden');
                els.editPhotoButtons?.classList.add('hidden');
            };
            r.readAsDataURL(scanFile);
        }
    };
}

// RE-SCAN button - uses existing persona image
if (els.triggerRescan) {
    els.triggerRescan.onclick = async () => {
        const pid = els.editPid?.value;
        if (!pid) return showNotification('⚠️ No persona to rescan', 'warning');
        
        els.scanMsg?.classList.remove('hidden');
        if (els.scanMsg) els.scanMsg.innerText = "Re-scanning...";
        els.triggerRescan.disabled = true;
        
        const fd = new FormData();
        fd.append('name', els.pName?.value || 'Character');
        fd.append('mode', 'edit');
        fd.append('pid', pid);
        fd.append('rescan', 'true');
        fd.append('model', els.visionModel?.value || 'qwen3-vl');
        fd.append('nsfw_mode', localStorage.getItem('nsfw_mode') === 'true' ? 'true' : 'false');
        
        try {
            const res = await fetch('/personas/create', { method: 'POST', body: fd });
            if (res.ok) {
                const data = await res.json();
                await loadPersonas();
                openModal('edit', pid);
                showNotification('✨ Re-scanned successfully', 'success');
            } else {
                const err = await res.json();
                showNotification(`❌ ${err.detail || 'Rescan failed'}`, 'error');
            }
        } catch (e) {
            showNotification(`❌ Error: ${e.message}`, 'error');
        } finally {
            els.scanMsg?.classList.add('hidden');
            els.triggerRescan.disabled = false;
        }
    };
}

// NEW PHOTO button - allows uploading a different image
if (els.triggerNewPhoto) {
    els.triggerNewPhoto.onclick = () => {
        els.scanPreviewCont?.classList.add('hidden');
        els.editPhotoButtons?.classList.add('hidden');
        els.scanDrop?.classList.remove('hidden');
        els.scanBtn?.classList.remove('hidden');
        if (els.scanBtn) els.scanBtn.innerText = "UPLOAD & RE-SCAN";
    };
}

if (els.scanBtn) {
    els.scanBtn.onclick = async () => {
        if (!els.pName?.value || !scanFile) return showNotification('⚠️ Name/Photo required!', 'warning');
        els.scanBtn.disabled = true;
        els.scanBtn.innerText = "SCANNING...";
        els.scanMsg?.classList.remove('hidden');
        
        const fd = new FormData();
        fd.append('name', els.pName.value);
        fd.append('file', scanFile);
        fd.append('model', els.visionModel?.value || 'qwen3-vl');
        fd.append('nsfw_mode', localStorage.getItem('nsfw_mode') === 'true' ? 'true' : 'false');
        if (els.editPid?.value) {
            fd.append('mode', 'edit');
            fd.append('pid', els.editPid.value);
        }
        
        try {
            const res = await fetch('/personas/create', { method: 'POST', body: fd });
            if (res.ok) {
                const data = await res.json();
                await loadPersonas();
                if (els.editPid) els.editPid.value = data.id;
                openModal('edit', data.id);
                showNotification(`✨ Persona Scanned`, 'success');
            }
        } catch (e) {
            showNotification(`❌ Error: ${e.message}`, 'error');
        } finally {
            els.scanBtn.disabled = false;
            els.scanBtn.innerText = "SCAN & AUTO-FILL";
            els.scanMsg?.classList.add('hidden');
        }
    };
}

if (els.updateManualBtn) {
    els.updateManualBtn.onclick = async () => {
        const pid = els.editPid?.value;
        if (!pid) return;
        
        const sub = {
            age: els.pAge?.value,
            ethnicity: els.pEthnicity?.value,
            body_proportions: {
                build: els.pBuild?.value,
                chest: els.pChest?.value,
                shoulders: els.pShoulders?.value,
                waist_to_chest_ratio: els.pWaist?.value,
                // NSFW body fields
                hips: els.pHips?.value || undefined,
                butt: els.pButt?.value || undefined,
                thighs: els.pThighs?.value || undefined,
                legs: els.pLegs?.value || undefined
            },
            body_type: els.pBuild?.value,
            face_structure: els.pFace?.value,
            skin: els.pSkin?.value,
            eyes: els.pEyes?.value,
            nose: els.pNose?.value,
            lips: els.pLips?.value,
            tattoos: els.pTattoos?.value,
            eyewear: els.pEyewear?.value,
            makeup: els.pMakeup?.value,
            hair: {
                color: els.pHairColor?.value,
                style: els.pHairStyle?.value,
                pubic: els.pPubic?.value || undefined
            },
            // NSFW intimate details
            intimate_details: {
                nipples: els.pNipples?.value || undefined
            }
        };
        
        await fetch(`/personas/${pid}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: els.pName?.value, subject: sub })
        });
        await loadPersonas();
        els.modal?.classList.add('hidden');
        showNotification(`✅ Updated`, 'success');
    };
}

if (els.triggerNewPhoto) {
    els.triggerNewPhoto.onclick = () => {
        els.scanPreviewCont?.classList.add('hidden');
        els.editPhotoButtons?.classList.add('hidden');
        els.scanDrop?.classList.remove('hidden');
        els.scanBtn?.classList.remove('hidden');
    };
}

// Persona change handler
if (els.persona) {
    els.persona.addEventListener('change', () => {
        localStorage.setItem('pref_persona', els.persona.value);
    });
}

// ===================== STYLES =====================

let allStyles = {};

async function loadStyles() {
    try {
        const res = await fetch('/styles');
        allStyles = await res.json();
        window.allStyles = allStyles;
        if (els.styleNarrative) {
            els.styleNarrative.innerHTML = `<option value="default">Standard</option>` +
                Object.keys(allStyles).map(name => `<option value="${name}">${name}</option>`).join('');
        }
        bindSticky(els.styleNarrative, 'promptStyle', 'default');
    } catch (e) {
        console.error(e);
    }
}

if (els.openStyleBtn) els.openStyleBtn.onclick = () => els.styleModal?.classList.remove('hidden');
if (els.closeStyleModal) els.closeStyleModal.onclick = () => els.styleModal?.classList.add('hidden');

if (els.analyzeStyleBtn) {
    els.analyzeStyleBtn.onclick = async () => {
        const raw = els.styleInputRaw?.value.trim();
        if (!raw) return showNotification("⚠️ Paste a prompt first", "warning");
        
        els.analyzeStyleBtn.innerText = "⏳ ANALYZING...";
        els.analyzeStyleBtn.disabled = true;
        
        try {
            const res = await fetch('/styles/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: raw, model: els.writerModel?.value || 'llama3.2' })
            });
            const data = await res.json();
            if (data.status === "error" || !data.instruction) throw new Error(data.message || "No instruction");
            if (els.styleInstruction) els.styleInstruction.value = data.instruction;
            els.styleAnalysisResult?.classList.remove('hidden');
        } catch (e) {
            showNotification(`❌ ${e.message}`, "error");
        } finally {
            els.analyzeStyleBtn.innerText = "ANALYZE STRUCTURE";
            els.analyzeStyleBtn.disabled = false;
        }
    };
}

if (els.saveStyleBtn) {
    els.saveStyleBtn.onclick = async () => {
        const name = els.styleName?.value.trim();
        const instr = els.styleInstruction?.value.trim();
        if (!name) return;
        
        await fetch('/styles/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, instruction: instr })
        });
        await loadStyles();
        if (els.styleNarrative) els.styleNarrative.value = name;
        els.styleModal?.classList.add('hidden');
        showNotification("✨ Saved", "success");
    };
}

// ===================== HISTORY =====================

async function updateHistoryUI() {
    console.log('[HISTORY SIDEBAR] Starting update...');
    try {
        const res = await fetch('/history');
        const data = await res.json();
        console.log('[HISTORY SIDEBAR] Raw response:', data);
        const h = Array.isArray(data) ? data : (data.items || []);
        console.log('[HISTORY SIDEBAR] Parsed items:', h.length, 'entries');
        
        if (els.historyList) {
            console.log('[HISTORY SIDEBAR] Found historyList element');
            if (h.length === 0) {
                els.historyList.innerHTML = '<div class="text-[10px] text-slate-500 text-center py-4">No history yet</div>';
            } else {
                els.historyList.innerHTML = h.slice(0, 5).map((item, i) => `
                    <div onclick="window.loadH(${i})" class="p-2 rounded bg-[#12151a] border border-white/5 hover:border-cyan-500/50 cursor-pointer text-[10px] text-slate-300 truncate transition-colors">
                        <span class="text-cyan-500 mr-1">${item.timestamp?.split(' ')[1] || '--'}</span> ${item.filename || 'unknown'}
                    </div>
                `).join('');
            }
        } else {
            console.warn('[HISTORY SIDEBAR] historyList element NOT found!');
        }
        window.histData = h;
    } catch (e) {
        console.error('[HISTORY SIDEBAR] Error:', e);
    }
}

window.loadH = (i) => {
    if (window.histData?.[i]?.json) {
        store.json = window.histData[i].json;
        switchTab('json');
    }
};

if (els.openHistoryBtn) {
    els.openHistoryBtn.onclick = () => {
        els.historyModal?.classList.remove('hidden');
        loadFullHistory();
    };
}

if (els.closeHistoryModal) {
    els.closeHistoryModal.onclick = () => els.historyModal?.classList.add('hidden');
}

async function loadFullHistory(search = '', sortBy = 'timestamp', sortOrder = 'desc') {
    try {
        const params = new URLSearchParams({ search, sort_by: sortBy, sort_order: sortOrder, limit: '200' });
        console.log('[HISTORY MODAL] Fetching with params:', params.toString());
        const res = await fetch(`/history?${params}`);
        const data = await res.json();
        console.log('[HISTORY MODAL] Raw response:', data);
        const items = Array.isArray(data) ? data : (data.items || []);
        console.log('[HISTORY MODAL] Parsed items:', items.length, 'entries');
        window.histData = items; // Update global histData
        renderFullHistory(items);
    } catch (e) {
        console.error('[HISTORY MODAL] Error:', e);
    }
}

function renderFullHistory(items) {
    console.log('[RENDER HISTORY] Items to render:', items.length);
    if (!els.historyListFull) {
        console.warn('[RENDER HISTORY] historyListFull element not found!');
        return;
    }
    
    // Show empty state if no items
    if (items.length === 0) {
        els.historyListFull.innerHTML = `
            <div class="flex flex-col items-center justify-center h-64 text-slate-500">
                <i class="fas fa-history text-4xl mb-4 opacity-50"></i>
                <p class="text-sm">No history yet</p>
                <p class="text-xs mt-1">Analyze some images to build your history</p>
            </div>
        `;
        // Update stats to zeros
        const histTotal = document.getElementById('hist-total');
        const histNewest = document.getElementById('hist-newest');
        const histPersonas = document.getElementById('hist-personas');
        const histModels = document.getElementById('hist-models');
        if (histTotal) histTotal.textContent = '0';
        if (histNewest) histNewest.textContent = '--';
        if (histPersonas) histPersonas.textContent = '0';
        if (histModels) histModels.textContent = '0';
        return;
    }
    
    els.historyListFull.innerHTML = items.map((item, i) => `
        <div class="flex items-center gap-3 p-3 bg-[#0d1117] rounded-lg border border-white/5 hover:border-cyan-500/30 cursor-pointer transition-all group" onclick="window.historyLoadItem(${i})" ondblclick="window.historyLoadAndGenerate(${i})">
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 mb-1">
                    <span class="text-xs text-slate-200 font-medium truncate">${item.filename || 'untitled'}</span>
                    ${item.persona && item.persona !== 'none' ? `<span class="text-[9px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded">${item.persona}</span>` : ''}
                </div>
                <div class="text-[10px] text-slate-500">${item.timestamp || '--'} • ${item.model || '--'}</div>
            </div>
            <button onclick="event.stopPropagation(); window.historyDeleteItem(${i})" class="text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all p-1">
                <i class="fas fa-trash text-xs"></i>
            </button>
        </div>
    `).join('');
    
    // Update stats
    const histTotal = document.getElementById('hist-total');
    const histNewest = document.getElementById('hist-newest');
    const histPersonas = document.getElementById('hist-personas');
    const histModels = document.getElementById('hist-models');
    
    if (histTotal) histTotal.textContent = items.length;
    if (items.length > 0) {
        if (histNewest) histNewest.textContent = items[0].timestamp?.split(' ')[0] || '--';
        const personas = new Set(items.map(i => i.persona).filter(p => p && p !== 'none'));
        if (histPersonas) histPersonas.textContent = personas.size;
        const models = new Set(items.map(i => i.model).filter(Boolean));
        if (histModels) histModels.textContent = models.size;
    }
}

window.historyLoadItem = (i) => {
    const items = window.histData || [];
    if (items[i]?.json) {
        store.json = items[i].json;
        switchTab('json');
        els.historyModal?.classList.add('hidden');
    }
};

window.historyLoadAndGenerate = (i) => {
    window.historyLoadItem(i);
    setTimeout(() => els.actionBtn?.click(), 100);
};

window.historyDeleteItem = async (i) => {
    if (!confirm('Delete this entry?')) return;
    try {
        await fetch(`/history/${i}`, { method: 'DELETE' });
        loadFullHistory();
        updateHistoryUI();
    } catch (e) {
        showNotification('Failed to delete', 'error');
    }
};

if (els.historySearch) {
    els.historySearch.addEventListener('input', (e) => {
        const [sortBy, sortOrder] = (els.historySort?.value || 'timestamp-desc').split('-');
        loadFullHistory(e.target.value, sortBy, sortOrder);
    });
}

if (els.historySort) {
    els.historySort.addEventListener('change', (e) => {
        const [sortBy, sortOrder] = e.target.value.split('-');
        loadFullHistory(els.historySearch?.value || '', sortBy, sortOrder);
    });
}

// ===================== SETTINGS =====================

if (els.settingsBtn) els.settingsBtn.onclick = () => els.settingsModal?.classList.remove('hidden');
if (els.closeSettings) els.closeSettings.onclick = () => els.settingsModal?.classList.add('hidden');

// Load saved API keys
if (els.keyGoogle) els.keyGoogle.value = localStorage.getItem('google_key') || "";
if (els.keyFal) els.keyFal.value = localStorage.getItem('fal_key') || "";
if (els.keyXai) els.keyXai.value = localStorage.getItem('xai_key') || "";

// Load NSFW mode
if (els.nsfwMode) els.nsfwMode.checked = localStorage.getItem('nsfw_mode') === 'true';

if (els.saveKeysBtn) {
    els.saveKeysBtn.onclick = () => {
        localStorage.setItem('google_key', els.keyGoogle?.value || '');
        localStorage.setItem('fal_key', els.keyFal?.value || '');
        localStorage.setItem('xai_key', els.keyXai?.value || '');
        
        // Save NSFW mode
        localStorage.setItem('nsfw_mode', els.nsfwMode?.checked ? 'true' : 'false');
        
        // Save image generation settings
        const imgGenSettings = {
            backend: els.imgGenBackend?.value || 'comfyui',
            url: els.imgGenUrl?.value || 'http://127.0.0.1:8188',
            width: els.imgGenWidth?.value || '1024',
            height: els.imgGenHeight?.value || '1024',
            steps: els.imgGenSteps?.value || '25'
        };
        localStorage.setItem('imgGenSettings', JSON.stringify(imgGenSettings));
        
        els.settingsModal?.classList.add('hidden');
        showNotification("🔑 Settings Saved", "success");
    };
}

// ===================== GENERATE IMAGE =====================

if (els.generateImageBtn) {
    els.generateImageBtn.onclick = async () => {
        // Get prompt from editor or store
        let prompt = '';
        let negativePrompt = '';
        
        if (currentMode === 'json' && store.json) {
            // Convert JSON to simple prompt
            const j = store.json;
            const parts = [];
            
            // Subject
            if (j.subject) {
                const s = j.subject;
                const subjectParts = [s.gender, s.age, s.ethnicity, s.body_type].filter(Boolean);
                if (subjectParts.length) parts.push(subjectParts.join(' '));
                if (s.hair) {
                    const hair = typeof s.hair === 'object' 
                        ? [s.hair.color, s.hair.style, s.hair.length].filter(Boolean).join(' ') + ' hair'
                        : s.hair;
                    parts.push(hair);
                }
            }
            
            // Clothing
            if (j.clothing) {
                const c = j.clothing;
                if (c.top) parts.push(c.top);
                if (c.bottom) parts.push(c.bottom);
                if (c.accessories) {
                    parts.push(Array.isArray(c.accessories) ? c.accessories.join(', ') : c.accessories);
                }
            }
            
            // Environment
            if (j.environment) {
                const e = j.environment;
                if (e.setting) parts.push(e.setting);
                if (e.time_of_day) parts.push(e.time_of_day);
            }
            
            // Pose
            if (j.pose?.description) parts.push(j.pose.description);
            
            prompt = parts.join(', ');
            negativePrompt = Array.isArray(j.negative_prompt) ? j.negative_prompt.join(', ') : (j.negative_prompt || '');
            
        } else if (currentMode === 'prompt' && store.prompt) {
            // Parse prompt - split by negative prompt separator if present
            const promptText = store.prompt;
            const negSplit = promptText.split(/---\s*\n*NEGATIVE PROMPT:\s*/i);
            prompt = negSplit[0].replace(/^\/\/ REFERENCE:.*\n\n/i, '').trim();
            negativePrompt = negSplit[1]?.trim() || store.negativePrompt || '';
        }
        
        if (!prompt) {
            showNotification('⚠️ No prompt to generate from', 'warning');
            return;
        }
        
        // Get settings
        const settings = JSON.parse(localStorage.getItem('imgGenSettings') || '{}');
        const backend = settings.backend || els.imgGenBackend?.value || 'comfyui';
        const apiUrl = settings.url || els.imgGenUrl?.value || 'http://127.0.0.1:8188';
        const width = parseInt(settings.width || els.imgGenWidth?.value || '1024');
        const height = parseInt(settings.height || els.imgGenHeight?.value || '1024');
        const steps = parseInt(settings.steps || els.imgGenSteps?.value || '25');
        
        showLoading("GENERATING", "Creating image...");
        console.log('[IMG GEN] Starting with backend:', backend);
        console.log('[IMG GEN] Prompt:', prompt.substring(0, 100) + '...');
        console.log('[IMG GEN] Negative:', negativePrompt.substring(0, 50) + '...');
        
        try {
            const res = await fetch('/generate-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    negative_prompt: negativePrompt,
                    backend,
                    api_url: apiUrl,
                    width,
                    height,
                    steps,
                    model: els.cloudModel?.value || 'flux'
                })
            });
            
            const data = await res.json();
            console.log('[IMG GEN] Response:', data);
            
            if (data.error) throw new Error(data.error);
            
            // Show result
            if (data.image_url || data.image_base64) {
                const imgResultModal = document.getElementById('img-result-modal');
                const imgResult = document.getElementById('img-result');
                if (imgResultModal && imgResult) {
                    if (data.image_base64) {
                        imgResult.src = `data:image/png;base64,${data.image_base64}`;
                    } else {
                        imgResult.src = data.image_url;
                    }
                    imgResultModal.classList.remove('hidden');
                }
                showNotification('🎨 Image generated!', 'success');
            } else {
                showNotification('✅ Request sent to ' + backend, 'success');
            }
            
        } catch (e) {
            console.error('[IMG GEN] Error:', e);
            showNotification('❌ ' + e.message, 'error');
        } finally {
            hideLoading();
        }
    };
}

if (els.quitBtn) {
    els.quitBtn.onclick = async () => {
        await fetch('/shutdown', { method: 'POST' });
        window.close();
    };
}

// ===================== VRAM =====================

if (els.flushBtn) {
    els.flushBtn.onclick = async () => {
        showNotification("🧹 Cleaning VRAM...", "info");
        try {
            await fetch('/system/free-vram', { method: 'POST' });
            showNotification("✨ Memory Freed", "success");
        } catch (e) {
            showNotification("❌ Cleanup Failed", "error");
        }
    };
}

// ===================== BATCH =====================

if (els.batchBtn) {
    els.batchBtn.onclick = () => {
        if (typeof window.openBatchModal === 'function') {
            window.openBatchModal();
        } else {
            showNotification('⚠️ Batch module not loaded', 'warning');
        }
    };
}

// ===================== CLOUD GENERATION =====================

if (els.closeImgModal) els.closeImgModal.onclick = () => els.imgModal?.classList.add('hidden');

// ===================== SYSTEM MONITOR =====================

function startSystemMonitor() {
    async function updateStats() {
        try {
            const res = await fetch('/system/stats');
            const data = await res.json();
            
            const getColor = (percent) => {
                if (percent >= 90) return 'text-red-400';
                if (percent >= 75) return 'text-yellow-400';
                return 'text-green-400';
            };
            
            if (data.cpu) {
                document.getElementById('mon-cpu').textContent = `${data.cpu.percent}%`;
                document.getElementById('mon-cpu').className = `text-[10px] font-mono ${getColor(data.cpu.percent)}`;
            }
            if (data.ram) {
                document.getElementById('mon-ram').textContent = `${data.ram.percent}%`;
                document.getElementById('mon-ram').className = `text-[10px] font-mono ${getColor(data.ram.percent)}`;
            }
            if (data.gpu?.available) {
                document.getElementById('mon-gpu').textContent = `${data.gpu.utilization || 0}%`;
                document.getElementById('mon-gpu').className = `text-[10px] font-mono ${getColor(data.gpu.utilization || 0)}`;
                document.getElementById('mon-vram').textContent = `${(data.gpu.vram_used / 1024).toFixed(1)}G`;
                document.getElementById('mon-vram').className = `text-[10px] font-mono ${getColor(data.gpu.vram_percent || 0)}`;
                if (data.gpu.temperature) {
                    document.getElementById('mon-temp').textContent = `${data.gpu.temperature}°`;
                    document.getElementById('mon-temp').className = `text-[10px] font-mono ${data.gpu.temperature >= 80 ? 'text-red-400' : data.gpu.temperature >= 65 ? 'text-yellow-400' : 'text-green-400'}`;
                }
            }
        } catch (e) {
            // Silent fail
        }
    }
    
    updateStats();
    setInterval(updateStats, 15000);
}

// ===================== INITIALIZATION =====================

function showOllamaWarning(message) {
    // Remove existing warning if any
    document.getElementById('ollama-warning')?.remove();
    
    const banner = document.createElement('div');
    banner.id = 'ollama-warning';
    banner.className = 'fixed top-0 left-0 right-0 bg-red-600 text-white text-center py-3 text-sm font-medium z-[100] flex items-center justify-center gap-3 shadow-lg';
    banner.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>${message}</span>
        <code class="bg-red-800 px-2 py-0.5 rounded text-xs ml-2">ollama serve</code>
        <button onclick="location.reload()" class="ml-4 bg-white text-red-600 px-3 py-1 rounded text-xs font-bold hover:bg-red-100">
            <i class="fas fa-sync-alt mr-1"></i> Retry
        </button>
    `;
    document.body.prepend(banner);
    
    // Also show notification
    showNotification("⚠️ " + message, "error", 15000);
}

(async () => {
    // Load models
    try {
        const res = await fetch('/models');
        const data = await res.json();
        
        if (data.ollama_running === false) {
            showOllamaWarning("Ollama is not running! Start it with:");
        } else if (data.vision_models?.length === 0) {
            showNotification("⚠️ No vision models found. Install with: ollama pull qwen2.5-vl", "warning", 10000);
        }
        
        const visionModels = data.vision_models || [];
        const textModels = data.text_models || data.all_model_names || [];
        const allModels = data.all_model_names || [...visionModels, ...textModels];
        
        if (els.visionModel && visionModels.length > 0) {
            els.visionModel.innerHTML = visionModels.map(m => `<option value="${m}">${m}</option>`).join('');
            bindSticky(els.visionModel, 'mem_vision', visionModels[0]);
        }
        
        if (els.writerModel && allModels.length > 0) {
            els.writerModel.innerHTML = allModels.map(m => `<option value="${m}">${m}</option>`).join('');
            bindSticky(els.writerModel, 'mem_writer', textModels[0] || allModels[0]);
        }
        
        if (els.taggerModel && visionModels.length > 0) {
            els.taggerModel.innerHTML = visionModels.map(m => `<option value="${m}">${m}</option>`).join('');
            bindSticky(els.taggerModel, 'mem_tagger', visionModels[0]);
        }
    } catch (e) {
        console.error('Failed to load models:', e);
        showOllamaWarning("Cannot connect to backend. Is Ollama running?");
    }
    
    // Load data
    await loadPersonas();
    await loadWardrobe();
    await loadStyles();
    updateHistoryUI();
    
    // Bind sticky preferences
    bindSticky(els.time, 'time');
    bindSticky(els.ratio, 'ratio');
    bindSticky(els.styleVisual, 'styleVisual');
    bindSticky(els.quality, 'quality');
    bindSticky(els.refMode, 'refMode', false);
    bindSticky(els.cloudModel, 'cloudModel', 'grok');
    bindSticky(els.hairStyle, 'hairStyle');
    bindSticky(els.hairColor, 'hairColor');
    bindSticky(els.makeup, 'makeup');
    bindSticky(els.glasses, 'glasses');
    bindSticky(els.hairSource, 'hairSource', 'persona');
    bindSticky(els.promptType, 'promptType', 'narrative');
    bindSticky(els.includeNegative, 'includeNegative', true); // Negative prompt toggle
    
    // Load version
    try {
        const vRes = await fetch('/version');
        const vData = await vRes.json();
        document.title = `AI Prompt Director v${vData.local}`;
        if (document.getElementById('app-version')) {
            document.getElementById('app-version').textContent = `v${vData.local}`;
        }
    } catch (e) {}
    
    // Load image generation settings
    try {
        const imgGenSettings = JSON.parse(localStorage.getItem('imgGenSettings') || '{}');
        if (els.imgGenBackend && imgGenSettings.backend) els.imgGenBackend.value = imgGenSettings.backend;
        if (els.imgGenUrl && imgGenSettings.url) els.imgGenUrl.value = imgGenSettings.url;
        if (els.imgGenWidth && imgGenSettings.width) els.imgGenWidth.value = imgGenSettings.width;
        if (els.imgGenHeight && imgGenSettings.height) els.imgGenHeight.value = imgGenSettings.height;
        if (els.imgGenSteps && imgGenSettings.steps) els.imgGenSteps.value = imgGenSettings.steps;
    } catch (e) {}
    
    // Load batch modal
    try {
        const batchRes = await fetch('/batch-modal');
        const batchHtml = await batchRes.text();
        const container = document.getElementById('batch-modal-container');
        if (container) {
            container.innerHTML = batchHtml;
            const scripts = container.querySelectorAll('script');
            scripts.forEach(oldScript => {
                const newScript = document.createElement('script');
                newScript.textContent = oldScript.textContent;
                oldScript.parentNode.replaceChild(newScript, oldScript);
            });
        }
    } catch (e) {
        console.error('Failed to load batch modal:', e);
    }
    
    // Start system monitor
    startSystemMonitor();
    
    // Initial UI state
    switchTab('json'); // Set default tab and button text
    updateCopyButtonVisibility();
    updatePlaceholderVisibility();
})();