/**
 * ComfyUI Provider - Simplified Integration
 * Just adds "ComfyUI" to the dropdown and handles generation
 */

(function() {
    'use strict';
    
    // Wait for DOM to be ready
    function init() {
        console.log('🖥️ ComfyUI Provider loading...');
        
        const cloudModel = document.getElementById('cloud-model');
        if (!cloudModel) {
            console.warn('cloud-model dropdown not found');
            return;
        }
        
        // Add ComfyUI option at the beginning
        const comfyOption = document.createElement('option');
        comfyOption.value = 'comfyui';
        comfyOption.textContent = '🖥️ ComfyUI (Local)';
        comfyOption.style.backgroundColor = '#0d1117';
        comfyOption.style.color = '#e2e8f0';
        
        // Insert as first option
        cloudModel.insertBefore(comfyOption, cloudModel.firstChild);
        
        // Fix dropdown styling
        cloudModel.style.backgroundColor = '#0d1117';
        cloudModel.style.color = '#e2e8f0';
        
        console.log('✓ Added ComfyUI option to dropdown');
        
        // Store original click handler
        const genBtn = document.getElementById('generate-cloud-btn');
        if (genBtn) {
            const originalHandler = genBtn.onclick;
            
            genBtn.onclick = async function(e) {
                const provider = cloudModel.value;
                
                if (provider === 'comfyui') {
                    e.preventDefault();
                    e.stopPropagation();
                    await generateWithComfyUI();
                    return false;
                }
                
                // Call original handler for other providers
                if (originalHandler) {
                    return originalHandler.call(this, e);
                }
            };
            console.log('✓ Hooked generate button');
        }
    }
    
    async function generateWithComfyUI() {
        // Get prompt from current editor content
        const editor = document.getElementById('editor');
        if (!editor) {
            showNotif('⚠️ Editor not found', 'error');
            return;
        }
        
        const content = editor.innerText || editor.textContent || '';
        
        // Extract positive and negative prompts
        let positive = '';
        let negative = 'lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, jpeg artifacts';
        
        if (content.includes('POSITIVE PROMPT:')) {
            const parts = content.split('NEGATIVE PROMPT:');
            positive = parts[0]
                .replace('POSITIVE PROMPT:', '')
                .replace('// REFERENCE:', '')
                .replace(/---/g, '')
                .trim();
            if (parts[1]) {
                negative = parts[1].trim();
            }
        } else if (content.includes('POSITIVE TAGS:')) {
            const parts = content.split('NEGATIVE:');
            positive = parts[0].replace('POSITIVE TAGS:', '').trim();
            if (parts[1]) {
                negative = parts[1].trim();
            }
        } else {
            // Use raw content (remove comments)
            positive = content.replace(/^\/\/.*$/gm, '').trim();
        }
        
        if (!positive || positive.startsWith('//')) {
            showNotif('⚠️ Generate a prompt first (JSON → PROMPT tab)', 'warning');
            return;
        }
        
        // Show loading modal
        const imgModal = document.getElementById('img-result-modal');
        const cloudLoader = document.getElementById('cloud-loader');
        const cloudResult = document.getElementById('cloud-result-img');
        const cloudError = document.getElementById('cloud-error');
        
        if (imgModal) imgModal.classList.remove('hidden');
        if (cloudResult) cloudResult.classList.add('hidden');
        if (cloudError) cloudError.classList.add('hidden');
        if (cloudLoader) {
            cloudLoader.classList.remove('hidden');
            cloudLoader.innerHTML = `
                <div class="w-20 h-20 border-4 border-green-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                <div class="text-green-400 font-mono animate-pulse tracking-widest">GENERATING WITH COMFYUI...</div>
                <div class="text-slate-500 text-xs mt-2">Please wait, this may take a moment</div>
            `;
        }
        
        // Get ComfyUI settings from localStorage or use defaults
        const settings = {
            steps: parseInt(localStorage.getItem('comfy_steps') || '25'),
            cfg: parseFloat(localStorage.getItem('comfy_cfg') || '7'),
            width: parseInt(localStorage.getItem('comfy_width') || '1024'),
            height: parseInt(localStorage.getItem('comfy_height') || '1024'),
            sampler: localStorage.getItem('comfy_sampler') || 'euler',
            scheduler: localStorage.getItem('comfy_scheduler') || 'normal',
            checkpoint: localStorage.getItem('comfy_checkpoint') || null
        };
        
        console.log('ComfyUI Generation:', { positive: positive.substring(0, 100) + '...', settings });
        
        try {
            const response = await fetch('/comfy/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    positive_prompt: positive,
                    negative_prompt: negative,
                    width: settings.width,
                    height: settings.height,
                    steps: settings.steps,
                    cfg: settings.cfg,
                    sampler: settings.sampler,
                    scheduler: settings.scheduler,
                    checkpoint: settings.checkpoint,
                    workflow_type: 'sdxl'  // Default, ComfyUI will use whatever model is loaded
                })
            });
            
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || errData.message || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.images && data.images.length > 0) {
                if (cloudResult) {
                    cloudResult.src = 'data:image/png;base64,' + data.images[0].data;
                    cloudResult.classList.remove('hidden');
                }
                showNotif('✅ Image generated with ComfyUI!', 'success');
            } else {
                throw new Error(data.message || 'No image returned');
            }
            
        } catch (error) {
            console.error('ComfyUI error:', error);
            
            let errorMessage = error.message;
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                errorMessage = 'Cannot connect to ComfyUI. Make sure it\'s running with: python main.py --listen';
            }
            
            if (cloudError) {
                cloudError.innerHTML = `
                    <div class="text-red-400">
                        <i class="fas fa-exclamation-triangle mr-2"></i>
                        <strong>ComfyUI Error</strong>
                    </div>
                    <div class="mt-2 text-sm">${errorMessage}</div>
                    <div class="mt-3 text-xs text-slate-500">
                        <p>Troubleshooting:</p>
                        <ul class="list-disc list-inside mt-1">
                            <li>Start ComfyUI: <code class="bg-slate-800 px-1 rounded">python main.py --listen</code></li>
                            <li>Check it's running on port 8188</li>
                            <li>Make sure a model is loaded</li>
                        </ul>
                    </div>
                `;
                cloudError.classList.remove('hidden');
            }
            
            showNotif('❌ ComfyUI: ' + errorMessage, 'error');
            
        } finally {
            if (cloudLoader) cloudLoader.classList.add('hidden');
        }
    }
    
    // Simple notification helper
    function showNotif(message, type = 'info') {
        // Use existing notification system if available
        if (typeof window.showNotification === 'function') {
            window.showNotification(message, type);
            return;
        }
        
        // Fallback
        const colors = {
            success: { border: '#22c55e', text: '#4ade80', bg: 'rgba(34,197,94,0.1)' },
            error: { border: '#ef4444', text: '#f87171', bg: 'rgba(239,68,68,0.1)' },
            warning: { border: '#f59e0b', text: '#fbbf24', bg: 'rgba(245,158,11,0.1)' },
            info: { border: '#0ea5e9', text: '#38bdf8', bg: 'rgba(14,165,233,0.1)' }
        };
        const c = colors[type] || colors.info;
        
        const n = document.createElement('div');
        n.style.cssText = `
            position: fixed; top: 16px; right: 16px; z-index: 9999;
            padding: 16px 24px; border-radius: 8px;
            border: 1px solid ${c.border}; background: #161b22;
            color: ${c.text}; font-size: 14px; font-weight: 500;
            transform: translateX(0); transition: transform 0.3s;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        `;
        n.textContent = message;
        document.body.appendChild(n);
        
        setTimeout(() => {
            n.style.transform = 'translateX(400px)';
            setTimeout(() => n.remove(), 300);
        }, 3000);
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Also expose for manual testing
    window.generateWithComfyUI = generateWithComfyUI;
    
    console.log('✅ ComfyUI Provider loaded');
})();
