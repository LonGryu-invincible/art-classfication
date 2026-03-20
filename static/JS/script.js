// --- BIẾN TOÀN CỤC ---
let globalRawFiles = [];    
let globalResults = [];    
let globalPreviews = {};    

const MAX_LIMIT = 30; 

// Các phần tử DOM
const fileInput = document.getElementById('file-input');
const folderInput = document.getElementById('folder-input');
const dropZone = document.getElementById('drop-zone');
const artGrid = document.getElementById('art-grid');
const exportBtn = document.getElementById('export-btn');
const scanningStatus = document.getElementById('scanning-status');
const btnBrowseFiles = document.getElementById('btn-browse-files');
const btnSelectFolder = document.getElementById('id-select-folder-trigger'); 

// --- HÀM KHỞI TẠO SỰ KIỆN ---
function initEventListeners() {
    if (!btnBrowseFiles || !fileInput || !dropZone) return;

    btnBrowseFiles.onclick = (e) => { e.stopPropagation(); fileInput.click(); };

    if (btnSelectFolder) {
        btnSelectFolder.onclick = (e) => { e.stopPropagation(); folderInput.click(); };
    }

    fileInput.onchange = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) handleUpload(files);
        e.target.value = ""; 
    };

    folderInput.onchange = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) handleUpload(files);
        e.target.value = ""; 
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    dropZone.ondragover = () => dropZone.classList.add('border-blue-500', 'bg-blue-500/5');
    dropZone.ondragleave = () => dropZone.classList.remove('border-blue-500', 'bg-blue-500/5');

    dropZone.ondrop = async (e) => {
        dropZone.classList.remove('border-blue-500', 'bg-blue-500/5');
        const items = e.dataTransfer.items;
        if (items) {
            let filesFound = [];
            for (let i = 0; i < items.length; i++) {
                const item = items[i].webkitGetAsEntry();
                if (item) {
                    const extracted = await scanFilesRecursively(item);
                    filesFound = filesFound.concat(extracted);
                }
            }
            handleUpload(filesFound);
        }
    };
}

async function scanFilesRecursively(entry) {
    let files = [];
    if (entry.isFile) {
        const file = await new Promise(resolve => entry.file(resolve));
        if (file.type.startsWith('image/')) files.push(file);
    } else if (entry.isDirectory) {
        const dirReader = entry.createReader();
        const entries = await new Promise(resolve => dirReader.readEntries(resolve));
        for (const childEntry of entries) {
            const childFiles = await scanFilesRecursively(childEntry);
            files = files.concat(childFiles);
        }
    }
    return files;
}

// --- XỬ LÝ UPLOAD VÀ GỌI AI ---
async function handleUpload(files) {
    let validFiles = files.filter(f => f.type.startsWith('image/'));
    if (validFiles.length === 0) return;
    
    if (validFiles.length > MAX_LIMIT) {
        alert(`⚠️ Giới hạn: Chỉ xử lý 30 ảnh đầu tiên.`);
        validFiles = validFiles.slice(0, MAX_LIMIT);
    }

    const contentArea = document.getElementById('drop-zone-content');
    const buttonsArea = document.querySelector('.flex.flex-wrap.justify-center'); 
    
    if(contentArea) contentArea.style.display = 'none';
    if(buttonsArea) buttonsArea.style.display = 'none';

    const loader = document.createElement('div');
    loader.id = "temp-loader";
    loader.innerHTML = `
        <div class="flex flex-col items-center animate-pulse py-12">
            <div class="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
            <p class="text-blue-400 font-bold uppercase tracking-widest text-[11px]">AI Vault Scanning ${validFiles.length} images...</p>
        </div>`;
    dropZone.appendChild(loader);

    const formData = new FormData();
    validFiles.forEach(file => {
        formData.append("files", file);
        const pureFileName = file.name.split('/').pop().split('\\').pop();
        const previewUrl = URL.createObjectURL(file);
        globalPreviews[pureFileName] = previewUrl; 
        globalRawFiles.push(file); 
    });

    try {
        const response = await fetch('/predict-batch', { method: 'POST', body: formData });
        if (!response.ok) throw new Error("Server error");
        
        const data = await response.json();
        globalResults = globalResults.concat(data.results);
        
        updateStats(data.total_latency_ms);
        renderArtGrid();
        
        exportBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        exportBtn.disabled = false;
    } catch (error) {
        console.error(error);
        alert("Lỗi kết nối Server!");
    } finally {
        if (loader) loader.remove();
        if(contentArea) contentArea.style.display = 'flex';
        if(buttonsArea) buttonsArea.style.display = 'flex';
        scanningStatus.innerText = `Archive updated: ${globalResults.length} items.`;
    }
}

// --- HIỂN THỊ KẾT QUẢ ---
function renderArtGrid() {
    artGrid.innerHTML = ''; 
    const groups = {};
    
    globalResults.forEach(item => {
        if (!groups[item.style]) groups[item.style] = [];
        groups[item.style].push(item);
    });

    for (const [style, items] of Object.entries(groups)) {
        const section = `
            <div class="col-span-full mt-8 mb-4 flex items-center space-x-3 border-b border-slate-800 pb-3">
                <div class="p-1.5 bg-blue-500/10 rounded-lg text-blue-400">
                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                </div>
                <h2 class="text-xl font-black text-white uppercase tracking-tight">${style} <span class="text-[10px] text-slate-500 ml-2 font-mono px-2 py-0.5 bg-slate-800 rounded-full">${items.length}</span></h2>
            </div>`;
        artGrid.insertAdjacentHTML('beforeend', section);

        items.forEach(item => {
            // SỬA LỖI TẠI ĐÂY: Ép kiểu số và không nhân 100 nữa
            const confRaw = parseFloat(item.confidence) || 0;
            const conf = confRaw.toFixed(1);

            const pureName = item.filename.split('/').pop().split('\\').pop();
            const imgSrc = globalPreviews[pureName] || 'https://via.placeholder.com/300?text=Not+Found';
            
            const card = `
                <div class="bg-slate-800/20 border border-slate-800 rounded-2xl overflow-hidden hover:border-blue-500/50 transition-all shadow-lg group">
                    <div class="h-48 overflow-hidden bg-slate-900 flex items-center justify-center">
                        <img src="${imgSrc}" class="w-full h-full object-cover group-hover:scale-110 transition-all duration-700" 
                             onerror="this.src='https://via.placeholder.com/300?text=Load+Error'">
                    </div>
                    <div class="p-4 border-t border-slate-800/50">
                        <p class="text-[9px] font-mono text-slate-500 truncate mb-2">${pureName}</p>
                        <div class="flex justify-between items-center">
                            <span class="text-blue-400 text-[10px] font-black">${conf}%</span>
                            <div class="flex-1 h-1 bg-slate-700 mx-3 rounded-full overflow-hidden">
                                <div class="bg-gradient-to-r from-blue-600 to-blue-400 h-full" style="width: ${conf}%"></div>
                            </div>
                        </div>
                    </div>
                </div>`;
            artGrid.insertAdjacentHTML('beforeend', card);
        });
    }
}

function updateStats(lastLatency) {
    const total = globalResults.length;
    document.getElementById('stat-total').innerText = total;
    document.getElementById('stat-processed').innerText = total;
    if (lastLatency) document.getElementById('stat-latency').innerText = `${lastLatency}ms`;
}

exportBtn.onclick = async () => {
    if (globalRawFiles.length === 0) return;
    const formData = new FormData();
    globalRawFiles.forEach(file => formData.append("files", file));
    const originalText = exportBtn.innerHTML;
    exportBtn.disabled = true;
    exportBtn.innerHTML = `Archiving...`;
    try {
        const response = await fetch('/export-sorted-zip', { method: 'POST', body: formData });
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Artivault_Export.zip`;
        a.click();
    } catch (e) { alert("Export failed!"); } 
    finally { exportBtn.disabled = false; exportBtn.innerHTML = originalText; }
};

initEventListeners();