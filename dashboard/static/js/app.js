/* Shared shell behavior: theme, sidebar, toast, modal, formatting, fetch helper. */

function toggleTheme() {
    const html = document.documentElement;
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    document.dispatchEvent(new CustomEvent('theme-changed', { detail: next }));
}
(function () {
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
})();

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (window.innerWidth <= 768) sidebar.classList.toggle('mobile-open');
    else sidebar.classList.toggle('collapsed');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
    toast.innerHTML = `<span style="font-size:1.2rem">${icons[type] || icons.info}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function showModal(title, message) {
    const overlay = document.getElementById('modalOverlay');
    if (!overlay) return;
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalMessage').textContent = message;
    overlay.classList.add('active');
}
function closeModal() {
    const overlay = document.getElementById('modalOverlay');
    if (overlay) overlay.classList.remove('active');
}

function fmtCurrency(v) {
    if (v === null || v === undefined || isNaN(v)) return '$0';
    const abs = Math.abs(v);
    if (abs >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
    if (abs >= 1e6) return '$' + (v / 1e6).toFixed(2) + 'M';
    if (abs >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'K';
    return '$' + v.toFixed(0);
}
function fmtSignedCurrency(v) { return (v >= 0 ? '+' : '') + fmtCurrency(v); }
function fmtNumber(v) { return Number(v).toLocaleString(); }

async function fetchJSON(url, options) {
    const r = await fetch(url, options);
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
}

function getChartColors() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        primary: '#2563EB',
        primaryLight: isDark ? '#60a5fa' : '#3b82f6',
        emerald: isDark ? '#34d399' : '#10B981',
        orange: isDark ? '#fbbf24' : '#F59E0B',
        red: isDark ? '#f87171' : '#ef4444',
        purple: '#8b5cf6',
        grid: isDark ? 'rgba(148, 163, 184, 0.1)' : 'rgba(15, 23, 42, 0.06)',
        text: isDark ? '#94a3b8' : '#475569',
        bg: isDark ? '#131b2e' : '#ffffff',
    };
}

const chartInstances = {};
function createChart(id, config) {
    const canvas = document.getElementById(id);
    if (!canvas || typeof Chart === 'undefined') return null;
    if (chartInstances[id]) chartInstances[id].destroy();
    const c = getChartColors();
    Chart.defaults.color = c.text;
    Chart.defaults.borderColor = c.grid;
    Chart.defaults.font.family = "'Inter', sans-serif";
    chartInstances[id] = new Chart(canvas.getContext('2d'), config);
    return chartInstances[id];
}

document.addEventListener('theme-changed', () => {
    if (typeof window.rerenderCharts === 'function') window.rerenderCharts();
});
window.addEventListener('resize', () => {
    Object.values(chartInstances).forEach((c) => c && c.resize());
});

// Mobile sidebar closes after navigating
document.addEventListener('DOMContentLoaded', () => {
    if (window.innerWidth <= 768) {
        document.querySelectorAll('.nav-item').forEach((a) =>
            a.addEventListener('click', () => document.getElementById('sidebar')?.classList.remove('mobile-open'))
        );
    }
});
