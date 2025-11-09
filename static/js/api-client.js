/**
 * 统一API客户端
 * 提供标准化的API调用接口，包含错误处理、重试机制和加载状态管理
 */

// API配置
const API_CONFIG = {
    baseURL: '/api',  // API基础路径
    timeout: 30000,   // 30秒超时
    retryCount: 3,    // 最大重试次数
    retryDelay: 1000, // 重试延迟（毫秒）
};

/**
 * 统一错误响应格式
 */
class APIError extends Error {
    constructor(message, status, data = null) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }
}

/**
 * 标准化API响应格式
 */
class APIResponse {
    constructor(success, data = null, error = null, message = null) {
        this.success = success;
        this.data = data;
        this.error = error;
        this.message = message;
    }

    static success(data, message = null) {
        return new APIResponse(true, data, null, message);
    }

    static error(error, message = null, data = null) {
        return new APIResponse(false, data, error, message);
    }
}

/**
 * 请求配置
 */
class RequestConfig {
    constructor(options = {}) {
        this.method = options.method || 'GET';
        this.headers = options.headers || {};
        this.body = options.body || null;
        this.timeout = options.timeout || API_CONFIG.timeout;
        this.retryCount = options.retryCount || API_CONFIG.retryCount;
        this.showLoading = options.showLoading !== false; // 默认显示加载状态
        this.showError = options.showError !== false;     // 默认显示错误提示
        this.abortController = options.abortController || null;
        this.loadingElement = options.loadingElement || null;
        this.loadingMessage = options.loadingMessage || '加载中...';
        this.errorContainer = options.errorContainer || 'body';
    }
}

/**
 * 加载状态管理器
 */
class LoadingManager {
    constructor() {
        this.loadingElements = new Map();
        this.loadingCount = 0;
    }

    /**
     * 显示加载状态
     */
    show(key, element, message = '加载中...') {
        if (this.loadingElements.has(key)) {
            return;
        }

        const el = typeof element === 'string' ? document.querySelector(element) : element;
        if (!el) return;

        const originalContent = el.innerHTML;
        const originalDisplay = el.style.display;

        const loadingHTML = `
            <div class="api-loading-indicator" data-loading-key="${key}">
                <div class="api-loading-spinner"></div>
                <span class="api-loading-message">${message}</span>
            </div>
        `;

        if (el.classList.contains('api-loadable')) {
            el.dataset.originalContent = originalContent;
            el.innerHTML = loadingHTML;
        } else {
            el.style.position = 'relative';
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'api-loading-overlay';
            loadingDiv.innerHTML = loadingHTML;
            el.appendChild(loadingDiv);
        }

        this.loadingElements.set(key, { element: el, originalContent, originalDisplay });
        this.loadingCount++;
    }

    /**
     * 隐藏加载状态
     */
    hide(key) {
        const loadingData = this.loadingElements.get(key);
        if (!loadingData) return;

        const { element, originalContent } = loadingData;

        const loadingIndicator = element.querySelector(`[data-loading-key="${key}"]`);
        const loadingOverlay = element.querySelector('.api-loading-overlay');

        if (loadingIndicator) loadingIndicator.remove();
        if (loadingOverlay) loadingOverlay.remove();

        if (element.classList.contains('api-loadable') && element.dataset.originalContent) {
            element.innerHTML = element.dataset.originalContent;
            delete element.dataset.originalContent;
        }

        this.loadingElements.delete(key);
        this.loadingCount--;
    }

    hideAll() {
        for (const key of this.loadingElements.keys()) {
            this.hide(key);
        }
    }
}

const loadingManager = new LoadingManager();

/**
 * 错误提示管理器
 */
class ErrorManager {
    static show(message, container = 'body', duration = 5000) {
        const containerEl = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;

        if (!containerEl) {
            console.error('错误提示容器不存在:', container);
            return;
        }

        const errorDiv = document.createElement('div');
        errorDiv.className = 'api-error-message';
        errorDiv.innerHTML = `
            <div class="api-error-content">
                <span class="api-error-icon">⚠️</span>
                <span class="api-error-text">${this.escapeHtml(message)}</span>
                <button class="api-error-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        containerEl.appendChild(errorDiv);

        if (duration > 0) {
            setTimeout(() => {
                if (errorDiv.parentElement) {
                    errorDiv.style.opacity = '0';
                    errorDiv.style.transition = 'opacity 0.3s';
                    setTimeout(() => errorDiv.remove(), 300);
                }
            }, duration);
        }

        return errorDiv;
    }

    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

/**
 * 带重试的请求函数
 */
async function fetchWithRetry(url, config) {
    let lastError = null;
    const maxRetries = config.retryCount || API_CONFIG.retryCount;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const controller = config.abortController || new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout);

            const fetchOptions = {
                method: config.method,
                headers: config.headers,
                signal: controller.signal,
            };

            if (config.body && !(config.body instanceof FormData)) {
                fetchOptions.headers['Content-Type'] = 'application/json';
            }

            if (config.body) {
                fetchOptions.body = config.body instanceof FormData 
                    ? config.body 
                    : JSON.stringify(config.body);
            }

            const response = await fetch(url, fetchOptions);
            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new APIError(
                    errorData.message || errorData.error || `HTTP ${response.status}: ${response.statusText}`,
                    response.status,
                    errorData
                );
            }

            return response;

        } catch (error) {
            lastError = error;

            if (attempt === maxRetries) break;

            const isRetryable = (
                error.name === 'AbortError' ||
                error.name === 'NetworkError' ||
                error.status >= 500 ||
                error.status === 408 ||
                error.status === 429
            );

            if (!isRetryable) break;

            const delay = config.retryDelay || API_CONFIG.retryDelay;
            await new Promise(resolve => setTimeout(resolve, delay * (attempt + 1)));
        }
    }

    throw lastError;
}

/**
 * 统一API客户端
 */
class APIClient {
    static async request(endpoint, options = {}) {
        const config = options instanceof RequestConfig ? options : new RequestConfig(options);
        // 处理URL：如果已经是完整URL或绝对路径，直接使用；否则添加baseURL前缀
        let url = endpoint;
        if (!url.startsWith('http')) {
            if (url.startsWith('/')) {
                // 绝对路径，直接使用
                url = url;
            } else {
                // 相对路径，添加baseURL前缀
                url = `${API_CONFIG.baseURL}${url.startsWith('/') ? '' : '/'}${url}`;
            }
        }
        const loadingKey = `api_${endpoint}_${Date.now()}`;

        try {
            if (config.showLoading && config.loadingElement) {
                loadingManager.show(loadingKey, config.loadingElement, config.loadingMessage);
            }

            const response = await fetchWithRetry(url, config);
            const data = await response.json();

            if (config.showLoading && config.loadingElement) {
                loadingManager.hide(loadingKey);
            }

            if (data.success === false || data.error) {
                const errorMsg = data.message || data.error || '请求失败';
                if (config.showError) {
                    ErrorManager.show(errorMsg, config.errorContainer);
                }
                return APIResponse.error(data.error || errorMsg, data.message, data);
            }

            return APIResponse.success(data.data || data, data.message);

        } catch (error) {
            if (config.showLoading && config.loadingElement) {
                loadingManager.hide(loadingKey);
            }

            let errorMessage = '请求失败';
            if (error instanceof APIError) {
                errorMessage = error.message;
            } else if (error.name === 'AbortError') {
                errorMessage = '请求超时，请检查网络连接';
            } else if (error.message) {
                errorMessage = error.message;
            }

            if (config.showError) {
                ErrorManager.show(errorMessage, config.errorContainer);
            }

            return APIResponse.error(errorMessage, errorMessage, error);
        }
    }

    static async get(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'GET' });
    }

    static async post(endpoint, body, options = {}) {
        return this.request(endpoint, { ...options, method: 'POST', body });
    }

    static async put(endpoint, body, options = {}) {
        return this.request(endpoint, { ...options, method: 'PUT', body });
    }

    static async delete(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'DELETE' });
    }

    static async upload(endpoint, formData, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'POST',
            body: formData,
            headers: {
                ...options.headers,
            },
        });
    }
}

window.APIClient = APIClient;
window.APIResponse = APIResponse;
window.APIError = APIError;
window.RequestConfig = RequestConfig;
window.ErrorManager = ErrorManager;
window.loadingManager = loadingManager;

