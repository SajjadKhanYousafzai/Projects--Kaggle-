// Resume Classifier JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips and popovers
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize animations
    initializeAnimations();
    
    // Initialize form enhancements
    initializeFormEnhancements();
    
    // Initialize scroll animations
    initializeScrollAnimations();
});

// Animation Functions
function initializeAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in-up');
    });
    
    // Floating animation for hero elements
    const floatingElements = document.querySelectorAll('.floating-card');
    floatingElements.forEach(element => {
        element.style.animation = 'float 6s ease-in-out infinite';
    });
}

// Form Enhancement Functions
function initializeFormEnhancements() {
    // Auto-resize textareas
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', autoResize);
        autoResize.call(textarea); // Initial call
    });
    
    // Add input validation feedback
    const inputs = document.querySelectorAll('.form-control');
    inputs.forEach(input => {
        input.addEventListener('blur', validateInput);
        input.addEventListener('input', clearValidation);
    });
}

// Auto-resize textarea
function autoResize() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
}

// Input validation
function validateInput() {
    const input = this;
    const value = input.value.trim();
    
    input.classList.remove('is-valid', 'is-invalid');
    
    if (input.hasAttribute('required') && value === '') {
        input.classList.add('is-invalid');
        showFieldError(input, 'This field is required');
    } else if (value !== '') {
        input.classList.add('is-valid');
        clearFieldError(input);
    }
}

// Clear validation
function clearValidation() {
    this.classList.remove('is-valid', 'is-invalid');
    clearFieldError(this);
}

// Show field error
function showFieldError(input, message) {
    clearFieldError(input);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    input.parentNode.appendChild(errorDiv);
}

// Clear field error
function clearFieldError(input) {
    const errorDiv = input.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Scroll Animations
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe animated elements
    const animatedElements = document.querySelectorAll('.stat-card, .feature-card, .metric-card');
    animatedElements.forEach(element => {
        observer.observe(element);
    });
}

// Utility Functions
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    
    // Initialize and show toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    
    toast.show();
    
    // Remove toast from DOM after hiding
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

// Copy to clipboard function
function copyToClipboard(text, successMessage = 'Copied to clipboard!') {
    if (navigator.clipboard && window.isSecureContext) {
        // Use modern clipboard API
        navigator.clipboard.writeText(text).then(function() {
            showToast(successMessage, 'success');
        }).catch(function(err) {
            showToast('Failed to copy to clipboard', 'danger');
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showToast(successMessage, 'success');
        } catch (err) {
            showToast('Failed to copy to clipboard', 'danger');
        }
        
        document.body.removeChild(textArea);
    }
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Animate number counting
function animateNumber(element, target, duration = 1000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(function() {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

// Smooth scroll to element
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Loading state management
function setLoadingState(button, loading = true) {
    const btnText = button.querySelector('.btn-text');
    const btnLoading = button.querySelector('.btn-loading');
    
    if (loading) {
        if (btnText) btnText.classList.add('d-none');
        if (btnLoading) btnLoading.classList.remove('d-none');
        button.disabled = true;
    } else {
        if (btnText) btnText.classList.remove('d-none');
        if (btnLoading) btnLoading.classList.add('d-none');
        button.disabled = false;
    }
}

// Form submission with loading state
function handleFormSubmission(formId, loadingText = 'Processing...') {
    const form = document.getElementById(formId);
    const submitBtn = form.querySelector('button[type="submit"]');
    
    if (form && submitBtn) {
        form.addEventListener('submit', function(e) {
            // Add loading spinner HTML if not present
            if (!submitBtn.querySelector('.btn-loading')) {
                const btnText = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <span class="btn-text">${btnText}</span>
                    <span class="btn-loading d-none">
                        <span class="spinner-border spinner-border-sm me-2"></span>
                        ${loadingText}
                    </span>
                `;
            }
            
            setLoadingState(submitBtn, true);
        });
    }
}

// Initialize form submission handlers
document.addEventListener('DOMContentLoaded', function() {
    handleFormSubmission('resumeForm', 'Analyzing...');
});

// API call wrapper
async function makeAPICall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showToast('An error occurred while processing your request', 'danger');
        throw error;
    }
}

// Feature analysis helpers
function analyzeTextComplexity(text) {
    const words = text.split(/\s+/).filter(word => word.length > 0);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const avgWordsPerSentence = words.length / sentences.length || 0;
    const avgCharsPerWord = text.replace(/\s/g, '').length / words.length || 0;
    
    return {
        words: words.length,
        sentences: sentences.length,
        avgWordsPerSentence: avgWordsPerSentence.toFixed(1),
        avgCharsPerWord: avgCharsPerWord.toFixed(1),
        complexity: avgWordsPerSentence > 15 ? 'High' : avgWordsPerSentence > 10 ? 'Medium' : 'Low'
    };
}

// Export functions for global use
window.ResumeClassifier = {
    showToast,
    copyToClipboard,
    formatNumber,
    animateNumber,
    smoothScrollTo,
    setLoadingState,
    makeAPICall,
    analyzeTextComplexity
};
