/**
 * Text Summarizer - Main JavaScript
 * Handles UI interactions and dynamic content
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
    
    // For pages with copy functionality
    initCopyFunctionality();
    
    // For forms with character counters
    initCharacterCounters();
    
    // For animated elements
    initAnimations();
});

/**
 * Initialize copy to clipboard functionality
 */
function initCopyFunctionality() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const contentToCopy = document.getElementById(targetId);
            
            if (contentToCopy) {
                navigator.clipboard.writeText(contentToCopy.textContent || contentToCopy.value).then(function() {
                    // Show success indicator
                    const originalText = button.textContent;
                    button.innerHTML = '<i data-feather="check"></i> Copied!';
                    if (typeof feather !== 'undefined') {
                        feather.replace();
                    }
                    
                    setTimeout(function() {
                        button.textContent = originalText;
                        if (typeof feather !== 'undefined') {
                            feather.replace();
                        }
                    }, 2000);
                }).catch(function() {
                    console.error('Failed to copy text');
                });
            }
        });
    });
}

/**
 * Initialize character counters for textarea elements
 */
function initCharacterCounters() {
    const textAreas = document.querySelectorAll('textarea[data-char-counter]');
    
    textAreas.forEach(function(textArea) {
        const counterId = textArea.getAttribute('data-char-counter');
        const counter = document.getElementById(counterId);
        
        if (counter) {
            // Update counter on load
            updateCharacterCount(textArea, counter);
            
            // Update counter on input
            textArea.addEventListener('input', function() {
                updateCharacterCount(this, counter);
            });
        }
    });
}

/**
 * Update character count display
 */
function updateCharacterCount(textArea, counter) {
    const currentLength = textArea.value.length;
    counter.textContent = currentLength;
    
    // Add visual indicator if approaching limit
    const maxLength = textArea.getAttribute('maxlength');
    if (maxLength) {
        const percentage = (currentLength / maxLength) * 100;
        if (percentage > 90) {
            counter.classList.add('text-danger');
        } else if (percentage > 75) {
            counter.classList.add('text-warning');
            counter.classList.remove('text-danger');
        } else {
            counter.classList.remove('text-warning', 'text-danger');
        }
    }
}

/**
 * Initialize animations for elements
 */
function initAnimations() {
    // Animate progress bars
    const progressBars = document.querySelectorAll('.animate-progress');
    progressBars.forEach(function(bar) {
        bar.classList.add('animate-width');
    });
    
    // Add scroll reveal animations if needed in the future
}

/**
 * Truncate text to a specified length
 * @param {string} text - Text to truncate
 * @param {number} length - Maximum length
 * @returns {string} - Truncated text
 */
function truncateText(text, length = 100) {
    if (!text || text.length <= length) {
        return text;
    }
    return text.substring(0, length) + '...';
}