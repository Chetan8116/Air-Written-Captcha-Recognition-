// This script will be added to the feature.html template to dynamically update the Current Mode display
function updateCurrentModeDisplay(mode) {
    console.log('updateCurrentModeDisplay called with mode:', mode);
    
    const currentModeDisplay = document.querySelector('.alert-info .badge');
    if (!currentModeDisplay) {
        console.error('Current mode display element not found');
        return;
    }
    
    // Remove all existing classes
    currentModeDisplay.className = 'badge';
    
    // Check for composite modes (specialized-parent format)
    const isCompositeMode = typeof mode === 'string' && mode.includes('-');
    let specializedMode, parentMode;
    
    if (isCompositeMode) {
        [specializedMode, parentMode] = mode.split('-');
        mode = specializedMode; // Use specialized mode for styling
    }
    
    // Set the appropriate classes and text based on mode
    if (mode === 'num') {
        currentModeDisplay.classList.add('bg-warning');
        currentModeDisplay.textContent = 'Numbers (0-9)';
    } else if (mode === 'alpha') {
        currentModeDisplay.classList.add('bg-success');
        currentModeDisplay.textContent = 'Alphabets (a-z)';
    } else if (mode === 'alphanum') {
        currentModeDisplay.classList.add('bg-info');
        currentModeDisplay.textContent = 'AlphaNumeric';
    } else if (mode === 'alphanum-uppercase') {
        currentModeDisplay.classList.add('bg-info');
        currentModeDisplay.textContent = 'Alphanumeric Uppercase (A-Z, 0-9)';
    } else if (mode === 'alphanum-lowercase') {
        currentModeDisplay.classList.add('bg-info');
        currentModeDisplay.textContent = 'Alphanumeric Lowercase (a-z, 0-9)';
    } else if (mode === 'alphanum-auto') {
        currentModeDisplay.classList.add('bg-info');
        currentModeDisplay.textContent = 'Alphanumeric Combined (A-Z, a-z, 0-9)';
    } else if (mode === 'uppercase') {
        currentModeDisplay.classList.add('bg-primary');
        // Add parent mode context if in a composite mode
        if (isCompositeMode && parentMode === 'alphanum') {
            currentModeDisplay.textContent = 'Alphanumeric Uppercase (A-Z, 0-9)';
        } else {
            currentModeDisplay.textContent = 'Alphabets Uppercase (A-Z)';
        }
    } else if (mode === 'lowercase') {
        currentModeDisplay.classList.add('bg-primary');
        // Add parent mode context if in a composite mode
        if (isCompositeMode && parentMode === 'alphanum') {
            currentModeDisplay.textContent = 'Alphanumeric Lowercase (a-z, 0-9)';
        } else {
            currentModeDisplay.textContent = 'Alphabets Lowercase (a-z)';
        }
    } else if (mode === 'auto_alphabet') {
        currentModeDisplay.classList.add('bg-primary');
        // Add parent mode context if in a composite mode
        if (isCompositeMode && parentMode === 'alphanum') {
            currentModeDisplay.textContent = 'Alphanumeric Combined (A-Z, a-z, 0-9)';
        } else {
            currentModeDisplay.textContent = 'Alphabets Combined (A-Z, a-z)';
        }
    } else if (mode === 'alphabet') {
        // Handle the case where server returns 'alphabet' mode
        currentModeDisplay.classList.add('bg-primary');
        currentModeDisplay.textContent = 'Alphabets (A-Z, a-z)';
    } else {
        currentModeDisplay.classList.add('bg-secondary');
        currentModeDisplay.textContent = 'Off - Select mode first';
    }
    
    console.log('Updated current mode display to:', currentModeDisplay.textContent);
}