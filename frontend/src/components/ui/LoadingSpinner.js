import React from 'react';
import './LoadingSpinner.css';

function LoadingSpinner({ message = 'Loading...', size = 'md' }) {
  return (
    <div className={`loading-spinner loading-spinner--${size}`} role="status" aria-live="polite">
      <div className="loading-spinner__ring" aria-hidden="true" />
      {message && <p className="loading-spinner__text">{message}</p>}
    </div>
  );
}

export default LoadingSpinner;
