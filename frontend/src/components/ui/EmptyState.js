import React from 'react';
import './EmptyState.css';

function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="empty-state card">
      {Icon && (
        <div className="empty-state__icon">
          <Icon />
        </div>
      )}
      <h3 className="empty-state__title">{title}</h3>
      {description && <p className="empty-state__description">{description}</p>}
      {action && <div className="empty-state__action">{action}</div>}
    </div>
  );
}

export default EmptyState;
