import React from 'react';
import './PageHeader.css';

function PageHeader({ title, subtitle, icon: Icon, actions, badge }) {
  return (
    <header className="page-header-block">
      <div className="page-header-block__main">
        {Icon && (
          <div className="page-header-block__icon">
            <Icon />
          </div>
        )}
        <div>
          <div className="page-header-block__title-row">
            <h1 className="page-header-block__title">{title}</h1>
            {badge}
          </div>
          {subtitle && <p className="page-header-block__subtitle">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="page-header-block__actions">{actions}</div>}
    </header>
  );
}

export default PageHeader;
