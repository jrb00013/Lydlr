import React from 'react';
import { Link } from 'react-router-dom';
import ExploreOffIcon from '@mui/icons-material/ExploreOff';
import './NotFound.css';

function NotFound() {
  return (
    <div className="not-found page-enter">
      <div className="not-found__mark" aria-hidden="true">
        <ExploreOffIcon />
      </div>
      <p className="not-found__code">404</p>
      <h1 className="not-found__title">Signal not found</h1>
      <p className="not-found__copy">
        That route is outside the control plane. Head back to the dashboard or pick a
        section from the sidebar.
      </p>
      <Link to="/" className="not-found__cta">
        Return to dashboard
      </Link>
    </div>
  );
}

export default NotFound;
