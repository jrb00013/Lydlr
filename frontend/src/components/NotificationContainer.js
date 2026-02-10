import React from 'react';
import Notification from './Notification';
import './Notification.css';

function NotificationContainer({ notifications, removeNotification }) {
  return (
    <div className="notification-container">
      {notifications.map((notification) => (
        <Notification
          key={notification.id}
          message={notification.message}
          type={notification.type}
          onClose={() => removeNotification(notification.id)}
          duration={notification.duration}
        />
      ))}
    </div>
  );
}

export default NotificationContainer;

