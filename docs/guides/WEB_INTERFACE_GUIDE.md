#  Lydlr Web Interface Guide

## Modern Full-Stack Interface for Real-Time Edge Compression

---

##  Overview

The Lydlr Web Interface provides a complete, modern dashboard for managing your revolutionary edge compression system in real-time. Built with React and FastAPI, it offers:

- **Real-time monitoring** with WebSocket updates
- **Interactive visualizations** with charts and graphs
- **Node management** with start/stop/restart controls
- **Model deployment** with drag-and-drop simplicity
- **Performance analytics** with historical data

---

##  Quick Start

### Option 1: Docker (Recommended)

```bash
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr

# Start everything
./start-lydlr.sh --build

# Or in detached mode
./start-lydlr.sh --build -d
```

Then open: **http://localhost**

### Option 2: Manual Launch

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm start
```

**Terminal 3 - MongoDB & Redis:**
```bash
docker-compose up mongodb redis
```

---

##  Dashboard Features

### 1. **Overview Dashboard**

**Real-time metrics at a glance:**
- Active nodes count
- Average compression ratio
- Average latency
- Average quality score

**Live charts:**
- Compression ratio over time
- Latency monitoring
- Quality score tracking

**WebSocket updates:** Metrics refresh automatically as data comes in.

---

### 2. **Nodes View** 

**Manage your edge nodes:**

- **List all registered nodes** with status indicators
- **View per-node metrics:**
  - Current model version
  - Compression ratio
  - Latency
  - Quality score
  - Bandwidth estimate

- **Control actions:**
  -  Start node
  -  Restart node
  -  Stop node

**Status indicators:**
-  Active
-  Inactive
-  Error

---

### 3. **Models View** 

**Manage your ML models:**

- **Browse available models** with metadata
- **Upload new models** (.pth files)
- **View model details:**
  - Version number
  - File size
  - Creation date
  - Training metadata (compression ratio, quality)

**Upload process:**
1. Click "Upload Model"
2. Select `.pth` file
3. Model automatically indexed and available for deployment

---

### 4. **Metrics View** 

**Analyze performance over time:**

- **Interactive charts** for:
  - Compression ratio trends
  - Latency patterns
  - Quality score history
  - Bandwidth estimates

- **Filter by node** or view aggregate data
- **Historical data** up to 50 recent measurements
- **Export capabilities** (coming soon)

**Chart types:**
- Line charts for trends
- Bar charts for comparisons
- Real-time updates

---

### 5. **Deployment View** 

**Deploy models to edge nodes:**

**Deployment wizard:**
1. **Select model** from dropdown
2. **Choose target nodes** (multi-select)
3. **Click Deploy** to initiate

**Deployment history:**
- View past deployments
- Track deployment status
- See which models are on which nodes
- Deployment timestamps

**Status tracking:**
-  Deploying
-  Deployed
-  Failed

---

##  UI/UX Features

### Modern Design
- **Gradient backgrounds** with purple/blue theme
- **Glassmorphism** effects for cards
- **Smooth animations** and transitions
- **Responsive layout** for all screen sizes

### Interactive Elements
- **Hover effects** on all interactive elements
- **Loading states** for async operations
- **Toast notifications** for actions
- **Real-time status indicators**

### Accessibility
- **High contrast** text and backgrounds
- **Large touch targets** for mobile
- **Keyboard navigation** support
- **Screen reader friendly**

---

##  API Integration

### Backend Endpoints Used

**Nodes:**
- `GET /api/nodes` - Fetch all nodes
- `POST /api/nodes/{id}/start` - Start node
- `POST /api/nodes/{id}/stop` - Stop node
- `POST /api/nodes/{id}/restart` - Restart node

**Models:**
- `GET /api/models` - List models
- `POST /api/models/upload` - Upload model

**Metrics:**
- `GET /api/metrics?node_id={id}&limit=50` - Get metrics
- `POST /api/metrics` - Store metrics

**Deployments:**
- `POST /api/deploy` - Deploy model
- `GET /api/deployments` - Deployment history

**System:**
- `GET /api/stats` - System statistics
- `GET /health` - Health check

### WebSocket Connection

**Real-time updates via WebSocket:**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/metrics');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'metrics_update') {
    updateChart(data.data);
  }
  
  if (data.type === 'node_status_change') {
    updateNodeStatus(data.node_id, data.status);
  }
};
```

---

##  Customization

### Changing Theme Colors

Edit `frontend/src/App.css`:

```css
/* Primary gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Change to your colors */
background: linear-gradient(135deg, #your-color1 0%, #your-color2 100%);
```

### Adding New Views

1. Create component in `frontend/src/components/`
2. Add route in `frontend/src/App.js`
3. Add nav link in navbar

Example:
```jsx
// NewView.js
export default function NewView() {
  return <div>New View</div>;
}

// App.js
<Route path="/new" element={<NewView />} />

// Navbar
<Link to="/new">New</Link>
```

### Custom API Endpoints

Add to `backend/main.py`:

```python
@app.get("/api/custom")
async def custom_endpoint():
    return {"message": "Custom data"}
```

---

##  Mobile Support

The interface is **fully responsive**:

- **Desktop** (1440px+): Full layout with sidebars
- **Tablet** (768px-1440px): Adapted grid layouts
- **Mobile** (< 768px): Stacked cards, touch-optimized

**Touch gestures:**
- Swipe to navigate charts
- Pull to refresh data
- Touch to expand cards

---

##  Development

### Hot Reloading

**Frontend:**
- Edit files in `frontend/src/`
- Browser auto-refreshes

**Backend:**
- Edit files in `backend/`
- Uvicorn auto-reloads

### Adding Dependencies

**Frontend:**
```bash
docker-compose exec frontend npm install package-name
```

**Backend:**
```bash
docker-compose exec backend pip install package-name
# Add to backend/requirements.txt
```

### Debugging

**Frontend console:**
- Open browser DevTools (F12)
- Check Console tab for errors
- Network tab for API calls

**Backend logs:**
```bash
docker-compose logs -f backend
```

---

##  Common Issues

### "Backend not connecting"

**Check:**
1. Backend is running: `docker-compose ps`
2. Health endpoint: `curl http://localhost:8000/health`
3. CORS settings in `backend/main.py`

### "Models not showing"

**Check:**
1. Models directory: `ls ros2/src/lydlr_ai/models/`
2. File permissions
3. Backend logs for errors

### "Charts not updating"

**Check:**
1. WebSocket connection in browser console
2. Redis is running: `docker-compose ps redis`
3. Metrics are being published

---

##  Best Practices

### For Production

1. **Build frontend for production:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Use environment variables:**
   - Never hardcode API URLs
   - Use `.env` files

3. **Enable HTTPS:**
   - Use SSL certificates
   - Update nginx configuration

4. **Implement authentication:**
   - JWT tokens
   - User sessions
   - Role-based access

5. **Rate limiting:**
   - Protect API endpoints
   - Prevent abuse

### For Development

1. **Use React DevTools** for debugging
2. **Check API docs** at `/docs`
3. **Monitor logs** continuously
4. **Test on multiple browsers**

---

##  Performance Tips

### Frontend Optimization
- Use React.memo for expensive components
- Implement virtual scrolling for large lists
- Lazy load routes with React.lazy()
- Optimize images and assets

### Backend Optimization
- Add database indexes
- Use Redis caching
- Implement pagination
- Compress responses with gzip

---

##  Learning Resources

- **React:** https://react.dev/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Recharts:** https://recharts.org/
- **Material-UI:** https://mui.com/
- **WebSockets:** https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API

---

##  Contributing

Want to add features or fix bugs?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

##  What's Next?

**Planned features:**
- [ ] User authentication and authorization
- [ ] Export metrics to CSV/JSON
- [ ] Advanced filtering and search
- [ ] Custom dashboard layouts
- [ ] Dark mode toggle
- [ ] Mobile app (React Native)
- [ ] Email/SMS alerts
- [ ] Multi-language support

---

**Enjoy managing your revolutionary compression system with style! **

