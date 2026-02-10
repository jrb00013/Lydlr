import React, { useState, useEffect, useContext } from 'react';
import './ModelsView.css';
import { NotificationContext } from '../App';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ModelsView() {
  const notification = useContext(NotificationContext);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_URL}/api/models`);
      const data = await response.json();
      setModels(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch models:', error);
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/models/upload`, {
        method: 'POST',
        body: formData
      });
      if (response.ok) {
        notification.showSuccess('Model uploaded successfully!');
        fetchModels();
      } else {
        const error = await response.json();
        notification.showError(`Failed to upload model: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to upload model:', error);
      notification.showError('Failed to upload model');
    } finally {
      setUploading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading models...</div>;
  }

  return (
    <div className="models-view">
      <div className="models-header">
        <h1 className="page-title">Models</h1>
        <div className="upload-section">
          <label htmlFor="file-upload" className="btn btn-primary">
            {uploading ? 'Uploading...' : 'Upload Model'}
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".pth"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
            disabled={uploading}
          />
        </div>
      </div>

      <div className="models-grid grid grid-3">
        {models.length === 0 ? (
          <div className="card">
            <p>No models available. Upload a model to get started.</p>
          </div>
        ) : (
          models.map((model, idx) => (
            <div key={idx} className="model-card card">
              <h3>{model.version}</h3>
              
              {/* Use Case Section */}
              <div className="model-section">
                <h4 className="section-title">Use Case</h4>
                <p className="section-description">
                  Multimodal sensor data compression for autonomous systems. This model compresses 
                  camera images, LiDAR point clouds, IMU data, and audio spectrograms to reduce 
                  bandwidth and storage requirements while maintaining reconstruction quality.
                </p>
              </div>

              {/* Basic Info */}
              <div className="model-info">
                <div className="info-row">
                  <span className="info-label">Size:</span>
                  <span className="info-value">{model.size_mb.toFixed(2)} MB</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Created:</span>
                  <span className="info-value">
                    {new Date(model.created_at).toLocaleDateString()}
                  </span>
                </div>
                <div className="info-row">
                  <span className="info-label">File:</span>
                  <span className="info-value filename">{model.filename}</span>
                </div>
              </div>

              {/* Training Data Section */}
              <div className="model-section">
                <h4 className="section-title">Training Data</h4>
                <div className="training-data-info">
                  {model.metadata?.training_data_source ? (
                    <>
                      <div className="metadata-item">
                        <span>Data Source:</span>
                        <span>{model.metadata.training_data_source}</span>
                      </div>
                      {model.metadata?.dataset_size && (
                        <div className="metadata-item">
                          <span>Dataset Size:</span>
                          <span>{model.metadata.dataset_size.toLocaleString()} samples</span>
                        </div>
                      )}
                      {model.metadata?.training_sequences && (
                        <div className="metadata-item">
                          <span>Sequences:</span>
                          <span>{model.metadata.training_sequences}</span>
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="section-description">
                      Trained on multimodal sensor data including:
                      <ul className="training-list">
                        <li><strong>Camera:</strong> RGB images (480×640, 3 channels)</li>
                        <li><strong>LiDAR:</strong> Point clouds (1024 points, 3D coordinates)</li>
                        <li><strong>IMU:</strong> 6-axis inertial data (acceleration + gyroscope)</li>
                        <li><strong>Audio:</strong> Spectrogram data (128×128 flattened)</li>
                      </ul>
                      {model.metadata?.use_synthetic_data !== undefined && (
                        <span className="data-type-badge">
                          {model.metadata.use_synthetic_data ? 'Synthetic Data' : 'Real Sensor Data'}
                        </span>
                      )}
                    </p>
                  )}
                  {model.metadata?.training_epochs && (
                    <div className="metadata-item">
                      <span>Training Epochs:</span>
                      <span>{model.metadata.training_epochs}</span>
                    </div>
                  )}
                  {model.metadata?.batch_size && (
                    <div className="metadata-item">
                      <span>Batch Size:</span>
                      <span>{model.metadata.batch_size}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Model Architecture & Technical Details */}
              {(model.metadata?.architecture || model.metadata?.input_shape || model.metadata?.output_shape) && (
                <div className="model-section">
                  <h4 className="section-title">Model Architecture</h4>
                  <div className="model-details">
                    {model.metadata.architecture && (
                      <div className="metadata-item">
                        <span>Architecture:</span>
                        <span>{model.metadata.architecture}</span>
                      </div>
                    )}
                    {model.metadata.input_shape && (
                      <div className="metadata-item">
                        <span>Input Shape:</span>
                        <span className="shape-value">{JSON.stringify(model.metadata.input_shape)}</span>
                      </div>
                    )}
                    {model.metadata.output_shape && (
                      <div className="metadata-item">
                        <span>Output Shape:</span>
                        <span className="shape-value">{JSON.stringify(model.metadata.output_shape)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Performance Metrics */}
              {model.metadata && Object.keys(model.metadata).length > 0 && (
                <div className="model-section">
                  <h4 className="section-title">Performance Metrics</h4>
                  <div className="model-metadata">
                    {model.metadata.final_compression_ratio && (
                      <div className="metadata-item">
                        <span>Compression Ratio:</span>
                        <span>{model.metadata.final_compression_ratio.toFixed(2)}x</span>
                      </div>
                    )}
                    {model.metadata.final_quality && (
                      <div className="metadata-item">
                        <span>Reconstruction Quality:</span>
                        <span>{(model.metadata.final_quality * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    {model.metadata?.psnr && (
                      <div className="metadata-item">
                        <span>PSNR:</span>
                        <span>{model.metadata.psnr.toFixed(2)} dB</span>
                      </div>
                    )}
                    {model.metadata?.ssim && (
                      <div className="metadata-item">
                        <span>SSIM:</span>
                        <span>{model.metadata.ssim.toFixed(3)}</span>
                      </div>
                    )}
                    {model.metadata?.inference_time_ms && (
                      <div className="metadata-item">
                        <span>Avg Inference Time:</span>
                        <span>{model.metadata.inference_time_ms.toFixed(1)} ms</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Additional Metadata */}
              {model.metadata && Object.keys(model.metadata).length > 0 && (
                <div className="model-section">
                  <h4 className="section-title">Additional Information</h4>
                  <div className="additional-info">
                    {model.metadata?.learning_rate && (
                      <div className="metadata-item">
                        <span>Learning Rate:</span>
                        <span>{model.metadata.learning_rate}</span>
                      </div>
                    )}
                    {model.metadata?.optimizer && (
                      <div className="metadata-item">
                        <span>Optimizer:</span>
                        <span>{model.metadata.optimizer}</span>
                      </div>
                    )}
                    {model.metadata?.loss_function && (
                      <div className="metadata-item">
                        <span>Loss Function:</span>
                        <span>{model.metadata.loss_function}</span>
                      </div>
                    )}
                    {model.metadata?.training_date && (
                      <div className="metadata-item">
                        <span>Training Date:</span>
                        <span>{new Date(model.metadata.training_date).toLocaleDateString()}</span>
                      </div>
                    )}
                    {model.metadata?.description && (
                      <p className="section-description">{model.metadata.description}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default ModelsView;

