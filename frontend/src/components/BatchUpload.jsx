import React, { useState, useCallback } from "react";
import api from "../api";

const BatchUpload = ({ onResults }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
  };

  const addFiles = (newFiles) => {
    const validFiles = newFiles.filter(f => f.type.startsWith('image/'));
    const fileObjects = validFiles.map(file => ({
      file,
      id: Math.random().toString(36).substring(7),
      name: file.name,
      status: 'pending', // pending, processing, done, error
      result: null
    }));
    
    setFiles(prev => [...prev, ...fileObjects]);
  };

  const processBatch = async () => {
    if (files.length === 0 || isProcessing) return;
    
    setIsProcessing(true);
    setProgress(0);
    
    const results = [];
    let completedCount = 0;
    
    const updatedFiles = [...files];
    
    for (let i = 0; i < updatedFiles.length; i++) {
      if (updatedFiles[i].status === 'done' || updatedFiles[i].status === 'error') {
        completedCount++;
        continue;
      }
      
      updatedFiles[i].status = 'processing';
      setFiles([...updatedFiles]);
      
      try {
        const formData = new FormData();
        formData.append("file", updatedFiles[i].file, updatedFiles[i].name);
        
        const res = await api.post("/predict", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        
        updatedFiles[i].status = 'done';
        updatedFiles[i].result = res.data;
        results.push(res.data);
      } catch (err) {
        console.error("Error processing file", updatedFiles[i].name, err);
        updatedFiles[i].status = 'error';
      }
      
      completedCount++;
      setProgress(completedCount / updatedFiles.length);
      setFiles([...updatedFiles]);
    }
    
    setIsProcessing(false);
    if (onResults && results.length > 0) {
      onResults(results);
    }
  };

  const removeFile = (id) => {
    if (isProcessing) return;
    setFiles(files.filter(f => f.id !== id));
  };

  return (
    <div className="batch-upload-container">
      <div 
        className={`batch-zone ${isDragging ? "batch-zone-active" : ""}`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input 
          type="file" 
          multiple 
          accept="image/*" 
          onChange={handleFileSelect} 
          hidden 
          id="batch-file-input"
          disabled={isProcessing}
        />
        <label htmlFor="batch-file-input" className="batch-zone-label">
          <span className="upload-icon" aria-hidden="true">⇪</span>
          <span>Drag multiple images here, or click to browse</span>
        </label>
      </div>

      {files.length > 0 && (
        <div className="batch-files-container">
          <div className="batch-header">
            <h3>Files ({files.length})</h3>
            <button 
              className="btn btn-primary" 
              onClick={processBatch}
              disabled={isProcessing || files.every(f => f.status === 'done' || f.status === 'error')}
            >
              {isProcessing ? "Processing..." : "Process Batch"}
            </button>
          </div>
          
          {isProcessing && (
            <div className="batch-progress-container">
              <div className="batch-progress" style={{ width: `${progress * 100}%` }}></div>
            </div>
          )}

          <div className="batch-file-list" role="list">
            {files.map(file => (
              <div key={file.id} className="batch-file-row" role="listitem">
                <span className="file-name" title={file.name}>{file.name}</span>
                <div className="file-status-wrap">
                  {file.status === 'pending' && <span className="status-badge status-pending">Pending</span>}
                  {file.status === 'processing' && <span className="status-badge status-processing">Processing...</span>}
                  {file.status === 'done' && <span className="status-badge status-done">Done</span>}
                  {file.status === 'error' && <span className="status-badge status-error">Error</span>}
                  
                  {!isProcessing && (file.status === 'pending' || file.status === 'error') && (
                    <button className="remove-btn" onClick={() => removeFile(file.id)} aria-label="Remove file">×</button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchUpload;
