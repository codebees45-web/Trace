import React, { useEffect, useRef, useState } from "react";

// Helper component for counting up
const AnimatedCounter = ({ value, isPercentage, duration = 1500, visible }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!visible) return;
    
    let startTime = null;
    let animationFrame;
    const targetValue = value;

    const updateCounter = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = timestamp - startTime;
      const percentage = Math.min(progress / duration, 1);
      
      // Easing out function
      const easeOut = 1 - Math.pow(1 - percentage, 3);
      setCount(easeOut * targetValue);

      if (percentage < 1) {
        animationFrame = requestAnimationFrame(updateCounter);
      }
    };

    animationFrame = requestAnimationFrame(updateCounter);
    return () => cancelAnimationFrame(animationFrame);
  }, [value, duration, visible]);

  if (isPercentage) {
    return <span>{(count * 100).toFixed(1)}%</span>;
  }
  return <span>{Math.floor(count)}</span>;
};

const ModelStats = ({ stats }) => {
  const containerRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  if (!stats) return null;

  return (
    <div className="model-stats" ref={containerRef} aria-label="Model Statistics">
      <div className="model-stats-grid">
        <div className="stat-card">
          <div className="stat-value">
            <AnimatedCounter value={stats.num_identities} visible={isVisible} />
          </div>
          <div className="stat-label">Identities</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            <AnimatedCounter value={stats.num_images} visible={isVisible} />
          </div>
          <div className="stat-label">Training Images</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            <AnimatedCounter value={stats.cv_accuracy} isPercentage={true} visible={isVisible} />
          </div>
          <div className="stat-label">CV Accuracy</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            <AnimatedCounter value={stats.feature_dim} visible={isVisible} />
          </div>
          <div className="stat-label">Feature Dimensions</div>
        </div>
      </div>
      <div className="model-stats-note">
        <p className="cnn-backbone-badge" style={{ color: '#00e5ff', fontWeight: 'bold', fontSize: '1.05rem', marginBottom: '8px' }}>
          ⚡ Core Architecture: Deep Convolutional Neural Network (CNN) Backbone (512-d embeddings)
        </p>
        <p><strong>Training Date:</strong> {stats.training_date ? new Date(stats.training_date).toLocaleDateString() : "Just now"}</p>
        <p><strong>Feature Mode:</strong> {stats.feature_mode || "Deep CNN Backbone (512-d)"}</p>
        <p><strong>Neural Engine:</strong> PyTorch MaskedFaceCNN / ResNet Residual Architecture</p>
      </div>
    </div>
  );
};

export default ModelStats;
