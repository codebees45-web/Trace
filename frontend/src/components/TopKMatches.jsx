import React, { useEffect, useState } from "react";

const TopKMatches = ({ matches }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Trigger animation after mount
    const timer = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(timer);
  }, []);

  if (!matches || matches.length === 0) return null;

  return (
    <div className="topk-list" role="list" aria-label="Top identity matches">
      {matches.map((match, index) => {
        const isTopMatch = index === 0;
        const confidencePct = (match.confidence * 100).toFixed(1);
        const delay = index * 150; // staggered delay
        
        return (
          <div 
            key={index} 
            className={`topk-item ${isTopMatch ? "topk-top-match" : ""}`}
            role="listitem"
          >
            <div className="topk-info">
              <span className="topk-rank">#{index + 1}</span>
              <span className="topk-name">{match.identity}</span>
              <span className="topk-pct">{confidencePct}%</span>
            </div>
            <div className="topk-bar">
              <div 
                className="topk-fill"
                style={{ 
                  width: mounted ? `${match.confidence * 100}%` : "0%",
                  transitionDelay: `${delay}ms`
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default TopKMatches;
