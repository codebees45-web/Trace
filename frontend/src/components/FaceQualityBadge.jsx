import { useState } from "react";

const FaceQualityBadge = ({ quality }) => {
  const [isHovered, setIsHovered] = useState(false);

  if (!quality) return null;

  const { overall, blur_score, brightness_score, resolution_ok } = quality;

  const badgeClass = "quality-badge " + (overall === "good" ? "quality-good" : overall === "fair" ? "quality-fair" : "quality-poor");
  const icon = overall === "good" ? "🟢" : overall === "fair" ? "🟡" : "🔴";

  return (
    <div 
      className={badgeClass}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="status"
      aria-label={`Face quality is ${overall}`}
    >
      <span className="quality-icon">{icon}</span>
      <span className="quality-text">
        {overall.charAt(0).toUpperCase() + overall.slice(1)} Quality
      </span>

      {isHovered && (
        <div className="quality-tooltip" role="tooltip">
          <div className="tooltip-row">
            <span>Blur Score:</span>
            <span>{blur_score?.toFixed(2) || "N/A"}</span>
          </div>
          <div className="tooltip-row">
            <span>Brightness:</span>
            <span>{brightness_score?.toFixed(2) || "N/A"}</span>
          </div>
          <div className="tooltip-row">
            <span>Resolution OK:</span>
            <span>{resolution_ok ? "Yes" : "No"}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FaceQualityBadge;
