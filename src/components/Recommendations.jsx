import React from "react";

const Recommendations = ({ recommendation, fetchRecommendations, loading }) => {
  return (
    <div className="recommendations">
      <h3>AI Recommendations</h3>
      <button onClick={fetchRecommendations} disabled={loading}>
        {loading ? "Fetching..." : "Get Recommendation"}
      </button>

      {recommendation && (
        <div className="recommendation-box">
          {recommendation.split("\n").map((line, index) => {
            const cleaned = line.replace(/\*\*/g, "").replace(/^- /, "• ");
            return <p key={index}>{cleaned}</p>;
          })}
        </div>
      )}
    </div>
  );
};

export default Recommendations;