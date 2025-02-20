import React from "react";

const Snake = ({ segments }) => (
  <>
    {segments.map((segment, index) => (
      <div
        key={index}
        className="snake-segment"
        style={{ left: `${segment.x * 20}px`, top: `${segment.y * 20}px` }}
      />
    ))}
  </>
);

export default Snake;
