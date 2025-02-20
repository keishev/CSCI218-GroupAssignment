import React from "react";

const Food = ({ position }) => (
  <div
    className="food"
    style={{ left: `${position.x * 20}px`, top: `${position.y * 20}px` }}
  />
);

export default Food;