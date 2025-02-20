import React from "react";

const StartScreen = ({ onStart }) => (
  <div className="start-screen">
    <h1>Welcome to our group assignment!</h1>
    <button className="start-button" onClick={onStart}>
      Start Game
    </button>
  </div>
);

export default StartScreen;
