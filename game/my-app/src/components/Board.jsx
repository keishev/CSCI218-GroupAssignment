import React from "react";

const Board = ({ snake, food, boardSize, score }) => (
  <div className="board-container">
    <div className="score-container">
      <div className="score-display">Score: {score}</div>
    </div>
    <div className="board">
      {Array.from({ length: boardSize }).map((_, row) =>
        Array.from({ length: boardSize }).map((_, col) => {
          const isSnake = snake.some(segment => segment.x === col && segment.y === row);
          const isFood = food.x === col && food.y === row;
          return (
            <div
              key={`${row}-${col}`}
              className={`cell ${isSnake ? "snake" : ""} ${isFood ? "food" : ""}`}
            ></div>
          );
        })
      )}
    </div>
  </div>
);

export default Board;
