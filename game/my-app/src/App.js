import React, { useEffect, useState, useRef } from "react";
import Swal from "sweetalert2";
import Board from "./components/Board";
import "./styles/global/App.css";

const GRID_SIZE = 20;
const BOARD_SIZE = 15;

const gestureToDirection = {
    "like": "UP",
    "dislike": "DOWN",
    "fist": "LEFT",
    "ok": "RIGHT"
};

const getRandomPosition = () => ({
  x: Math.floor(Math.random() * BOARD_SIZE),
  y: Math.floor(Math.random() * BOARD_SIZE),
});

function App() {
  const [snake, setSnake] = useState([{ x: 5, y: 5 }]);
  const [food, setFood] = useState(getRandomPosition());
  const [direction, setDirection] = useState("RIGHT");
  const [isGameActive, setIsGameActive] = useState(false); // Start with the game inactive
  const [score, setScore] = useState(0);
  const [showStartScreen, setShowStartScreen] = useState(true); // Show start screen initially
  const gameLoopRef = useRef(null);

  const startGame = () => {
    setShowStartScreen(false);
    setIsGameActive(true);
  };

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:5001");

    socket.onopen = () => {
      console.log("WebSocket connection established");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (!isGameActive) return;

      const gestureDirection = gestureToDirection[data.gesture];

      if (gestureDirection === "LEFT" && direction !== "RIGHT") setDirection("LEFT");
      if (gestureDirection === "RIGHT" && direction !== "LEFT") setDirection("RIGHT");
      if (gestureDirection === "UP" && direction !== "DOWN") setDirection("UP");
      if (gestureDirection === "DOWN" && direction !== "UP") setDirection("DOWN");
    };

    socket.onerror = (error) => {
      console.error("WebSocket Error: ", error);
    };

    socket.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => socket.close();
  }, [direction, isGameActive]);

  useEffect(() => {
    if (isGameActive) {
      gameLoopRef.current = setInterval(moveSnake, 200);
    } else {
      clearInterval(gameLoopRef.current);
    }
    return () => clearInterval(gameLoopRef.current);
  }, [snake, isGameActive]);

  const moveSnake = () => {
    if (!isGameActive) return;

    let newSnake = [...snake];
    let head = { ...newSnake[0] };

    if (direction === "UP") head.y -= 1;
    if (direction === "DOWN") head.y += 1;
    if (direction === "LEFT") head.x -= 1;
    if (direction === "RIGHT") head.x += 1;

    if (
      head.x < 0 || head.x >= BOARD_SIZE ||
      head.y < 0 || head.y >= BOARD_SIZE ||
      newSnake.some((segment) => segment.x === head.x && segment.y === head.y)
    ) {
      showGameOverAlert();
      setIsGameActive(false);
      return;
    }

    if (head.x === food.x && head.y === food.y) {
      setFood(getRandomPosition());
      setScore(prevScore => prevScore + 10);
    } else {
      newSnake.pop();
    }

    newSnake.unshift(head);
    setSnake(newSnake);
  };

  const showGameOverAlert = () => {
    Swal.fire({
      title: 'Game Over!',
      text: `Your score: ${score}. Do you want to play again?`,
      icon: 'error',
      showCancelButton: true,
      confirmButtonText: 'Play Again',
      cancelButtonText: 'Quit',
      allowOutsideClick: false,
    }).then((result) => {
      if (result.isConfirmed) {
        setSnake([{ x: 5, y: 5 }]);
        setFood(getRandomPosition());
        setDirection("RIGHT");
        setIsGameActive(true);
        setScore(0);
      } else {
        setIsGameActive(false);
        setShowStartScreen(true);
      }
    });
  };

  return (
    <div className="game-container">
      {showStartScreen ? (
        <div className="start-screen">
          <h1>Welcome to our group assignment!</h1>
          <button className="start-button" onClick={startGame}>
            Start Game
          </button>
        </div>
      ) : (
        <>
          <Board snake={snake} food={food} boardSize={BOARD_SIZE} score={score} />
          {!isGameActive && <h3>Game is not active. Please restart to play again.</h3>}
        </>
      )}
    </div>
  );
}

export default App;
