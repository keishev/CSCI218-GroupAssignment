import React, { useEffect, useState, useRef } from "react";
import Swal from "sweetalert2";
import Board from "./components/Board";
import "./styles/global/App.css";

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
  const [isGameActive, setIsGameActive] = useState(false);
  const [score, setScore] = useState(0);
  const [showStartScreen, setShowStartScreen] = useState(true);
  const [showLoadingScreen, setShowLoadingScreen] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const gameLoopRef = useRef(null);
  const socketRef = useRef(null);

  const startGame = async () => {
    setShowStartScreen(false);
    setShowLoadingScreen(true);
    try {
      const response = await fetch("http://localhost:5003/start-camera");
      const data = await response.json();
      console.log(data.status);

      if (data) {
        Swal.fire("Success", "Webcam started successfully! Get ready to play!", "success");
        setTimeout(() => {
          setShowLoadingScreen(false);
          setIsGameActive(true);
          setGameOver(false);
          startWebSocketConnection();
        }, 6000); 
      } else {
        Swal.fire("Error", "Failed to start the webcam. Please try again.", "error");
        setShowStartScreen(true);
        setShowLoadingScreen(false);
      }
    } catch (error) {
      console.error("Failed to start the webcam:", error);
      Swal.fire("Error", "Failed to start the webcam. Please try again.", "error");
      setShowStartScreen(true);
      setShowLoadingScreen(false);
    }
  };

  const startWebSocketConnection = () => {
    socketRef.current = new WebSocket("ws://localhost:5001");

    socketRef.current.onopen = () => {
      console.log("WebSocket connection established");
    };

    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (!isGameActive) return;

      const gestureDirection = gestureToDirection[data.gesture];
      if (gestureDirection) {
        setDirection(prevDirection => {
          if (
            (gestureDirection === "LEFT" && prevDirection !== "RIGHT") ||
            (gestureDirection === "RIGHT" && prevDirection !== "LEFT") ||
            (gestureDirection === "UP" && prevDirection !== "DOWN") ||
            (gestureDirection === "DOWN" && prevDirection !== "UP")
          ) {
            return gestureDirection;
          }
          return prevDirection;
        });
      }
    };

    socketRef.current.onerror = (error) => {
      console.error("WebSocket Error: ", error);
    };

    socketRef.current.onclose = () => {
      console.log("WebSocket connection closed");
    };
  };

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
    setGameOver(true);
    setIsGameActive(false);
    if (socketRef.current) socketRef.current.close();
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
        setGameOver(false);
        startWebSocketConnection();
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
      ) : showLoadingScreen ? (
        <div className="loading-screen">
          <h1>Initializing Webcam... Please wait :)</h1>
          <div className="loader"></div>
        </div>
      ) : (
        <div className="game-and-camera-container">
          <div className="game-area">
            <Board snake={snake} food={food} boardSize={BOARD_SIZE} score={score} />
            {gameOver && <h3>Game Over! Your score: {score}</h3>}
          </div>
          <div className="camera-feed">
            <h2>Webcam Feed</h2>
            <img src="http://localhost:5002/video_feed" alt="Webcam Feed" />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
