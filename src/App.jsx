
import React, { useState } from "react";
import AddTask from "./components/AddTask";
import AddList from "./components/AddList";
import GenerateTaskForm from "./components/GenerateTaskForm";
import Recommendations from "./components/Recommendations";
import "./App.css";

const App = () => {
  const [refresh, setRefresh] = useState(false);
  const [recommendation, setRecommendation] = useState("");
  const [loading, setLoading] = useState(false);

  
 // const fetchTasks = () => setRefresh(!refresh);
 const fetchTasks = () => setRefresh(prev => !prev);



  const fetchRecommendations = async () => {
    setLoading(true);
    try {
       const res = await fetch('http://127.0.0.1:8000/recommendations');
      const data = await res.json();
      setRecommendation(data.recommendation || "No tasks to recommend.");
    }  catch (error) {
      console.error('Error fetching recommendations:', error);
      setRecommendation('Failed to fetch recommendations.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>📝 Task Manager App</h1>

      <AddTask fetchTasks={fetchTasks} />
      <GenerateTaskForm fetchTasks={fetchTasks} />
      <AddList refresh={refresh} fetchRecommendations={fetchRecommendations} />
      <Recommendations
        recommendation={recommendation}
        fetchRecommendations={fetchRecommendations}
        loading={loading}
      />
    </div>
  );
};

export default App;