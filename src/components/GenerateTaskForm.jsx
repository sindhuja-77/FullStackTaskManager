
import React, { useState } from 'react';

const GenerateTaskForm = ({ fetchTasks }) => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if(loading || !prompt.trim()) return; //prevent repid re-clicks
    setLoading(true);

    try {
      const res = await fetch('http://127.0.0.1:8000/generate_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to generate task');
      }

      const newTask = await res.json(); // this is the new task object
      setPrompt('');
      fetchTasks(newTask); // trigger a refresh in TaskList
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Generate Task via AI</h2>
      <input
        type="text"
        placeholder="Enter a task prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Task'}
      </button>
    </div>
  );
};

export default GenerateTaskForm;