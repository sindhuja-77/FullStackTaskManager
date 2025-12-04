
import React, { useEffect, useState } from 'react';

const AddList = ({ refresh,fetchRecommendations }) => {
  const [tasks, setTasks] = useState([]);
  const [editIndex, setEditIndex] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const [editDescription, setEditDescription] = useState('');

  const fetchTasks = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/tasks");
      const data = await res.json();
      setTasks(data);
    } catch (error) {
      console.error('Error fetching tasks:', error);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, [refresh]);

  const deleteTask = async (taskId) => {
    try {
      await fetch(`http://127.0.0.1:8000/tasks/${taskId}`, {
        method: 'DELETE',
      });
     await fetchTasks();
      fetchRecommendations();
    } catch (error) {
      console.error('Error deleting task:', error);
    }
  };

  const toggleCompleted = async (task) => {
    try {
      await fetch(`http://127.0.0.1:8000/tasks/${task.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...task, completed: !task.completed }),
      });
      fetchTasks();
      fetchRecommendations();
    } catch (error) {
      console.error('Error updating task:', error);
    }
  };

  const saveEdit = async (task) => {
    try {
      await fetch(`http://127.0.0.1:8000/tasks/${task.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: editTitle,
          description: editDescription,
          completed: task.completed,
        }),
      });
      setEditIndex(null);
      fetchTasks();
      fetchRecommendations();
    } catch (error) {
      console.error('Error saving edit:', error);
    }
  };

  return (
    <ul>
      {tasks.length === 0 ? (
        <p>No tasks yet. ✅</p>
      ) : (
        tasks.map((task, index) => (
          <li key={task.id}>
            {editIndex === index ? (
              <>
                <input
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                />
                <input
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                />
                <button onClick={() => saveEdit(task)}>Save</button>
                <button onClick={() => setEditIndex(null)}>Cancel</button>
              </>
            ) : (
              <>
                <span
                  style={{
                    textDecoration: task.completed ? 'line-through' : 'none',
                  }}
                >
                  {task.title} - {task.description}
                </span>
                <button onClick={() => toggleCompleted(task)}>
                  {task.completed ? 'Undo' : 'Complete'}
                </button>
                <button onClick={() => deleteTask(task.id)}>Delete</button>
                <button
                  onClick={() => {
                    setEditIndex(index);
                    setEditTitle(task.title);
                    setEditDescription(task.description);
                  }}
                >
                  Edit
                </button>
              </>
            )}
          </li>
        ))
      )}
    </ul>
  );
};

export default AddList;


  