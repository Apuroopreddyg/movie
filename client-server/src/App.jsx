import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';


const App = () => {
  const [movie, setMovie] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    setMovie(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.get('http://127.0.0.1:5000/recommend', {
        params: { movie }
      });
      setRecommendations(response.data.recommendations);
      setError('');
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
      setRecommendations([]);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center mb-4">Movie Recommendation System</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="input-group">
          <input
            type="text"
            value={movie}
            onChange={handleInputChange}
            className="form-control"
            placeholder="Enter a movie name"
          />
          <div className="input-group-append">
            <button style={{margin:"10px",height:"50px"}} className="btn btn-primary" type="submit">Get Recommendations</button>
          </div>
        </div>
      </form>
      {error && <div className="alert alert-danger">{error}</div>}
      <ul className="list-group">
        {recommendations.map((rec, index) => (
          <li key={index} className="list-group-item">{rec}</li>
        ))}
      </ul>
    </div>
  );
};

export default App;
