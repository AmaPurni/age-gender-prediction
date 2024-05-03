import React, { useState } from 'react';
import logo from './logo.svg';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap CSS
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setImageFile(file);

    // Display the selected image
    const url = URL.createObjectURL(file);
    setImageUrl(url);
  };

  const handlePredict = async () => {
    if (!imageFile) {
      alert('Please select an image file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="mb-4">Gender and Age Prediction App</h1>
        <input type="file" accept="image/*" onChange={handleImageUpload} className="mb-3" />
        <button onClick={handlePredict} className="btn btn-primary mb-3">Predict</button>

        {imageUrl && (
          <div className="mb-4">
            <h2>Selected Image</h2>
            <img src={imageUrl} alt="Selected" className="img-fluid uploaded-image" />
          </div>
        )}

        {prediction && (
          <div>
            <h2>Prediction Results</h2>
            <p>Predicted Age: {prediction.age}</p>
            <p>Predicted Gender: {prediction.gender}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
