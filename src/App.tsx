import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import LandingPage from './pages/LandingPage';
import PredictionDashboard from './pages/PredictionDashboard';
import ModelComparison from './pages/ModelComparison';
import Analytics from './pages/Analytics';
import AboutProject from './pages/AboutProject';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/predict" element={<PredictionDashboard />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/about" element={<AboutProject />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
