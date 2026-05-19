import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Brain, Moon, Zap, Coffee, Smartphone, HeartPulse, RefreshCw, Database } from 'lucide-react';
import { ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts';

const PredictionDashboard: React.FC = () => {
  const [isPredicting, setIsPredicting] = useState(false);
  const [results, setResults] = useState<{ depression: number; burnout: number; wellness: number } | null>(null);

  const [inputs, setInputs] = useState({
    sleep: 7,
    stress: 5,
    work: 8,
    mood: 6,
    screen: 5,
    activity: 3
  });

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    setResults(null);
    
    // Simulate API call and ML inference
    setTimeout(() => {
      // Dummy logic based on inputs
      const badHabits = (10 - inputs.sleep) + inputs.stress + (inputs.work - 8) + (10 - inputs.mood) + inputs.screen - inputs.activity;
      
      const depressionRisk = Math.min(95, Math.max(5, 15 + (badHabits * 2.5) + Math.random() * 10));
      const burnoutRisk = Math.min(98, Math.max(10, 20 + (inputs.stress * 4) + (inputs.work * 2) - (inputs.sleep * 2) + Math.random() * 10));
      const wellnessScore = Math.min(100, Math.max(0, 100 - ((depressionRisk + burnoutRisk) / 2) + Math.random() * 5));

      setResults({
        depression: parseFloat(depressionRisk.toFixed(1)),
        burnout: parseFloat(burnoutRisk.toFixed(1)),
        wellness: parseFloat(wellnessScore.toFixed(1))
      });
      setIsPredicting(false);
    }, 2000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputs({ ...inputs, [e.target.name]: parseFloat(e.target.value) });
  };

  const GaugeChart = ({ value, label, color }: { value: number; label: string; color: string }) => {
    const data = [{ name: label, value: value, fill: color }];
    return (
      <div className="flex flex-col items-center justify-center relative h-48 w-48 mx-auto">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart 
            cx="50%" cy="50%" 
            innerRadius="70%" outerRadius="100%" 
            barSize={15} data={data} 
            startAngle={180} endAngle={0}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
            <RadialBar background={{ fill: 'rgba(255,255,255,0.05)' }} dataKey="value" cornerRadius={10} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute flex flex-col items-center justify-center top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-[20%] text-center">
          <span className="text-3xl font-bold" style={{ color }}>{value}%</span>
          <span className="text-xs text-gray-400 mt-1 uppercase tracking-wider">{label}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <motion.h1 
          initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold mb-4"
        >
          Real-time <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">Prediction Inference</span>
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
          className="text-gray-400"
        >
          Enter patient lifestyle parameters to simulate the ensemble ML prediction pipeline.
        </motion.p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Input Form */}
        <motion.div 
          initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-5 glass-card p-8"
        >
          <div className="flex items-center space-x-3 mb-6 pb-4 border-b border-white/10">
            <Activity className="text-primary" />
            <h2 className="text-xl font-bold">Input Parameters</h2>
          </div>

          <form onSubmit={handlePredict} className="space-y-6">
            <div className="space-y-4">
              {[
                { name: 'sleep', label: 'Sleep (Hours)', icon: <Moon size={16}/>, min: 0, max: 24 },
                { name: 'stress', label: 'Stress Level (1-10)', icon: <Zap size={16}/>, min: 1, max: 10 },
                { name: 'work', label: 'Work (Hours/Day)', icon: <Coffee size={16}/>, min: 0, max: 24 },
                { name: 'mood', label: 'Mood Score (1-10)', icon: <HeartPulse size={16}/>, min: 1, max: 10 },
                { name: 'screen', label: 'Screen Time (Hours)', icon: <Smartphone size={16}/>, min: 0, max: 24 },
                { name: 'activity', label: 'Physical Activity (Hours)', icon: <Activity size={16}/>, min: 0, max: 10 },
              ].map((field) => (
                <div key={field.name}>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-300 flex items-center space-x-2">
                      <span className="text-gray-500">{field.icon}</span>
                      <span>{field.label}</span>
                    </label>
                    <span className="text-primary font-mono text-sm">{inputs[field.name as keyof typeof inputs]}</span>
                  </div>
                  <input 
                    type="range" 
                    name={field.name}
                    min={field.min} max={field.max} step="0.5"
                    value={inputs[field.name as keyof typeof inputs]}
                    onChange={handleChange}
                    className="w-full accent-primary bg-white/10 h-2 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              ))}
            </div>

            <button 
              type="submit" 
              disabled={isPredicting}
              className={`w-full btn-primary py-4 mt-8 flex items-center justify-center space-x-2 ${isPredicting ? 'opacity-70 cursor-not-allowed' : ''}`}
            >
              {isPredicting ? (
                <>
                  <RefreshCw className="animate-spin" size={20} />
                  <span>Running Inference...</span>
                </>
              ) : (
                <>
                  <Brain size={20} />
                  <span>Run ML Pipeline</span>
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Results Panel */}
        <motion.div 
          initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-7"
        >
          <div className="glass-card p-8 h-full relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-[80px]" />
            
            <div className="flex items-center space-x-3 mb-8 pb-4 border-b border-white/10 relative z-10">
              <Brain className="text-secondary" />
              <h2 className="text-xl font-bold">Prediction Output</h2>
            </div>

            {isPredicting ? (
              <div className="h-64 flex flex-col items-center justify-center text-primary relative z-10">
                <RefreshCw className="w-12 h-12 animate-spin mb-4" />
                <p className="text-lg font-mono animate-pulse">Computing ensemble weights...</p>
              </div>
            ) : results ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                className="relative z-10"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                  <div className="bg-[#0a0a1a]/50 border border-white/5 rounded-2xl p-4">
                    <GaugeChart value={results.depression} label="Depression Risk" color={results.depression > 70 ? '#ef4444' : results.depression > 40 ? '#f59e0b' : '#10b981'} />
                  </div>
                  <div className="bg-[#0a0a1a]/50 border border-white/5 rounded-2xl p-4">
                    <GaugeChart value={results.burnout} label="Burnout Risk" color={results.burnout > 70 ? '#ef4444' : results.burnout > 40 ? '#f59e0b' : '#10b981'} />
                  </div>
                </div>

                <div className="bg-gradient-to-r from-primary/10 to-secondary/10 border border-white/10 rounded-2xl p-6 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-1">Overall Wellness Score</h3>
                    <p className="text-3xl font-bold text-white">{results.wellness} <span className="text-lg font-normal text-gray-500">/ 100</span></p>
                  </div>
                  <div className="w-16 h-16 rounded-full border-4 border-success flex items-center justify-center bg-success/10 text-success">
                    <HeartPulse size={28} />
                  </div>
                </div>

                <div className="mt-6 p-4 bg-white/5 rounded-xl border border-white/5 text-sm text-gray-400">
                  <span className="text-primary font-bold">AI Insight: </span>
                  {results.wellness > 75 ? "Patient is exhibiting healthy lifestyle patterns." :
                   results.wellness > 50 ? "Moderate stress indicators detected. Recommend lifestyle adjustments." :
                   "High risk profile. Immediate intervention or consultation recommended."}
                </div>
              </motion.div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-gray-500 relative z-10">
                <Database className="w-16 h-16 mb-4 opacity-50" />
                <p>Awaiting data for inference.</p>
              </div>
            )}
          </div>
        </motion.div>

      </div>
    </div>
  );
};

export default PredictionDashboard;
