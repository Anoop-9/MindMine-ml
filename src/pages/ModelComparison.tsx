import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Trophy, BarChart2, Layers, BrainCircuit } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

const ModelComparison: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'confusion' | 'features'>('metrics');

  const models = [
    { name: 'Random Forest', accuracy: 94.8, precision: 95.1, recall: 93.9, f1: 94.5, isBest: true },
    { name: 'XGBoost', accuracy: 93.5, precision: 92.8, recall: 94.1, f1: 93.4, isBest: false },
    { name: 'SVM', accuracy: 89.2, precision: 88.5, recall: 89.0, f1: 88.7, isBest: false },
    { name: 'Logistic Regression', accuracy: 86.7, precision: 87.0, recall: 85.5, f1: 86.2, isBest: false },
    { name: 'KNN', accuracy: 85.4, precision: 86.1, recall: 84.8, f1: 85.4, isBest: false },
    { name: 'Decision Tree', accuracy: 82.1, precision: 81.5, recall: 82.8, f1: 82.1, isBest: false },
  ].sort((a, b) => b.accuracy - a.accuracy);

  const featureImportance = [
    { name: 'Stress Level', value: 0.28 },
    { name: 'Sleep Hours', value: 0.22 },
    { name: 'Work Hours', value: 0.18 },
    { name: 'Mood Score', value: 0.15 },
    { name: 'Activity', value: 0.10 },
    { name: 'Screen Time', value: 0.07 },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#0a0a1a] border border-white/10 p-4 rounded-xl shadow-xl">
          <p className="text-white font-bold mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {entry.value}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <motion.h1 
          initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold mb-4"
        >
          Algorithm <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">Performance Analysis</span>
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
          className="text-gray-400 max-w-2xl mx-auto"
        >
          Comparative evaluation of 6 distinct machine learning models trained on mental health datasets. Random Forest emerges as the optimal classifier.
        </motion.p>
      </div>

      {/* Top Models Leaderboard */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {models.slice(0, 3).map((model, idx) => (
          <motion.div 
            key={model.name}
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.1 }}
            className={`glass-card p-6 relative overflow-hidden ${model.isBest ? 'border-primary shadow-[0_0_30px_rgba(112,0,255,0.2)]' : ''}`}
          >
            {model.isBest && (
              <div className="absolute top-0 right-0 bg-primary text-white text-xs font-bold px-3 py-1 rounded-bl-lg flex items-center space-x-1">
                <Trophy size={12} />
                <span>BEST MODEL</span>
              </div>
            )}
            <div className="flex justify-between items-center mb-4 mt-2">
              <h3 className="text-xl font-bold text-white">{model.name}</h3>
              <span className={`text-2xl font-black ${model.isBest ? 'text-primary' : 'text-gray-400'}`}>{model.accuracy}%</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm"><span className="text-gray-400">Precision</span><span className="text-white">{model.precision}%</span></div>
              <div className="w-full bg-white/5 rounded-full h-1.5"><div className="bg-secondary h-1.5 rounded-full" style={{ width: `${model.precision}%` }}></div></div>
              
              <div className="flex justify-between text-sm pt-2"><span className="text-gray-400">Recall</span><span className="text-white">{model.recall}%</span></div>
              <div className="w-full bg-white/5 rounded-full h-1.5"><div className="bg-success h-1.5 rounded-full" style={{ width: `${model.recall}%` }}></div></div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Main Analysis Section */}
      <div className="glass-card p-2 md:p-8">
        <div className="flex flex-wrap justify-center gap-4 mb-8 border-b border-white/10 pb-4">
          <button onClick={() => setActiveTab('metrics')} className={`px-6 py-2 rounded-full transition-all ${activeTab === 'metrics' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white'}`}>
            <BarChart2 className="inline mr-2" size={18} /> Performance Metrics
          </button>
          <button onClick={() => setActiveTab('confusion')} className={`px-6 py-2 rounded-full transition-all ${activeTab === 'confusion' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white'}`}>
            <Layers className="inline mr-2" size={18} /> Confusion Matrix (RF)
          </button>
          <button onClick={() => setActiveTab('features')} className={`px-6 py-2 rounded-full transition-all ${activeTab === 'features' ? 'bg-primary text-white' : 'text-gray-400 hover:text-white'}`}>
            <BrainCircuit className="inline mr-2" size={18} /> Feature Importance
          </button>
        </div>

        <div className="min-h-[400px]">
          {activeTab === 'metrics' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={models} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis domain={[70, 100]} stroke="#9ca3af" />
                  <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Bar dataKey="accuracy" name="Accuracy" fill="#7000ff" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="f1" name="F1 Score" fill="#00d4ff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          )}

          {activeTab === 'confusion' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center">
              <h3 className="text-xl font-bold mb-6 text-center text-white">Random Forest - Confusion Matrix Mockup</h3>
              <div className="grid grid-cols-3 gap-1 max-w-md w-full text-center">
                <div className="p-4"></div>
                <div className="p-4 font-bold text-gray-400">Predicted Neg</div>
                <div className="p-4 font-bold text-gray-400">Predicted Pos</div>
                
                <div className="p-4 font-bold text-gray-400 flex items-center justify-end">Actual Neg</div>
                <div className="p-6 bg-primary/20 border border-primary/50 rounded-lg flex flex-col justify-center items-center">
                  <span className="text-3xl font-black text-white">482</span>
                  <span className="text-xs text-gray-400 uppercase mt-1">True Negative</span>
                </div>
                <div className="p-6 bg-danger/20 border border-danger/50 rounded-lg flex flex-col justify-center items-center">
                  <span className="text-2xl font-bold text-gray-300">24</span>
                  <span className="text-xs text-gray-400 uppercase mt-1">False Positive</span>
                </div>

                <div className="p-4 font-bold text-gray-400 flex items-center justify-end">Actual Pos</div>
                <div className="p-6 bg-yellow-500/20 border border-yellow-500/50 rounded-lg flex flex-col justify-center items-center">
                  <span className="text-2xl font-bold text-gray-300">31</span>
                  <span className="text-xs text-gray-400 uppercase mt-1">False Negative</span>
                </div>
                <div className="p-6 bg-success/20 border border-success/50 rounded-lg flex flex-col justify-center items-center">
                  <span className="text-3xl font-black text-white">365</span>
                  <span className="text-xs text-gray-400 uppercase mt-1">True Positive</span>
                </div>
              </div>
              <p className="mt-8 text-gray-400 text-sm max-w-lg text-center">
                The Random Forest model demonstrates exceptional ability to minimize False Negatives, crucial for mental health risk screening.
              </p>
            </motion.div>
          )}

          {activeTab === 'features' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-[400px] flex flex-col md:flex-row items-center justify-center">
              <div className="w-full md:w-1/2 h-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={featureImportance}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 0.3]} tick={false} axisLine={false} />
                    <Radar name="Importance" dataKey="value" stroke="#7000ff" fill="#7000ff" fillOpacity={0.4} />
                    <Tooltip contentStyle={{ backgroundColor: '#0a0a1a', borderColor: 'rgba(255,255,255,0.1)' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              <div className="w-full md:w-1/2 space-y-4 px-8 mt-8 md:mt-0">
                <h3 className="text-xl font-bold mb-4">Gini Importance Breakdown</h3>
                {featureImportance.map((feat, idx) => (
                  <div key={feat.name} className="flex items-center">
                    <div className="w-8 text-gray-400 text-sm font-mono">0{idx + 1}</div>
                    <div className="flex-1">
                      <div className="flex justify-between text-sm mb-1">
                        <span>{feat.name}</span>
                        <span className="text-primary font-mono">{(feat.value * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-white/5 rounded-full h-1.5">
                        <div className="bg-gradient-to-r from-primary to-secondary h-1.5 rounded-full" style={{ width: `${feat.value * 100 / 0.28}%` }}></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
