import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import { TrendingDown, TrendingUp, Calendar, Clock } from 'lucide-react';

const Analytics: React.FC = () => {
  const weeklyTrendData = [
    { day: 'Mon', stress: 7, wellness: 65, sleep: 5 },
    { day: 'Tue', stress: 8, wellness: 55, sleep: 4.5 },
    { day: 'Wed', stress: 6, wellness: 70, sleep: 6.5 },
    { day: 'Thu', stress: 5, wellness: 75, sleep: 7 },
    { day: 'Fri', stress: 4, wellness: 85, sleep: 8 },
    { day: 'Sat', stress: 3, wellness: 90, sleep: 9 },
    { day: 'Sun', stress: 4, wellness: 88, sleep: 8.5 },
  ];

  // Dummy data for correlation scatter plot
  const correlationData = Array.from({ length: 50 }, () => {
    const sleep = 4 + Math.random() * 6; // 4 to 10 hours
    // Negative correlation with some noise
    const stress = Math.max(1, Math.min(10, 10 - (sleep - 4) * 1.5 + (Math.random() * 3 - 1.5)));
    return { sleep, stress, z: 100 };
  });

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#0a0a1a] border border-white/10 p-3 rounded-lg shadow-xl text-sm">
          <p className="font-bold mb-1">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-12">
        <div>
          <motion.h1 
            initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold mb-4"
          >
            Cohort <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">Analytics</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
            className="text-gray-400 max-w-xl"
          >
            Aggregated anonymized data visualizations demonstrating trends in stress, sleep, and overall wellness.
          </motion.p>
        </div>
        <motion.div 
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          className="mt-6 md:mt-0 flex items-center space-x-4 text-sm"
        >
          <div className="bg-white/5 border border-white/10 px-4 py-2 rounded-full flex items-center space-x-2">
            <Calendar size={16} className="text-gray-400" />
            <span>Last 7 Days</span>
          </div>
          <div className="bg-white/5 border border-white/10 px-4 py-2 rounded-full flex items-center space-x-2">
            <Clock size={16} className="text-gray-400" />
            <span>Real-time Sync</span>
          </div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Weekly Wellness Trend */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold">Weekly Wellness Trend</h3>
            <div className="flex items-center text-success text-sm font-medium">
              <TrendingUp size={16} className="mr-1" /> +12.5%
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={weeklyTrendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorWellness" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="day" stroke="#9ca3af" axisLine={false} tickLine={false} />
                <YAxis stroke="#9ca3af" axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="wellness" name="Wellness Score" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorWellness)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Stress vs Sleep Dynamics */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold">Stress vs. Sleep Dynamics</h3>
            <div className="flex items-center text-danger text-sm font-medium">
              <TrendingDown size={16} className="mr-1" /> Inverse Correl.
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={weeklyTrendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="day" stroke="#9ca3af" axisLine={false} tickLine={false} />
                <YAxis yAxisId="left" stroke="#ef4444" axisLine={false} tickLine={false} domain={[0, 10]} />
                <YAxis yAxisId="right" orientation="right" stroke="#00d4ff" axisLine={false} tickLine={false} domain={[0, 10]} />
                <Tooltip content={<CustomTooltip />} />
                <Line yAxisId="left" type="monotone" dataKey="stress" name="Stress Level" stroke="#ef4444" strokeWidth={3} dot={{ r: 4, fill: '#ef4444' }} />
                <Line yAxisId="right" type="monotone" dataKey="sleep" name="Sleep (hrs)" stroke="#00d4ff" strokeWidth={3} dot={{ r: 4, fill: '#00d4ff' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Scatter Correlation */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass-card p-6 lg:col-span-2">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold">Population Distribution: Sleep vs Stress</h3>
            <div className="text-sm text-gray-400">n = 50 (Simulated Cohort)</div>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" dataKey="sleep" name="Sleep Hours" stroke="#9ca3af" domain={[3, 11]} label={{ value: 'Sleep (Hours)', position: 'insideBottom', offset: -10, fill: '#9ca3af' }} />
                <YAxis type="number" dataKey="stress" name="Stress Level" stroke="#9ca3af" domain={[0, 11]} label={{ value: 'Stress Level (1-10)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <ZAxis type="number" dataKey="z" range={[50, 50]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} 
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-[#0a0a1a] border border-white/10 p-2 rounded-lg shadow-xl text-xs">
                          <p className="text-gray-300">Sleep: <span className="text-secondary font-bold">{Number(payload[0].value).toFixed(1)} hrs</span></p>
                          <p className="text-gray-300">Stress: <span className="text-danger font-bold">{Number(payload[1].value).toFixed(1)}/10</span></p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Population" data={correlationData} fill="#7000ff" fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

      </div>
    </div>
  );
};

export default Analytics;
