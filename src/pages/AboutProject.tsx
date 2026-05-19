import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Database, Cpu, Target, ArrowRight, ShieldCheck, Zap } from 'lucide-react';

const AboutProject: React.FC = () => {
  const methodologies = [
    {
      icon: <Database className="text-secondary" size={24} />,
      title: "Data Preprocessing",
      desc: "Handling missing values via median imputation, scaling numerical features with StandardScaler, and encoding categorical variables using One-Hot Encoding. Outliers were detected and treated using the IQR method."
    },
    {
      icon: <Cpu className="text-primary" size={24} />,
      title: "Feature Engineering",
      desc: "Created composite mental health indicators by combining stress, sleep, and mood scores. Applied Recursive Feature Elimination (RFE) to identify the top 6 most impactful predictors."
    },
    {
      icon: <Target className="text-success" size={24} />,
      title: "Model Training & Tuning",
      desc: "Trained 6 distinct algorithms. Used GridSearchCV for hyperparameter tuning. Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) to ensure robust recall for high-risk minority cases."
    }
  ];

  const timeline = [
    { phase: "Phase 1: Research", desc: "Literature review on ML applications in psychological screening." },
    { phase: "Phase 2: Data Engineering", desc: "Sourcing, cleaning, and formatting the mental health dataset." },
    { phase: "Phase 3: Model Architecture", desc: "Designing the ensemble comparison pipeline." },
    { phase: "Phase 4: Frontend Integration", desc: "Building this premium React-based analytical dashboard." },
  ];

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-12">
      <div className="text-center mb-16">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="inline-block bg-white/5 border border-white/10 rounded-full p-4 mb-6">
          <BookOpen className="text-primary w-12 h-12" />
        </motion.div>
        <motion.h1 
          initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
          className="text-4xl md:text-5xl font-bold mb-4"
        >
          Project <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">Documentation</span>
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
          className="text-gray-400 max-w-3xl mx-auto text-lg leading-relaxed"
        >
          MindMine AI is an advanced major college project conceptualized as a Silicon Valley SaaS product. It demonstrates the intersection of Machine Learning classification and premium frontend web development.
        </motion.p>
      </div>

      {/* Problem Statement */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-8 md:p-12 mb-12 relative overflow-hidden">
        <div className="absolute -right-20 -top-20 w-64 h-64 bg-danger/10 rounded-full blur-[80px]" />
        <h2 className="text-2xl font-bold mb-4 flex items-center"><ShieldCheck className="mr-3 text-danger" /> The Problem</h2>
        <p className="text-gray-300 leading-relaxed text-lg max-w-4xl">
          Mental health conditions like depression and severe burnout often go undetected until they reach critical stages. Traditional screening relies on subjective questionnaires with high latency. This project aims to predict mental health risks early by analyzing objective lifestyle factors (sleep, work hours, screen time, physical activity) using Machine Learning.
        </p>
      </motion.div>

      {/* Methodology Cards */}
      <div className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center">Machine Learning Methodology</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {methodologies.map((item, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.1 }}
              className="glass-card p-6 border-t-4 border-t-transparent hover:border-t-primary transition-all"
            >
              <div className="mb-4 bg-white/5 w-12 h-12 flex items-center justify-center rounded-xl">{item.icon}</div>
              <h3 className="text-xl font-bold mb-3">{item.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        {/* Timeline */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h2 className="text-2xl font-bold mb-8 flex items-center"><Zap className="mr-3 text-yellow-400" /> Development Timeline</h2>
          <div className="space-y-8 pl-4 border-l-2 border-white/10">
            {timeline.map((step, idx) => (
              <div key={idx} className="relative">
                <div className="absolute -left-[21px] top-1 w-3 h-3 rounded-full bg-primary ring-4 ring-[#030014]" />
                <h3 className="font-bold text-white mb-1">{step.phase}</h3>
                <p className="text-gray-400 text-sm">{step.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Future Improvements */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="glass-card p-8 border-l-4 border-secondary">
          <h2 className="text-2xl font-bold mb-6">Future Scope</h2>
          <ul className="space-y-4">
            {[
              "Integration with wearable devices (Apple Watch, Fitbit) for real-time biological data.",
              "Deployment of actual trained ML models via Python Flask/FastAPI backend.",
              "Implementing Deep Learning (LSTMs) for longitudinal time-series prediction.",
              "Adding secure user authentication and historical data tracking."
            ].map((item, idx) => (
              <li key={idx} className="flex items-start space-x-3 text-gray-300">
                <ArrowRight className="text-secondary shrink-0 mt-1" size={16} />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>
    </div>
  );
};

export default AboutProject;
