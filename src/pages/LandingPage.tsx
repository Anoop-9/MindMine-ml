import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Brain, Sparkles, Activity, ShieldCheck, Zap, Database } from 'lucide-react';

const LandingPage: React.FC = () => {
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  const features = [
    { icon: <Brain className="text-primary" size={32} />, title: "Advanced ML Models", desc: "Powered by 6 state-of-the-art machine learning algorithms." },
    { icon: <Activity className="text-secondary" size={32} />, title: "Real-time Analysis", desc: "Instant mental health risk prediction based on lifestyle inputs." },
    { icon: <ShieldCheck className="text-success" size={32} />, title: "High Accuracy", desc: "Achieving up to 94.8% accuracy with our top-performing model." },
  ];

  return (
    <div className="w-full flex flex-col items-center justify-center">
      {/* Hero Section */}
      <section className="relative w-full max-w-7xl mx-auto px-4 pt-20 pb-32 flex flex-col items-center text-center">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="max-w-4xl mx-auto z-10"
        >
          <motion.div variants={fadeIn} className="inline-flex items-center space-x-2 bg-white/5 border border-white/10 rounded-full px-4 py-2 mb-8">
            <Sparkles className="text-secondary" size={16} />
            <span className="text-sm font-medium text-gray-300">MindMine AI v2.0 is now live</span>
          </motion.div>
          
          <motion.h1 variants={fadeIn} className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8">
            Analyze Mental Health with
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-purple-400 to-secondary animate-pulse-glow">
              Artificial Intelligence
            </span>
          </motion.h1>
          
          <motion.p variants={fadeIn} className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
            A state-of-the-art prediction system designed to identify depression and burnout risks early using advanced Machine Learning models.
          </motion.p>
          
          <motion.div variants={fadeIn} className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/predict" className="w-full sm:w-auto btn-primary flex items-center justify-center space-x-2 text-lg">
              <Activity size={20} />
              <span>Analyze Now</span>
            </Link>
            <Link to="/models" className="w-full sm:w-auto px-8 py-3 rounded-full font-semibold text-white bg-white/5 border border-white/10 hover:bg-white/10 transition-colors flex items-center justify-center space-x-2 text-lg">
              <Database size={20} />
              <span>View Models</span>
            </Link>
          </motion.div>
        </motion.div>

        {/* Hero Image Mockup */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-20 relative w-full max-w-5xl mx-auto"
        >
          <div className="absolute inset-0 bg-gradient-to-t from-[#030014] via-transparent to-transparent z-10" />
          <div className="glass-card overflow-hidden border border-white/10 shadow-[0_0_50px_rgba(112,0,255,0.2)]">
            <div className="h-12 border-b border-white/10 bg-white/5 flex items-center px-4 space-x-2">
              <div className="w-3 h-3 rounded-full bg-danger/80" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <div className="w-3 h-3 rounded-full bg-success/80" />
            </div>
            <div className="p-8 bg-[#0a0a1a] h-[400px] flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)', backgroundSize: '30px 30px' }} />
              <div className="z-10 text-center">
                <Brain className="w-32 h-32 text-primary mx-auto mb-6 animate-float" />
                <h3 className="text-2xl font-bold text-white mb-2">System Ready</h3>
                <p className="text-gray-400">Awaiting user input for real-time inference.</p>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="w-full max-w-7xl mx-auto px-4 py-24 relative">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-4">Enterprise-Grade Architecture</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">Built with premium web technologies to deliver a fast, responsive, and breathtaking user experience.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feat, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.2 }}
              className="glass-card p-8 group hover:-translate-y-2 transition-transform"
            >
              <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                {feat.icon}
              </div>
              <h3 className="text-xl font-bold mb-3">{feat.title}</h3>
              <p className="text-gray-400 leading-relaxed">{feat.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="w-full max-w-4xl mx-auto px-4 py-24 text-center">
        <div className="glass-card p-12 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 z-0" />
          <div className="relative z-10">
            <Zap className="w-16 h-16 text-yellow-400 mx-auto mb-6" />
            <h2 className="text-4xl font-bold mb-6">Ready to Experience the Future?</h2>
            <p className="text-xl text-gray-300 mb-8 max-w-xl mx-auto">
              Run an instant analysis using our ensemble of simulated machine learning models.
            </p>
            <Link to="/predict" className="btn-primary inline-flex items-center space-x-2 text-xl px-10 py-4">
              <span>Start Analysis</span>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
