import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BrainCircuit, Activity, BarChart3, Info, LayoutDashboard } from 'lucide-react';

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { name: 'Home', path: '/', icon: <BrainCircuit size={18} /> },
    { name: 'Predict', path: '/predict', icon: <Activity size={18} /> },
    { name: 'Models', path: '/models', icon: <LayoutDashboard size={18} /> },
    { name: 'Analytics', path: '/analytics', icon: <BarChart3 size={18} /> },
    { name: 'About', path: '/about', icon: <Info size={18} /> },
  ];

  return (
    <div className="flex flex-col min-h-screen relative w-full max-w-full m-0 p-0 overflow-x-hidden border-none text-left">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none z-[-1] flex justify-center items-center overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/20 rounded-full blur-[100px] animate-pulse-glow" />
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-secondary/20 rounded-full blur-[100px] animate-pulse-glow" style={{ animationDelay: '2s' }} />
      </div>

      <nav className="fixed top-0 w-full z-50 bg-[#030014]/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <Link to="/" className="flex items-center space-x-3 group">
              <img src="/logo.png" alt="MindMine AI Logo" className="w-10 h-10 rounded-xl object-cover group-hover:scale-105 transition-transform shadow-[0_0_15px_rgba(112,0,255,0.4)]" />
              <span className="text-2xl font-bold text-white tracking-tight">Mind<span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">Mine</span> AI</span>
            </Link>

            <div className="hidden md:flex space-x-1">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.name}
                    to={item.path}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all duration-300 ${isActive
                      ? 'bg-white/10 text-white shadow-[0_0_15px_rgba(255,255,255,0.1)]'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }`}
                  >
                    {item.icon}
                    <span className="font-medium">{item.name}</span>
                  </Link>
                );
              })}
            </div>

            <div className="md:hidden">
              {/* Mobile menu button could go here */}
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-grow pt-24 pb-12 w-full">
        {children}
      </main>

      <footer className="w-full border-t border-white/10 bg-[#030014]/50 backdrop-blur-md py-8">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <img src="/logo.png" alt="MindMine AI Logo" className="w-8 h-8 rounded-lg object-cover shadow-[0_0_10px_rgba(112,0,255,0.3)]" />
            <span className="text-xl font-bold">MindMine AI</span>
          </div>
          <p className="text-gray-400 text-sm mb-4">
            Advanced Mental Health Risk Prediction System.
            <br />
            Built for presentation & demonstration.
          </p>
          <p className="text-gray-500 text-sm flex items-center justify-center space-x-1">
            <span>&copy; 2025</span>
            <span className="mx-1">•</span>
            <span>Crafted with ❤️ by Anoop</span>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default MainLayout;
