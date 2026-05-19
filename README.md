# MindMine AI 🧠

MindMine AI is an advanced, premium mental health risk prediction system dashboard designed for a major college project. It simulates a state-of-the-art Machine Learning ensemble model to predict Depression and Burnout risks based on objective lifestyle inputs.

This project is built to resemble a high-end Silicon Valley AI SaaS product with a futuristic UI, glassmorphism, animated charts, and professional presentation quality.

> **Note:** This is a **frontend-only** application designed for demonstration and presentation purposes. All ML models and predictions are simulated within the React client. There is no backend or database required to run this application.

## 🚀 Features

- **Premium UI/UX:** Dark mode, glassmorphism, floating glow effects, and micro-animations.
- **Real-time Inference Simulation:** Predict depression risk, burnout risk, and wellness score instantly.
- **Model Comparison:** Visual analysis of Random Forest (Best Model), XGBoost, SVM, Logistic Regression, KNN, and Decision Tree.
- **Advanced Visualizations:** Animated gauges, Recharts integration, confusion matrix mockup, and feature importance radar charts.
- **100% Client-Side:** Runs entirely in the browser, making it extremely easy to deploy and share.

## 💻 Tech Stack

- **Framework:** React 19 + Vite + TypeScript
- **Styling:** Tailwind CSS v4
- **Animations:** Framer Motion
- **Icons:** Lucide React
- **Charts:** Recharts
- **Routing:** React Router DOM

## 🛠️ Local Development

To run this project locally, follow these simple steps:

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## 🌐 Netlify Deployment Guide

This project is fully ready for zero-configuration static deployment on Netlify.

### Method 1: Drag and Drop (Easiest)
1. Run `npm run build` in your terminal. This will create a `dist/` folder.
2. Go to [Netlify Drop](https://app.netlify.com/drop).
3. Drag and drop the `dist/` folder into the upload box.
4. Your site is now live!

### Method 2: Git Integration
1. Push this repository to GitHub, GitLab, or Bitbucket.
2. Log into [Netlify](https://app.netlify.com) and click **"Add new site" -> "Import an existing project"**.
3. Connect your Git provider and select this repository.
4. Netlify will auto-detect Vite settings:
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`
5. Click **"Deploy site"**.

*Note: A `netlify.toml` file is included to handle React Router client-side routing automatically.*

## 📂 Project Structure

```
src/
├── assets/         # Images and static assets
├── components/     # Reusable UI components
├── layouts/        # Page layout wrappers (Navbar, Footer)
├── pages/          # Main application pages
│   ├── LandingPage.tsx
│   ├── PredictionDashboard.tsx
│   ├── ModelComparison.tsx
│   ├── Analytics.tsx
│   └── AboutProject.tsx
├── App.tsx         # Routing configuration
├── index.css       # Global styles & Tailwind config
└── main.tsx        # React entry point
```

---
*Developed for Major Project Presentation. Demonstrates high-proficiency in modern web architecture and UI/UX design.*
