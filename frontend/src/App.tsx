import React, { useState } from "react";
import { AnalyzePage } from "./pages/AnalyzePage";
import { HistoryPage } from "./pages/HistoryPage";
import { AboutPage } from "./pages/AboutPage";

type ActivePage = "analyze" | "history" | "about";

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<ActivePage>("analyze");

  return (
    <div className="min-h-screen flex flex-col font-sans antialiased text-slate-100 select-none">
      {/* Global Navigation Bar */}
      <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/60 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-4 md:px-8 py-4 flex items-center justify-between">
          {/* Logo */}
          <div 
            onClick={() => setCurrentPage("analyze")} 
            className="flex items-center gap-2 cursor-pointer group"
          >
            <span className="w-8 h-8 rounded-xl bg-gradient-to-tr from-sky-400 to-indigo-600 flex items-center justify-center font-heading font-extrabold text-white text-sm shadow-md shadow-sky-500/20 group-hover:scale-105 transition-transform">
              &Phi;
            </span>
            <span className="font-heading font-extrabold tracking-tight text-slate-200 text-sm md:text-base group-hover:text-white transition-colors">
              Hybrid<span className="text-sky-400">Detector</span>
            </span>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1 md:gap-2">
            {(["analyze", "history", "about"] as ActivePage[]).map((page) => (
              <button
                key={page}
                onClick={() => setCurrentPage(page)}
                className={`px-3 py-1.5 md:px-4 md:py-2 rounded-xl text-xs md:text-sm font-heading font-semibold capitalize transition-all cursor-pointer ${
                  currentPage === page
                    ? "bg-slate-800 text-sky-400 border border-slate-700/60"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
                }`}
              >
                {page === "analyze" ? "Verify" : page}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Workspace Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 md:px-8 py-8 md:py-12">
        {currentPage === "analyze" && <AnalyzePage />}
        {currentPage === "history" && <HistoryPage />}
        {currentPage === "about" && <AboutPage />}
      </main>

      {/* Global Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/20 py-8 text-center text-[10px] text-slate-500 font-sans">
        <div className="max-w-6xl mx-auto px-4 space-y-2">
          <p>© {new Date().getFullYear()} Hybrid AI fake content detector platform. All rights reserved.</p>
          <p className="text-[9px] text-slate-600">
            Engineered with FastAPI, EasyOCR, DistilBERT, PyTorch CASIA-ELA, and FaceForensics++ models.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
