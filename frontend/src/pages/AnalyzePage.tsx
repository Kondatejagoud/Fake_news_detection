import React, { useState, useEffect } from "react";
import { analyzeTextOrUrl, analyzeMediaFile, getAnalysisResult } from "../api/client";
import type { AnalysisResponse } from "../api/client";
import { Dashboard } from "../components/Dashboard";

type TabType = "url" | "text" | "image" | "video";

export const AnalyzePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>("url");
  const [inputValue, setInputValue] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // Loading & State variables
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [processingDuration, setProcessingDuration] = useState("0.00");

  // Load shared analysis from URL parameter on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sharedId = params.get("analysis_id");
    if (sharedId) {
      const loadSharedReport = async () => {
        setLoading(true);
        setLoadingStep("Loading archived AI forensics report...");
        try {
          const startTime = performance.now();
          const res = await getAnalysisResult(sharedId);
          const endTime = performance.now();
          setProcessingDuration(((endTime - startTime) / 1000).toFixed(2));
          setResult(res);
        } catch (err: any) {
          setError(err.message || "Failed to load shared analysis record.");
        } finally {
          setLoading(false);
          setLoadingStep("");
        }
      };
      loadSharedReport();
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const maxMb = 15;
      if (file.size > maxMb * 1024 * 1024) {
        setError(`File size exceeds the maximum limit of ${maxMb}MB to prevent container out-of-memory errors.`);
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const executeAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    setProcessingDuration("0.00");

    const startTime = performance.now();

    try {
      if (activeTab === "url") {
        if (!inputValue.trim()) throw new Error("Please enter a valid URL.");
        
        setLoadingStep("Connecting to source and scraping body content...");
        setTimeout(() => setLoadingStep("Analyzing text patterns & checking claim databases..."), 1500);
        setTimeout(() => setLoadingStep("Fusing module outputs and generating explanation report..."), 3000);
        
        const res = await analyzeTextOrUrl("url", inputValue);
        const duration = ((performance.now() - startTime) / 1000).toFixed(2);
        setProcessingDuration(duration);
        setResult(res);
      } 
      else if (activeTab === "text") {
        if (!inputValue.trim()) throw new Error("Please write some text to analyze.");
        
        setLoadingStep("Parsing claim semantics...");
        setTimeout(() => setLoadingStep("Checking database for registered propaganda patterns..."), 1200);
        setTimeout(() => setLoadingStep("Evaluating syntax style and computing risk consensus..."), 2400);
        
        const res = await analyzeTextOrUrl("text", inputValue);
        const duration = ((performance.now() - startTime) / 1000).toFixed(2);
        setProcessingDuration(duration);
        setResult(res);
      } 
      else if (activeTab === "image") {
        if (!selectedFile) throw new Error("Please upload an image file.");
        
        setLoadingStep("Reading image headers and metadata...");
        setTimeout(() => setLoadingStep("Computing JPEG Error Level Analysis (ELA)..."), 1500);
        setTimeout(() => setLoadingStep("Running neural splicing detector models..."), 3000);
        setTimeout(() => setLoadingStep("Extracting OCR textual contents (if any)..."), 4500);
        
        const res = await analyzeMediaFile("image", selectedFile);
        const duration = ((performance.now() - startTime) / 1000).toFixed(2);
        setProcessingDuration(duration);
        setResult(res);
      } 
      else if (activeTab === "video") {
        if (!selectedFile) throw new Error("Please upload a video file.");
        
        setLoadingStep("Sampling frames (1 frame per second)...");
        setTimeout(() => setLoadingStep("Detecting and cropping faces (MTCNN/Haar)..."), 2000);
        setTimeout(() => setLoadingStep("Scanning face crops using deepfake classifier..."), 4500);
        setTimeout(() => setLoadingStep("Assessing boundary gradients and Laplacian blur ratios..."), 7000);
        setTimeout(() => setLoadingStep("Fusing consensus score..."), 9000);
        
        const res = await analyzeMediaFile("video", selectedFile);
        const duration = ((performance.now() - startTime) / 1000).toFixed(2);
        setProcessingDuration(duration);
        setResult(res);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred during content analysis.");
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const handleReset = () => {
    setResult(null);
    setInputValue("");
    setSelectedFile(null);
    setError(null);
    setProcessingDuration("0.00");
    // Clean up URL query parameters
    window.history.replaceState({}, document.title, window.location.pathname);
  };

  return (
    <div className="space-y-8">
      {/* Title block */}
      <div className="text-center space-y-3 print:hidden">
        <h1 className="text-4xl md:text-5xl font-heading font-extrabold tracking-tight bg-gradient-to-r from-sky-400 via-indigo-400 to-rose-400 bg-clip-text text-transparent">
          Multimodal Fake Content Detection
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto text-sm md:text-base leading-relaxed">
          Verify digital authenticity by pasting a URL, typing out text, or uploading media files. Driven by OCR, NLP, and Computer Vision.
        </p>
      </div>

      {!result && !loading ? (
        <div className="max-w-2xl mx-auto glass-panel rounded-3xl p-6 md:p-8 space-y-6 print:hidden">
          {/* Tab Navigation */}
          <div className="flex border-b border-slate-800 gap-1 p-1 bg-slate-950/40 rounded-xl">
            {(["url", "text", "image", "video"] as TabType[]).map((tab) => (
              <button
                key={tab}
                onClick={() => {
                  setActiveTab(tab);
                  setError(null);
                }}
                className={`flex-1 py-2.5 rounded-lg text-xs md:text-sm font-heading font-semibold capitalize transition-all cursor-pointer ${
                  activeTab === tab
                    ? "bg-slate-800 text-sky-400 shadow-md"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab === "url" ? "Verify URL" : tab}
              </button>
            ))}
          </div>

          {/* Form Content */}
          <form onSubmit={executeAnalysis} className="space-y-6">
            {activeTab === "url" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-400">Article/Post URL</label>
                <input
                  type="url"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="https://example.com/social-post-or-news-article"
                  className="w-full px-4 py-3 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 placeholder-slate-600 transition-all font-sans text-sm"
                  required
                />
              </div>
            )}

            {activeTab === "text" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-400">Statement Content</label>
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Paste article paragraphs, claims, or text blocks here to evaluate writing style, sensationalism, and check claim databases..."
                  rows={6}
                  className="w-full px-4 py-3 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 placeholder-slate-600 transition-all font-sans text-sm resize-none"
                  required
                />
              </div>
            )}

            {activeTab === "image" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-400">Upload Image File</label>
                <div className="relative border-2 border-dashed border-slate-800 rounded-2xl p-8 hover:border-sky-500/50 transition-all text-center flex flex-col items-center justify-center bg-slate-950/20">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="space-y-3">
                    <div className="text-slate-400 text-sm">
                      {selectedFile ? (
                        <span className="text-sky-400 font-medium">{selectedFile.name}</span>
                      ) : (
                        <span>Drag & drop or <span className="text-sky-400 underline">browse</span> image file</span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-500">Supports PNG, JPG, JPEG (Max size: 15MB)</p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "video" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-400">Upload Video File</label>
                <div className="relative border-2 border-dashed border-slate-800 rounded-2xl p-8 hover:border-sky-500/50 transition-all text-center flex flex-col items-center justify-center bg-slate-950/20">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="space-y-3">
                    <div className="text-slate-400 text-sm">
                      {selectedFile ? (
                        <span className="text-sky-400 font-medium">{selectedFile.name}</span>
                      ) : (
                        <span>Drag & drop or <span className="text-sky-400 underline">browse</span> video file</span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-500">Supports MP4, AVI, MOV (Max size: 15MB)</p>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs md:text-sm">
                <strong>Error: </strong> {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full py-3.5 rounded-xl bg-gradient-to-r from-sky-500 via-sky-600 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-heading font-semibold text-sm transition-all cursor-pointer shadow-lg shadow-sky-500/10"
            >
              Run Forensics Check
            </button>
          </form>
        </div>
      ) : null}

      {/* Loading view */}
      {loading && (
        <div className="max-w-md mx-auto glass-panel rounded-3xl p-8 flex flex-col items-center justify-center text-center space-y-6 my-12 animate-pulse print:hidden">
          <div className="w-16 h-16 border-4 border-slate-800 border-t-sky-500 rounded-full animate-spin" />
          <div className="space-y-2">
            <h3 className="font-heading font-bold text-slate-200 text-lg">Analyzing Authenticity</h3>
            <p className="text-xs text-sky-400 font-semibold h-4 transition-all duration-300">{loadingStep}</p>
          </div>
          <p className="text-[10px] text-slate-500 leading-relaxed">
            Running OCR pipelines, NLP claim validations, and digital forensics. This may take up to a minute depending on file size.
          </p>
        </div>
      )}

      {/* Results View */}
      {result && !loading && (
        <Dashboard data={result} processingTime={processingDuration} onReset={handleReset} />
      )}
    </div>
  );
};
