import React, { useState, useEffect } from "react";
import { analyzeTextOrUrl, analyzeMediaFile, getAnalysisResult } from "../api/client";
import type { AnalysisResponse } from "../api/client";
import { Dashboard } from "../components/Dashboard";

type TabType = "url" | "text" | "image" | "video";

export const saveToLocalHistory = (res: AnalysisResponse) => {
  try {
    const existing = localStorage.getItem("hybrid_detector_history");
    const historyList: AnalysisResponse[] = existing ? JSON.parse(existing) : [];
    if (!historyList.some((item) => item.analysis_id === res.analysis_id)) {
      const updated = [res, ...historyList].slice(0, 50);
      localStorage.setItem("hybrid_detector_history", JSON.stringify(updated));
    }
  } catch (e) {
    console.error("Failed to save to local history:", e);
  }
};

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
          saveToLocalHistory(res);
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
      let res: AnalysisResponse;
      if (activeTab === "url") {
        if (!inputValue.trim()) throw new Error("Please enter a valid URL.");
        
        setLoadingStep("Connecting to source and scraping body content...");
        setTimeout(() => setLoadingStep("Analyzing text patterns & checking claim databases..."), 1500);
        setTimeout(() => setLoadingStep("Fusing module outputs and generating explanation report..."), 3000);
        
        res = await analyzeTextOrUrl("url", inputValue);
      } 
      else if (activeTab === "text") {
        if (!inputValue.trim()) throw new Error("Please write some text to analyze.");
        
        setLoadingStep("Parsing claim semantics...");
        setTimeout(() => setLoadingStep("Checking database for registered propaganda patterns..."), 1200);
        setTimeout(() => setLoadingStep("Evaluating syntax style and computing risk consensus..."), 2400);
        
        res = await analyzeTextOrUrl("text", inputValue);
      } 
      else if (activeTab === "image") {
        if (!selectedFile) throw new Error("Please upload an image file.");
        
        setLoadingStep("Reading image headers and metadata...");
        setTimeout(() => setLoadingStep("Computing JPEG Error Level Analysis (ELA)..."), 1500);
        setTimeout(() => setLoadingStep("Running neural splicing detector models..."), 3000);
        setTimeout(() => setLoadingStep("Extracting OCR textual contents (if any)..."), 4500);
        
        res = await analyzeMediaFile("image", selectedFile);
      } 
      else if (activeTab === "video") {
        if (!selectedFile) throw new Error("Please upload a video file.");
        
        setLoadingStep("Sampling frames (1 frame per second)...");
        setTimeout(() => setLoadingStep("Detecting and cropping faces (MTCNN/Haar)..."), 2000);
        setTimeout(() => setLoadingStep("Scanning face crops using deepfake classifier..."), 4500);
        setTimeout(() => setLoadingStep("Assessing boundary gradients and Laplacian blur ratios..."), 7000);
        setTimeout(() => setLoadingStep("Fusing consensus score..."), 9000);
        
        res = await analyzeMediaFile("video", selectedFile);
      } else {
        throw new Error("Invalid tab selection.");
      }

      const duration = ((performance.now() - startTime) / 1000).toFixed(2);
      setProcessingDuration(duration);
      setResult(res);
      saveToLocalHistory(res);
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
        <h1 className="text-4xl md:text-5xl font-heading font-extrabold tracking-tight text-slate-900">
          Multimodal Fake Content Detection
        </h1>
        <p className="text-slate-600 max-w-xl mx-auto text-sm md:text-base leading-relaxed">
          Verify digital authenticity by pasting a URL, typing out text, or uploading media files. Driven by OCR, NLP, and Computer Vision.
        </p>
      </div>

      {!result && !loading ? (
        <div className="max-w-2xl mx-auto bg-white rounded-3xl p-6 md:p-8 space-y-6 shadow-sm border border-slate-200/80 print:hidden">
          {/* Tab Navigation */}
          <div className="flex border border-slate-200/80 gap-1 p-1 bg-slate-50/80 rounded-xl">
            {(["url", "text", "image", "video"] as TabType[]).map((tab) => (
              <button
                key={tab}
                onClick={() => {
                  setActiveTab(tab);
                  setError(null);
                }}
                className={`flex-1 py-2.5 rounded-lg text-xs md:text-sm font-heading font-semibold capitalize transition-all cursor-pointer ${
                  activeTab === tab
                    ? "bg-blue-100/80 text-blue-600 shadow-2xs border border-blue-200/60 font-bold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/80"
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
                <label className="text-xs uppercase font-heading font-semibold text-slate-500 tracking-wider">Article/Post URL</label>
                <input
                  type="url"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="https://example.com/social-post-or-news-article"
                  className="w-full px-4 py-3 rounded-xl bg-white border border-slate-200 text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 placeholder-slate-400 transition-all font-sans text-sm shadow-2xs"
                  required
                />
              </div>
            )}

            {activeTab === "text" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-500 tracking-wider">Statement Content</label>
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Paste article paragraphs, claims, or text blocks here to evaluate writing style, sensationalism, and check claim databases..."
                  rows={6}
                  className="w-full px-4 py-3 rounded-xl bg-white border border-slate-200 text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 placeholder-slate-400 transition-all font-sans text-sm resize-none shadow-2xs"
                  required
                />
              </div>
            )}

            {activeTab === "image" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-500 tracking-wider">Upload Image File</label>
                <div className="relative border-2 border-dashed border-slate-300 rounded-2xl p-8 hover:border-blue-500/60 hover:bg-blue-50/30 transition-all text-center flex flex-col items-center justify-center bg-slate-50/50">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="space-y-3">
                    <div className="text-slate-600 text-sm">
                      {selectedFile ? (
                        <span className="text-blue-600 font-semibold">{selectedFile.name}</span>
                      ) : (
                        <span>Drag &amp; drop or <span className="text-blue-600 underline font-semibold">browse</span> image file</span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-400">Supports PNG, JPG, JPEG (Max size: 15MB)</p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "video" && (
              <div className="space-y-2">
                <label className="text-xs uppercase font-heading font-semibold text-slate-500 tracking-wider">Upload Video File</label>
                <div className="relative border-2 border-dashed border-slate-300 rounded-2xl p-8 hover:border-blue-500/60 hover:bg-blue-50/30 transition-all text-center flex flex-col items-center justify-center bg-slate-50/50">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="space-y-3">
                    <div className="text-slate-600 text-sm">
                      {selectedFile ? (
                        <span className="text-blue-600 font-semibold">{selectedFile.name}</span>
                      ) : (
                        <span>Drag &amp; drop or <span className="text-blue-600 underline font-semibold">browse</span> video file</span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-400">Supports MP4, AVI, MOV (Max size: 15MB)</p>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs md:text-sm">
                <strong>Error: </strong> {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-heading font-semibold text-sm transition-all cursor-pointer shadow-md shadow-blue-500/15 hover:shadow-lg hover:shadow-blue-500/25"
            >
              Run Forensics Check
            </button>
          </form>
        </div>
      ) : null}

      {/* Loading view */}
      {loading && (
        <div className="max-w-md mx-auto bg-white rounded-3xl p-8 flex flex-col items-center justify-center text-center space-y-6 my-12 shadow-sm border border-slate-200 animate-pulse print:hidden">
          <div className="w-16 h-16 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin" />
          <div className="space-y-2">
            <h3 className="font-heading font-bold text-slate-900 text-lg">Analyzing Authenticity</h3>
            <p className="text-xs text-blue-600 font-semibold h-4 transition-all duration-300">{loadingStep}</p>
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
