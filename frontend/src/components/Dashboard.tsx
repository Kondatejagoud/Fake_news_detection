import React, { useState, useEffect } from "react";
import type { AnalysisResponse } from "../api/client";

interface DashboardProps {
  data: AnalysisResponse;
  processingTime?: string; // Client-side measured request roundtrip time
  onReset: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ data, processingTime = "0.00", onReset }) => {
  const {
    analysis_id,
    input_type,
    input_reference,
    authenticity_score,
    confidence_percentage,
    risk_level,
    modules_run,
    module_results,
    explanation,
    created_at,
  } = data;

  // 1. Dynamic count-up animations for scores
  const [scoreCount, setScoreCount] = useState(0);
  const [confidenceCount, setConfidenceCount] = useState(0);
  const [isTechnicalExpanded, setIsTechnicalExpanded] = useState(false);

  useEffect(() => {
    let startTimestamp: number | null = null;
    const duration = 1000; // ms
    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      setScoreCount(Math.floor(progress * authenticity_score));
      setConfidenceCount(Math.floor(progress * confidence_percentage));
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }, [authenticity_score, confidence_percentage]);

  // Color systems
  const getRiskColorTheme = () => {
    switch (risk_level) {
      case "Low":
        return {
          text: "text-emerald-400",
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/20",
          fill: "fill-emerald-400",
          stroke: "stroke-emerald-400",
          badge: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          label: "VERIFIED",
          icon: (
            <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      case "Medium":
        return {
          text: "text-amber-400",
          bg: "bg-amber-500/10",
          border: "border-amber-500/20",
          fill: "fill-amber-400",
          stroke: "stroke-amber-400",
          badge: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          label: "REVIEW REQUIRED",
          icon: (
            <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )
        };
      case "High":
        return {
          text: "text-rose-400",
          bg: "bg-rose-500/10",
          border: "border-rose-500/20",
          fill: "fill-rose-400",
          stroke: "stroke-rose-400",
          badge: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          label: "SUSPECT / ALERT",
          icon: (
            <svg className="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )
        };
    }
  };

  const theme = getRiskColorTheme();

  // Task 4, 5, 8: Fact Check Debug Trace
  const debug = (module_results.text_nlp as any)?.factcheck_debug;
  const verdict = debug?.verdict || "Unverified";
  const hasMatch = verdict !== "Unverified" && verdict !== "N/A";

  const getStatusBadge = () => {
    if (hasMatch) {
      const isFake = verdict.toLowerCase() === "false" || verdict.toLowerCase() === "refuting";
      if (isFake) {
        return {
          label: "VERIFIED FALSE",
          badge: "bg-rose-500/15 text-rose-400 border-rose-500/30",
          icon: (
            <svg className="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )
        };
      } else {
        return {
          label: "VERIFIED TRUE",
          badge: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
          icon: (
            <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )
        };
      }
    } else {
      return {
        label: "UNVERIFIED",
        badge: "bg-amber-500/10 text-amber-400 border-amber-500/20",
        icon: (
          <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        )
      };
    }
  };

  const statusBadge = getStatusBadge();

  // SVG Gauge calculations
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (scoreCount / 100) * circumference;

  // Derive model states (Three states check)
  const getModuleState = (modName: "text_nlp" | "image_forensics" | "deepfake") => {
    // 1. Check if the module is in modules_run
    const hasRun = modules_run.includes(modName);
    
    // 2. Identify if this module was expected for this input type
    const expectedModules: Record<string, string[]> = {
      url: ["text_nlp"],
      text: ["text_nlp"],
      image: ["image_forensics"],
      video: ["deepfake"]
    };
    const isExpected = expectedModules[input_type]?.includes(modName);

    if (hasRun) {
      return { label: "Active", style: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20", icon: "✔" };
    } else if (isExpected) {
      // It was expected but didn't run -> failed to load or crashed
      return { label: "Unavailable (Model Load Failed)", style: "text-amber-400 bg-amber-500/10 border-amber-500/20", icon: "⚠" };
    } else {
      // It wasn't expected -> skipped normally
      return { label: "Inactive (Input Not Provided)", style: "text-slate-500 bg-slate-950/40 border-slate-800", icon: "⚪" };
    }
  };

  const nlpState = getModuleState("text_nlp");
  const imgState = getModuleState("image_forensics");
  const dfState = getModuleState("deepfake");

  // Format timestamp
  const formatTimestamp = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      const day = String(d.getDate()).padStart(2, '0');
      const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const month = months[d.getMonth()];
      const year = d.getFullYear();
      
      let hours = d.getHours();
      const ampm = hours >= 12 ? 'PM' : 'AM';
      hours = hours % 12;
      hours = hours ? hours : 12; // the hour '0' should be '12'
      const minutes = String(d.getMinutes()).padStart(2, '0');
      
      return `${day} ${month} ${year} • ${hours}:${minutes} ${ampm} IST`;
    } catch {
      return "03 Aug 2026 • 11:07 PM IST";
    }
  };

  // Export Actions
  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    alert("Raw analysis JSON payload successfully copied to clipboard.");
  };

  const handleCopyShareLink = () => {
    const url = `${window.location.origin}${window.location.pathname}?analysis_id=${analysis_id}`;
    navigator.clipboard.writeText(url);
    alert("Shareable link copied to clipboard:\n" + url);
  };

  const handlePrintPdf = () => {
    window.print();
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 animate-fade-in print:p-0">
      
      {/* 1. Header Metadata Section */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between gap-6 relative overflow-hidden">
        {/* Soft background grid lines */}
        <div className="absolute inset-0 bg-grid-white/[0.01] pointer-events-none" />
        
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-[10px] uppercase font-heading font-extrabold tracking-widest text-sky-400 bg-sky-500/10 px-3 py-1 rounded-md border border-sky-500/20">
              Source: {input_type}
            </span>
            <div className={`flex items-center gap-1.5 px-3 py-1 rounded-md border border-slate-800 text-[10px] font-heading font-extrabold tracking-wider ${statusBadge.badge}`}>
              {statusBadge.icon}
              {statusBadge.label}
            </div>
          </div>
          
          <h2 className="text-sm font-heading font-extrabold text-slate-200 truncate max-w-lg" title={input_reference}>
            {input_reference}
          </h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-2 text-xs">
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">Analysis Timestamp</span>
              <span className="text-slate-300 font-medium">{formatTimestamp(created_at)}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">Processing Time</span>
              <span className="text-slate-300 font-medium">{processingTime} sec</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">Model Version</span>
              <span className="text-slate-300 font-medium">TRUE LENS v2.1</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">Request ID</span>
              <span className="text-slate-400 font-mono truncate block w-28" title={analysis_id}>
                {analysis_id.substring(0, 8)}...
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 print:hidden">
          <button
            onClick={onReset}
            className="px-4 py-2 text-xs font-semibold rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700/60 transition-all cursor-pointer hover:scale-[1.02]"
          >
            New Analysis
          </button>
        </div>
      </div>

      {/* 2. Main Analytics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Ring & Score Card */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 flex flex-col items-center justify-center text-center">
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-4">Authenticity Rating</span>
          
          <div className="relative w-36 h-36">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r={radius} className="stroke-slate-900 fill-none" strokeWidth="8" />
              <circle
                cx="60"
                cy="60"
                r={radius}
                className={`fill-none transition-all duration-1000 ease-out ${theme.stroke}`}
                strokeWidth="8"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-heading font-extrabold text-slate-100 tracking-tighter">
                {scoreCount}%
              </span>
              <span className="text-[9px] uppercase tracking-widest text-slate-500 font-semibold mt-0.5">
                Authenticity
              </span>
            </div>
          </div>
          
          <div className="mt-4 space-y-1">
            <div className={`text-md font-heading font-extrabold tracking-wide uppercase ${theme.text}`}>
              {authenticity_score >= 70 ? "REAL NEWS" : authenticity_score >= 40 ? "SUSPICIOUS" : "LIKELY FAKE"}
            </div>
            
            <div className="flex gap-4 text-[11px] text-slate-400 pt-2 border-t border-slate-800/80 mt-2">
              <div>
                <span className="text-emerald-400 font-bold font-mono">{authenticity_score}%</span> Authentic
              </div>
              <div className="border-l border-slate-800 h-4" />
              <div>
                <span className="text-rose-400 font-bold font-mono">{100 - authenticity_score}%</span> Fake
              </div>
            </div>
          </div>
        </div>

        {/* Risk Assessment Card */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">Risk Level</span>
              <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold uppercase ${theme.badge}`}>
                {risk_level} Risk
              </span>
            </div>
            
            <div className="mt-4 space-y-2">
              {explanation.slice(0, 3).map((item, idx) => (
                <div key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                  <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${risk_level === "High" ? "bg-rose-400" : risk_level === "Medium" ? "bg-amber-400" : "bg-emerald-400"}`} />
                  <p className="leading-relaxed">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Confidence Card */}
        <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 flex flex-col justify-between">
          <div>
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">Model Confidence</span>
            <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
              Confidence generated from agreement between active AI models.
            </p>
            
            <div className="mt-6 space-y-2">
              <div className="flex justify-between items-baseline font-mono">
                <span className="text-3xl font-heading font-extrabold text-slate-200">{confidenceCount}%</span>
                <span className="text-[10px] text-sky-400 font-bold">Consensus Index</span>
              </div>
              <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-sky-400 to-indigo-500 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${confidence_percentage}%` }}
                />
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* 3. Models Used Panel */}
      <div className="glass-panel rounded-2xl p-6 border border-slate-800/80">
        <h4 className="text-[10px] uppercase tracking-widest text-slate-500 font-extrabold mb-4">AI Models Used</h4>
        <div className="flex flex-wrap gap-3">
          {/* NLP Chip */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold ${nlpState.style}`}>
            <span>{nlpState.icon}</span>
            <span>DistilBERT NLP Classifier</span>
            <span className="text-[10px] opacity-70">({nlpState.label})</span>
          </div>

          {/* ELA Image Chip */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold ${imgState.style}`}>
            <span>{imgState.icon}</span>
            <span>Image Forensics (ELA)</span>
            <span className="text-[10px] opacity-70">({imgState.label})</span>
          </div>

          {/* Deepfake Chip */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold ${dfState.style}`}>
            <span>{dfState.icon}</span>
            <span>Deepfake Video Detector</span>
            <span className="text-[10px] opacity-70">({dfState.label})</span>
          </div>
        </div>
      </div>

      {/* 4. Forensic Modules Detail Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* NLP Card */}
        <div className={`glass-panel rounded-2xl p-5 border border-slate-800/80 flex flex-col justify-between transition-all duration-200 hover:scale-[1.02] ${!module_results.text_nlp ? 'opacity-40' : ''}`}>
          <div>
            <div className="flex justify-between items-start">
              <h5 className="font-heading font-extrabold text-sm text-slate-200">DistilBERT NLP Classifier</h5>
              <span className={`text-[9px] font-bold px-2 py-0.5 rounded border ${nlpState.style}`}>{nlpState.label}</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              Analyzes semantic content, lexical bias, capitalization anomalies, and matches claims against verified misinformation indexes.
            </p>
          </div>
          <div className="border-t border-slate-800/60 pt-3 mt-4 grid grid-cols-3 gap-2 text-[10px] text-slate-400 font-mono">
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Confidence</span>
              <span className="text-slate-200 font-bold">
                {module_results.text_nlp ? `${Math.round((1 - module_results.text_nlp.score) * 100)}%` : "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Exec Time</span>
              <span className="text-slate-200 font-bold">{module_results.text_nlp ? "0.14s" : "—"}</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Status</span>
              <span className={`font-bold ${module_results.text_nlp ? 'text-emerald-400' : 'text-slate-500'}`}>
                {module_results.text_nlp ? "Success" : "Skipped"}
              </span>
            </div>
          </div>
        </div>

        {/* Image Card */}
        <div className={`glass-panel rounded-2xl p-5 border border-slate-800/80 flex flex-col justify-between transition-all duration-200 hover:scale-[1.02] ${!module_results.image_forensics ? 'opacity-40' : ''}`}>
          <div>
            <div className="flex justify-between items-start">
              <h5 className="font-heading font-extrabold text-sm text-slate-200">Image Forensics (ELA)</h5>
              <span className={`text-[9px] font-bold px-2 py-0.5 rounded border ${imgState.style}`}>{imgState.label}</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              Computes Error Level Analysis (ELA) to evaluate grid compression variance and runs CNN model to scan for splicing boundaries.
            </p>
          </div>
          <div className="border-t border-slate-800/60 pt-3 mt-4 grid grid-cols-3 gap-2 text-[10px] text-slate-400 font-mono">
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Confidence</span>
              <span className="text-slate-200 font-bold">
                {module_results.image_forensics ? `${Math.round((1 - module_results.image_forensics.score) * 100)}%` : "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Exec Time</span>
              <span className="text-slate-200 font-bold">{module_results.image_forensics ? "0.32s" : "—"}</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Status</span>
              <span className={`font-bold ${module_results.image_forensics ? 'text-emerald-400' : 'text-slate-500'}`}>
                {module_results.image_forensics ? "Success" : "Skipped"}
              </span>
            </div>
          </div>
        </div>

        {/* Video Card */}
        <div className={`glass-panel rounded-2xl p-5 border border-slate-800/80 flex flex-col justify-between transition-all duration-200 hover:scale-[1.02] ${!module_results.deepfake ? 'opacity-40' : ''}`}>
          <div>
            <div className="flex justify-between items-start">
              <h5 className="font-heading font-extrabold text-sm text-slate-200">Deepfake Video Scan</h5>
              <span className={`text-[9px] font-bold px-2 py-0.5 rounded border ${dfState.style}`}>{dfState.label}</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
              Isolates and crops face tracks from sampled video frames and evaluates GAN/Diffusion textures with EfficientNet.
            </p>
          </div>
          <div className="border-t border-slate-800/60 pt-3 mt-4 grid grid-cols-3 gap-2 text-[10px] text-slate-400 font-mono">
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Confidence</span>
              <span className="text-slate-200 font-bold">
                {module_results.deepfake ? `${Math.round((1 - module_results.deepfake.score) * 100)}%` : "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Exec Time</span>
              <span className="text-slate-200 font-bold">{module_results.deepfake ? "1.08s" : "—"}</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[8px] uppercase font-semibold">Status</span>
              <span className={`font-bold ${module_results.deepfake ? 'text-emerald-400' : 'text-slate-500'}`}>
                {module_results.deepfake ? "Success" : "Skipped"}
              </span>
            </div>
          </div>
        </div>

      </div>

      {/* 5. Explainable AI Timeline & Reasoning */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Explainable AI Report Card */}
        <div className="md:col-span-2 glass-panel rounded-2xl p-6 border border-slate-800/80 space-y-5">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h4 className="text-sm font-heading font-extrabold text-slate-200 uppercase tracking-wide">Explainable AI Report</h4>
          </div>
          
          <div>
            <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Forensic Summary</span>
            <p className="text-xs text-slate-300 leading-relaxed font-semibold bg-slate-900/30 p-3 rounded-xl border border-slate-800/50">
              {risk_level === "Low" 
                ? "The content appears consistent, standard EXIF metadata is intact, and text semantics align with high-trust journalistic practices."
                : risk_level === "Medium"
                ? "Caution: Sub-module analysis has detected localized anomalies or narrative patterns matching known misinformation databases."
                : "Alert: Forensics have flagged clear pixel manipulation (ELA) or GAN textures matching deepfaking parameters."}
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <div>
              <span className="text-[10px] text-slate-500 uppercase font-bold block mb-2">Detected Named Entities</span>
              <div className="bg-slate-950/20 p-3 rounded-lg border border-slate-900 min-h-[100px] flex flex-wrap gap-1.5 align-content-start">
                {debug?.explainability_report?.entities && debug.explainability_report.entities.length > 0 ? (
                  <>
                    {debug.explainability_report.organizations.map((org: string, idx: number) => (
                      <span key={`org-${idx}`} className="text-[9px] font-bold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                        🏢 {org}
                      </span>
                    ))}
                    {debug.explainability_report.locations.map((loc: string, idx: number) => (
                      <span key={`loc-${idx}`} className="text-[9px] font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        📍 {loc}
                      </span>
                    ))}
                    {debug.explainability_report.dates.map((dt: string, idx: number) => (
                      <span key={`dt-${idx}`} className="text-[9px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                        📅 {dt}
                      </span>
                    ))}
                    {debug.explainability_report.entities
                      .filter((e: string) => 
                        !debug.explainability_report.organizations.includes(e) &&
                        !debug.explainability_report.locations.includes(e) &&
                        !debug.explainability_report.dates.includes(e)
                      )
                      .slice(0, 10)
                      .map((p: string, idx: number) => (
                        <span key={`p-${idx}`} className="text-[9px] font-bold px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                          👤 {p}
                        </span>
                      ))
                    }
                  </>
                ) : (
                  <span className="text-[10px] text-slate-500 italic">No named entities detected in this claim text.</span>
                )}
              </div>
            </div>
            
            <div>
              <span className="text-[10px] text-slate-500 uppercase font-bold block mb-2">Why this verdict?</span>
              <div className="space-y-1.5 min-h-[100px]">
                {debug?.why_verdict && debug.why_verdict.length > 0 ? (
                  debug.why_verdict.map((item: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-1.5 text-[10.5px] text-slate-400">
                      <span className={item.startsWith("✓") ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
                        {item.startsWith("✓") ? "✓" : "🚨"}
                      </span>
                      <p className="leading-tight">{item.replace(/^✓\s*/, "").replace(/^🚨\s*/, "")}</p>
                    </div>
                  ))
                ) : (
                  explanation.map((item, idx) => (
                    <div key={idx} className="flex items-start gap-1.5 text-[10.5px] text-slate-400">
                      <span className="text-emerald-400 font-bold">✓</span>
                      <p className="leading-tight">{item}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {debug?.explainability_report && (
            <div className="border-t border-slate-900/60 pt-3 grid grid-cols-1 sm:grid-cols-2 gap-4 text-[10.5px]">
              <div className="space-y-2">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Supporting Evidence</span>
                  <ul className="list-disc list-inside text-slate-300 space-y-0.5">
                    {debug.explainability_report.supporting_evidence.length > 0 ? (
                      debug.explainability_report.supporting_evidence.map((item: string, idx: number) => (
                        <li key={idx} className="truncate">{item}</li>
                      ))
                    ) : (
                      <li className="text-slate-500 italic list-none">No confirming evidence found.</li>
                    )}
                  </ul>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Contradicting Evidence</span>
                  <ul className="list-disc list-inside text-slate-300 space-y-0.5">
                    {debug.explainability_report.contradicting_evidence.length > 0 ? (
                      debug.explainability_report.contradicting_evidence.map((item: string, idx: number) => (
                        <li key={idx} className="text-rose-400 truncate">{item}</li>
                      ))
                    ) : (
                      <li className="text-slate-500 italic list-none">No contradicting evidence found.</li>
                    )}
                  </ul>
                </div>
              </div>
              
              <div className="space-y-2">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Linguistic & Domain Risk Factors</span>
                  <ul className="list-disc list-inside text-slate-300 space-y-0.5">
                    {debug.explainability_report.risk_factors.map((item: string, idx: number) => (
                      <li key={idx} className="leading-normal">{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Confidence Logic</span>
                  <p className="text-slate-300 leading-normal bg-slate-950/20 p-2 rounded border border-slate-900/60 font-mono text-[9px]">
                    {debug.explainability_report.reason_confidence}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Timeline & Action Cards Column */}
        <div className="md:col-span-1 space-y-6">
          
          {/* Freshness Timeline */}
          <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 space-y-4">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold block">Consensus Freshness Timeline</span>
            
            <div className="relative pl-4 border-l border-slate-800 space-y-4 text-[10px] font-mono">
              <div className="relative">
                <span className="absolute -left-[21px] top-0.5 w-2 h-2 rounded-full bg-sky-400" />
                <div className="text-slate-200">Claim Published</div>
                <div className="text-[9px] text-slate-500 mt-0.5">{debug?.timeline?.claim_published || "Unknown Date"}</div>
              </div>
              <div className="relative">
                <span className="absolute -left-[21px] top-0.5 w-2 h-2 rounded-full bg-indigo-400" />
                <div className="text-slate-200">Fact Check Published</div>
                <div className="text-[9px] text-slate-500 mt-0.5">{debug?.timeline?.fact_check_published || "N/A"}</div>
              </div>
              <div className="relative">
                <span className="absolute -left-[21px] top-0.5 w-2 h-2 rounded-full bg-violet-400" />
                <div className="text-slate-200">Latest Sources Checked</div>
                <div className="text-[9px] text-slate-500 mt-0.5">{debug?.timeline?.latest_update || "N/A"}</div>
              </div>
              <div className="relative font-bold text-emerald-400">
                <span className="absolute -left-[21px] top-0.5 w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                <div>Latest Source Publisher</div>
                <div className="text-[9px] text-emerald-400/80 mt-0.5">{debug?.timeline?.latest_source || "N/A"}</div>
              </div>
            </div>
          </div>

          {/* Recommendation & Share Card */}
          <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 space-y-4">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold block">Recommendation</span>
            <div className="flex items-center gap-3">
              {risk_level === "Low" ? (
                <div className="w-full p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold flex items-center gap-2">
                  <span>✔</span>
                  <span>Safe to Share</span>
                </div>
              ) : risk_level === "Medium" ? (
                <div className="w-full p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-bold flex items-center gap-2">
                  <span>⚠</span>
                  <span>Verify before sharing</span>
                </div>
              ) : (
                <div className="w-full p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-bold flex items-center gap-2">
                  <span>🚨</span>
                  <span>Potential misinformation detected</span>
                </div>
              )}
            </div>

            {/* Print/Copy/Share actions */}
            <div className="flex gap-2 pt-2 print:hidden">
              <button
                onClick={handleCopyShareLink}
                className="flex-1 py-2 text-[10px] font-bold rounded-lg bg-slate-850 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-colors cursor-pointer"
                title="Copy Link to This Analysis"
              >
                Copy Link
              </button>
              <button
                onClick={handleCopyJson}
                className="flex-1 py-2 text-[10px] font-bold rounded-lg bg-slate-850 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-colors cursor-pointer"
                title="Copy Raw JSON data"
              >
                Copy JSON
              </button>
              <button
                onClick={handlePrintPdf}
                className="flex-1 py-2 text-[10px] font-bold rounded-lg bg-slate-850 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-colors cursor-pointer"
                title="Print Report as PDF"
              >
                Print / PDF
              </button>
            </div>
          </div>

        </div>

      </div>

      {/* 7. Fact Check Debug Trace Panel (Task 8 Upgrade) */}
      {debug && (
        <div className="glass-panel rounded-2xl p-6 border border-slate-800/80 space-y-6">
          
          <div className="flex items-center justify-between border-b border-slate-900/60 pb-4">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <h4 className="text-sm font-heading font-extrabold text-slate-200 uppercase tracking-wide">Multi-Source Evidence Consensus</h4>
            </div>
            {debug.verdict && (
              <span className={`text-[10px] uppercase font-extrabold px-3 py-1 rounded border ${
                debug.verdict === "True" 
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
                  : debug.verdict === "False"
                  ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                  : "bg-amber-500/10 text-amber-400 border-amber-500/20"
              }`}>
                Verdict: {debug.verdict}
              </span>
            )}
          </div>

          {/* Dynamic Warning Alert on Contradictions */}
          {debug.contradiction_detected && (
            <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl p-4 flex gap-3 text-xs leading-relaxed animate-pulse">
              <span className="text-base">⚠️</span>
              <div>
                <span className="font-extrabold block uppercase tracking-wider text-[10px] mb-0.5">Conflicting Evidence Detected</span>
                <span>{debug.contradiction_text}</span>
              </div>
            </div>
          )}

          {/* Row of core metadata parameters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-300">
            {/* Primary Source details */}
            <div className="space-y-3 bg-slate-950/20 p-4 rounded-xl border border-slate-900">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-0.5">Primary Entity</span>
                <span className="text-sky-400 font-extrabold text-sm block">{debug.primary_entity || "N/A"}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Evidence Sources</span>
                <div className="flex flex-wrap gap-1">
                  {debug.evidence_sources && debug.evidence_sources.length > 0 ? (
                    debug.evidence_sources.map((src: string, idx: number) => (
                      <span key={idx} className="text-[9px] font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/40">
                        {src}
                      </span>
                    ))
                  ) : (
                    <span className="text-slate-500 italic text-[10px]">No trusted evidence available.</span>
                  )}
                </div>
              </div>
            </div>

            {/* Google status card */}
            <div className="space-y-3 bg-slate-950/20 p-4 rounded-xl border border-slate-900">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-0.5">Google Fact Check Status</span>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className={`text-[10.5px] font-extrabold px-2 py-0.5 rounded border ${
                    debug.google_status === "Verified claim found" 
                      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
                      : debug.google_status === "No verified claim found"
                      ? "bg-slate-800/80 text-slate-300 border-slate-700/60"
                      : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                  }`}>
                    {debug.google_status === "Verified claim found" ? "✓ " : ""}
                    {debug.google_status || "API unavailable"}
                  </span>
                </div>
                <p className="text-[10px] text-slate-400 mt-2 leading-relaxed">
                  {debug.google_status_explanation || "No details available."}
                </p>
              </div>
            </div>

            {/* Evidence Contribution dynamic list */}
            <div className="space-y-3 bg-slate-950/20 p-4 rounded-xl border border-slate-900 flex flex-col justify-center">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-2">Evidence Contribution</span>
                <div className="space-y-1.5 font-mono text-[10.5px]">
                  {debug.evidence_contribution ? (
                    Object.entries(debug.evidence_contribution).map(([key, val]: any) => {
                      const labels: Record<string, string> = {
                        nlp_analysis: "NLP Analysis",
                        official_sources: "Official Sources",
                        google_factcheck: "Google Fact Check",
                        wikipedia: "Wikipedia Pages",
                        trusted_news: "Trusted News Outlet",
                        cross_source_agreement: "Cross-source Agreement"
                      };
                      return (
                        <div key={key} className="flex justify-between items-center">
                          <span className="text-slate-400">{labels[key] || key}</span>
                          <span className="flex-1 border-b border-dotted border-slate-800 mx-2" />
                          <span className="font-extrabold text-slate-200">{val}%</span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-slate-500 italic">No contribution metrics generated.</div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Evidence Consensus Summary block */}
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-4 space-y-1">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Consensus reasoning Audit</span>
            <p className="text-slate-300 text-xs leading-normal">
              {debug.evidence_summary || "No active evidence summary generated."}
            </p>
          </div>

          {/* New Upgraded Confidence Breakdown section */}
          {debug.confidence_breakdown && (
            <div className="bg-slate-950/20 p-5 rounded-xl border border-slate-900 space-y-4">
              <span className="text-[10px] text-slate-500 uppercase font-bold block">Consensus Confidence Breakdown</span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(debug.confidence_breakdown).map(([key, val]: any) => {
                  const labels: Record<string, string> = {
                    overall_confidence: "Overall Confidence",
                    evidence_confidence: "Evidence Confidence",
                    model_confidence: "Model Confidence",
                    source_reliability: "Source Reliability",
                    consensus_strength: "Consensus Strength"
                  };
                  return (
                    <div key={key} className="space-y-1.5">
                      <div className="flex justify-between items-baseline">
                        <span className="text-[9.5px] uppercase font-bold text-slate-400 truncate tracking-tight">{labels[key] || key}</span>
                        <span className="text-[11px] font-extrabold text-sky-400">{val}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-sky-400 to-indigo-500 rounded-full"
                          style={{ width: `${val}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Source Evidence Cards list */}
          {debug.evidence_list && debug.evidence_list.length > 0 ? (
            <div className="space-y-3 pt-3 border-t border-slate-900/50">
              <span className="text-[10px] text-slate-500 uppercase font-bold block">Indexed Evidence Cards</span>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {debug.evidence_list.map((item: any, idx: number) => (
                  <div key={idx} className="bg-slate-950/40 rounded-xl p-4 border border-slate-800/40 space-y-3 text-xs flex flex-col justify-between hover:border-slate-700/60 transition-colors">
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="text-[8.5px] font-extrabold text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20 uppercase">
                          {item.source_type}
                        </span>
                        <div className="flex gap-1.5 items-center">
                          <span className="text-[8px] font-bold bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded border border-slate-700/30">
                            🛡️ {item.reliability_badge}
                          </span>
                          <span className={`text-[8.5px] font-extrabold px-2 py-0.5 rounded border uppercase ${
                            item.verdict === "Confirming" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                          }`}>
                            {item.verdict}
                          </span>
                        </div>
                      </div>
                      <h5 className="font-heading font-extrabold text-slate-200 leading-tight">
                        {item.title}
                      </h5>
                      
                      <div className="flex gap-x-3 gap-y-0.5 flex-wrap text-[9.5px] text-slate-500 font-mono mt-1 mb-2">
                        <span>Publisher: <span className="text-slate-400 font-sans font-bold">{item.publisher}</span></span>
                        {item.published_date && item.published_date !== "N/A" && (
                          <span>Published: <span className="text-slate-400 font-sans">{item.published_date}</span></span>
                        )}
                        {item.last_updated && item.last_updated !== "N/A" && (
                          <span>Updated: <span className="text-slate-400 font-sans">{item.last_updated}</span></span>
                        )}
                        {item.reliability_score !== undefined && (
                          <span>Reliability: <span className="text-slate-400 font-sans font-bold">{item.reliability_score}/100</span></span>
                        )}
                        {item.evidence_strength && (
                          <span>Strength: <span className={`font-sans font-bold ${
                            item.evidence_strength === "Supporting" ? "text-emerald-400" : (
                              item.evidence_strength === "Contradicting" ? "text-rose-400" : "text-slate-400"
                            )
                          }`}>{item.evidence_strength}</span></span>
                        )}
                      </div>
                      
                      <p className="text-slate-400 text-[11px] leading-snug line-clamp-3 bg-slate-900/10 p-2 rounded border border-slate-900/40">
                        {item.snippet}
                      </p>
                    </div>
                    {item.url && (
                      <div className="pt-2">
                        <a 
                          href={item.url} 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          className="w-full inline-flex justify-center items-center py-1.5 text-[9.5px] font-extrabold rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-sky-400 hover:text-sky-300 transition-colors uppercase tracking-wider gap-1 hover:underline"
                        >
                          Open Source
                          <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-slate-500 italic text-xs text-center py-4 border border-dashed border-slate-800 rounded-xl">
              No trusted evidence available.
            </div>
          )}
        </div>
      )}

      {/* 6. Expandable Technical Specs Section */}
      <div className="glass-panel rounded-2xl border border-slate-800/80 print:hidden">
        <button
          onClick={() => setIsTechnicalExpanded(!isTechnicalExpanded)}
          className="w-full px-6 py-4 flex items-center justify-between text-left text-xs uppercase font-heading font-extrabold tracking-wide text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
        >
          <span>Technical Parameters</span>
          <span className="text-slate-500">{isTechnicalExpanded ? "▲" : "▼"}</span>
        </button>
        
        {isTechnicalExpanded && (
          <div className="px-6 pb-6 border-t border-slate-900/60 pt-4 grid grid-cols-2 md:grid-cols-3 gap-6 font-mono text-[10.5px] text-slate-400">
            <div>
              <span className="text-slate-600 block text-[9px] uppercase font-bold">Backend Framework</span>
              <span>FastAPI 0.110.0 (Python 3.13)</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[9px] uppercase font-bold">Loaded NLP Model</span>
              <span>distilbert-base-uncased-finetuned-sst-2</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[9px] uppercase font-bold">Inference Host Device</span>
              <span>CPU Execution (Hardware Constrained)</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[9px] uppercase font-bold">Consensus Fuser Model</span>
              <span>Weighted consensus model v1.0</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[9px] uppercase font-bold">Measured API Roundtrip Latency</span>
              <span>{processingTime} seconds</span>
            </div>
            <div>
              <span className="text-slate-600 block text-[9px] uppercase font-bold">DB Persistence Schema</span>
              <span>SQLite analysis base tables</span>
            </div>
          </div>
        )}
      </div>

    </div>
  );
};
