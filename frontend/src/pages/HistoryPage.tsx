import React, { useEffect, useState } from "react";
import type { AnalysisResponse } from "../api/client";
import { Dashboard } from "../components/Dashboard";

export const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<AnalysisResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Selected history item to display in the Dashboard view
  const [selectedItem, setSelectedItem] = useState<AnalysisResponse | null>(null);

  const fetchHistory = () => {
    setLoading(true);
    setError(null);
    try {
      const existing = localStorage.getItem("hybrid_detector_history");
      const historyList: AnalysisResponse[] = existing ? JSON.parse(existing) : [];
      setHistory(historyList);
    } catch (err: any) {
      setError("Failed to load local analysis history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const getScoreBadgeClass = (score: number) => {
    if (score >= 70) return "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (score >= 40) return "bg-amber-50 text-amber-700 border-amber-200";
    return "bg-rose-50 text-rose-700 border-rose-200";
  };

  const getRiskBadgeClass = (risk: string) => {
    if (risk === "Low") return "bg-emerald-50 text-emerald-700 border border-emerald-200";
    if (risk === "Medium") return "bg-amber-50 text-amber-700 border border-amber-200";
    return "bg-rose-50 text-rose-700 border border-rose-200";
  };

  if (selectedItem) {
    return (
      <div className="space-y-4">
        <button
          onClick={() => {
            setSelectedItem(null);
            fetchHistory(); // Refresh history list
          }}
          className="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 transition-all cursor-pointer mb-2"
        >
          &larr; Back to History List
        </button>
        <Dashboard data={selectedItem} onReset={() => setSelectedItem(null)} />
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-fade-in">
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-heading font-extrabold text-slate-900">
          Analysis Records
        </h1>
        <p className="text-slate-600 max-w-xl mx-auto text-sm leading-relaxed">
          Review the list of recently executed checks and their corresponding risk outcomes.
        </p>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-16 space-y-4">
          <div className="w-12 h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin" />
          <p className="text-sm text-slate-500">Loading historical audit trails...</p>
        </div>
      ) : error ? (
        <div className="bg-white rounded-3xl p-8 text-center space-y-4 border border-slate-200 shadow-xs">
          <p className="text-rose-600 text-sm font-semibold">{error}</p>
          <button
            onClick={fetchHistory}
            className="px-5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 font-medium transition-colors border border-slate-300 cursor-pointer"
          >
            Retry Fetch
          </button>
        </div>
      ) : history.length === 0 ? (
        <div className="bg-white rounded-3xl p-12 text-center space-y-4 border border-slate-200 shadow-xs">
          <p className="text-slate-600 text-sm">No analysis reports have been logged yet.</p>
          <p className="text-xs text-slate-400">Run a forensics check to see results here.</p>
        </div>
      ) : (
        <div className="bg-white rounded-3xl overflow-hidden border border-slate-200 shadow-xs">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50/80 border-b border-slate-200 text-[10px] uppercase tracking-wider text-slate-600 font-heading font-bold">
                  <th className="px-6 py-4">Source Type</th>
                  <th className="px-6 py-4">Source Reference</th>
                  <th className="px-6 py-4 text-center">Score</th>
                  <th className="px-6 py-4 text-center">Risk Level</th>
                  <th className="px-6 py-4 text-right">Scanned At</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.map((item) => (
                  <tr
                    key={item.analysis_id}
                    onClick={() => setSelectedItem(item)}
                    className="hover:bg-slate-50/80 transition-colors cursor-pointer group"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
                        {item.input_type}
                      </span>
                    </td>
                    <td className="px-6 py-4 max-w-xs truncate font-medium text-slate-800 group-hover:text-blue-600 transition-colors">
                      {item.input_reference}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <span className={`px-2.5 py-1 rounded-lg border font-mono font-bold text-xs ${getScoreBadgeClass(item.authenticity_score)}`}>
                        {item.authenticity_score}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-center">
                      <span className={`px-2.5 py-1 rounded-md text-[10px] uppercase font-bold tracking-wider ${getRiskBadgeClass(item.risk_level)}`}>
                        {item.risk_level}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-xs text-slate-500">
                      {new Date(item.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

