// API Client for the Hybrid Multimodal Fake Content Detection application.

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export interface AnalysisResponse {
  analysis_id: string;
  input_type: "url" | "text" | "image" | "video";
  input_reference: string;
  authenticity_score: number;
  confidence_percentage: number;
  risk_level: "Low" | "Medium" | "High";
  modules_run: string[];
  module_results: {
    text_nlp?: {
      score: number;
      flagged_claims: string[];
    };
    image_forensics?: {
      score: number;
      manipulated_regions: number;
    };
    deepfake?: {
      score: number;
      faces_detected: number;
    };
  };
  explanation: string[];
  created_at: string;
}

export async function analyzeTextOrUrl(
  inputType: "text" | "url",
  content: string
): Promise<AnalysisResponse> {
  const payload =
    inputType === "url"
      ? { input_type: "url", url: content }
      : { input_type: "text", text: content };

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis request failed with code ${response.status}`);
  }

  return response.json();
}

export async function analyzeMediaFile(
  inputType: "image" | "video",
  file: File
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("input_type", inputType);
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Media upload analysis failed with code ${response.status}`);
  }

  return response.json();
}

export async function getAnalysisResult(analysisId: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/api/analyze/${analysisId}`);
  if (!response.ok) {
    throw new Error(`Failed to retrieve results for analysis ID: ${analysisId}`);
  }
  return response.json();
}

export async function getAnalysesHistory(): Promise<AnalysisResponse[]> {
  const response = await fetch(`${API_BASE}/api/analyses`);
  if (!response.ok) {
    throw new Error("Failed to retrieve analysis history.");
  }
  return response.json();
}
