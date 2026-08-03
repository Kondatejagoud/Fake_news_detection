import React from "react";

export const AboutPage: React.FC = () => {
  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-fade-in">
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-heading font-extrabold bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
          How the Platform Works
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto text-sm leading-relaxed">
          Explore the science behind our hybrid multimodal fake content detection pipelines.
        </p>
      </div>

      {/* Grid: Extraction & Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* NLP Card */}
        <div className="glass-panel rounded-3xl p-6 space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 text-xl font-bold">
            T
          </div>
          <h3 className="text-xl font-heading font-bold text-slate-200">Text & Claims Analysis</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            The text pipeline analyzes content in two ways:
          </p>
          <ul className="list-disc pl-5 text-xs text-slate-400 space-y-2 leading-relaxed">
            <li>
              <strong className="text-slate-300">Semantic Sentiment Classification:</strong> Uses a distilled Transformer model (DistilBERT) trained on misinformation datasets to analyze sensationalism, lexical bias, and writing styles.
            </li>
            <li>
              <strong className="text-slate-300">Claim Database Querying:</strong> Converts the input statement into dense vector embeddings using Sentence-Transformers and computes the cosine similarity against a database of verified false claims.
            </li>
          </ul>
        </div>

        {/* Image Card */}
        <div className="glass-panel rounded-3xl p-6 space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-violet-500/10 border border-violet-500/30 flex items-center justify-center text-violet-400 text-xl font-bold">
            I
          </div>
          <h3 className="text-xl font-heading font-bold text-slate-200">Image Forensics (ELA)</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Images undergo strict metadata inspection and digital compression checks:
          </p>
          <ul className="list-disc pl-5 text-xs text-slate-400 space-y-2 leading-relaxed">
            <li>
              <strong className="text-slate-300">Error Level Analysis (ELA):</strong> Re-saves the image at a specific JPEG quality ratio (95%) and measures the absolute pixel-by-pixel difference. Authentic images show homogeneous error patterns, while edited or spliced elements display bright, high-variance edges.
            </li>
            <li>
              <strong className="text-slate-300">Splicing CNN:</strong> Processes the image using a convolutional network trained to spot boundary artifacts, color level imbalances, and local compression anomalies.
            </li>
          </ul>
        </div>

        {/* Video Card */}
        <div className="glass-panel rounded-3xl p-6 space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 text-xl font-bold">
            V
          </div>
          <h3 className="text-xl font-heading font-bold text-slate-200">Deepfake Detection</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Videos are processed frame-by-frame using a dynamic temporal system:
          </p>
          <ul className="list-disc pl-5 text-xs text-slate-400 space-y-2 leading-relaxed">
            <li>
              <strong className="text-slate-300">Face Extraction:</strong> OpenCV Haar Cascades and MTCNN scan sampled video frames (at 1 frame per second) to isolate and crop facial regions.
            </li>
            <li>
              <strong className="text-slate-300">GAN/Diffusion Texture Scan:</strong> Cropped faces are fed into an EfficientNet classifier trained on FaceForensics++ to identify synthesized facial grids, warping boundaries, and abnormal Laplacian blur indicators.
            </li>
          </ul>
        </div>

        {/* Fusion Card */}
        <div className="glass-panel rounded-3xl p-6 space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 text-xl font-bold">
            F
          </div>
          <h3 className="text-xl font-heading font-bold text-slate-200">Decision Fusion Consensus</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Combines and normalizes scores using a weighted consensus mathematical formula:
          </p>
          <ul className="list-disc pl-5 text-xs text-slate-400 space-y-2 leading-relaxed">
            <li>
              <strong className="text-slate-300">Weighted Average:</strong> Weights are assigned based on empirical reliability (Text: 35%, Image: 40%, Deepfake: 45%). Unused modules (e.g. no video uploaded) are ignored.
            </li>
            <li>
              <strong className="text-slate-300">Disagreement Penalty:</strong> Reduces the overall analysis confidence score if different modules disagree sharply (e.g., if text NLP reports genuine but image forensics flags digital editing).
            </li>
          </ul>
        </div>
      </div>

      {/* Mathematical Formulations Section */}
      <div className="glass-panel rounded-3xl p-8 space-y-4">
        <h3 className="text-xl font-heading font-bold text-slate-200">Consensus Mathematical Formulas</h3>
        <p className="text-sm text-slate-400 leading-relaxed">
          The final authenticity score and confidence ratings are calculated as follows:
        </p>
        <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-900 overflow-x-auto font-mono text-xs md:text-sm text-slate-300 space-y-4 leading-loose">
          <div>
            <div className="text-sky-400 font-bold mb-1">// Weighted Fake Probability Formula</div>
            <div>P_fake = &Sigma; (w_i * s_i) / &Sigma; w_i  [for all active modules i]</div>
          </div>
          <div>
            <div className="text-sky-400 font-bold mb-1">// Authenticity Score (displayed 0-100)</div>
            <div>Score_auth = round((1 - P_fake) * 100)</div>
          </div>
          <div>
            <div className="text-sky-400 font-bold mb-1">// Confidence Rating (with Disagreement Penalty)</div>
            <div>Confidence = Base_confidence - (30 * (Max_score - Min_score))</div>
          </div>
        </div>
      </div>
    </div>
  );
};
