/**
 * TCI Calculator + k(s) Estimator — JavaScript Reference Implementation
 * Thermodynamic Cognition Index
 * By Nile Green / PermaMind
 * Paper: https://zenodo.org/records/19263435
 */

const GRADES = [
  { threshold: 0.6, grade: "A", stage: "Generativity" },
  { threshold: 0.4, grade: "B", stage: "Learning" },
  { threshold: 0.3, grade: "C", stage: "At Risk" },
  { threshold: 0.1, grade: "D", stage: "Collapse Warning" },
  { threshold: -Infinity, grade: "F", stage: "Collapse Imminent" },
];

function getGrade(tci) {
  for (const { threshold, grade, stage } of GRADES) {
    if (tci >= threshold) return { grade, stage };
  }
  return { grade: "F", stage: "Collapse Imminent" };
}

// ─── TCI Calculator ───────────────────────────────────────────────

class TCICalculator {
  /**
   * @param {Object} options
   * @param {number} options.fSurvival - Survival floor for this substrate
   */
  constructor({ fSurvival }) {
    if (fSurvival < 0) throw new Error("fSurvival must be >= 0");
    this.fSurvival = fSurvival;
  }

  /**
   * Compute TCI at a single timestep.
   * @param {number} fTotal - Current prediction-error energy
   * @param {number} k - Current k(s) from KEstimator
   * @returns {Object} TCI result
   */
  compute(fTotal, k) {
    const surplus = fTotal - this.fSurvival;
    const tci = k * surplus;
    const { grade, stage } = getGrade(tci);

    return {
      tci: Math.round(tci * 10000) / 10000,
      surplus: Math.round(surplus * 10000) / 10000,
      grade,
      stage,
      fTotal,
      fSurvival: this.fSurvival,
      k: Math.round(k * 10000) / 10000,
    };
  }
}

// ─── k(s) Estimator ───────────────────────────────────────────────

class KEstimator {
  /**
   * @param {Object} options
   * @param {number} options.windowSize - Rolling window size (default 100)
   * @param {number} options.alpha - Normalization constant (default 1.0)
   * @param {number} options.decay - EMA decay (default 0.99)
   * @param {number} options.kInit - Initial k value (default 0.1)
   * @param {number} options.minSteps - Steps before k is considered stable (default 500)
   */
  constructor({
    windowSize = 100,
    alpha = 1.0,
    decay = 0.99,
    kInit = 0.1,
    minSteps = 500,
  } = {}) {
    this.windowSize = windowSize;
    this.alpha = alpha;
    this.decay = decay;
    this.kInit = kInit;
    this.minSteps = minSteps;

    this._surplusWindow = [];
    this._complexityWindow = [];
    this._priorSurplusMean = null;
    this._priorComplexityMean = null;
    this._kEma = kInit;
    this._steps = 0;
  }

  /**
   * Update k(s) with a new observation.
   * @param {number} fSurplus - Current surplus (fTotal - fSurvival)
   * @param {number} complexity - Behavioral complexity estimate
   * @returns {number} Current k(s) estimate
   */
  update(fSurplus, complexity) {
    this._surplusWindow.push(fSurplus);
    this._complexityWindow.push(complexity);
    if (this._surplusWindow.length > this.windowSize) {
      this._surplusWindow.shift();
      this._complexityWindow.shift();
    }
    this._steps++;

    if (this._surplusWindow.length < this.windowSize) {
      return this._kEma;
    }

    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const curSurplusMean = mean(this._surplusWindow);
    const curComplexityMean = mean(this._complexityWindow);

    if (this._priorSurplusMean !== null) {
      const dSurplus = curSurplusMean - this._priorSurplusMean;
      const dComplexity = curComplexityMean - this._priorComplexityMean;

      if (Math.abs(dSurplus) > 1e-8) {
        let kRaw = this.alpha * (dComplexity / dSurplus);
        kRaw = Math.max(0.01, Math.min(kRaw, 10.0));
        this._kEma = this.decay * this._kEma + (1 - this.decay) * kRaw;
      }
    }

    this._priorSurplusMean = curSurplusMean;
    this._priorComplexityMean = curComplexityMean;

    return this._kEma;
  }

  get k() { return this._kEma; }
  get steps() { return this._steps; }
  get isStable() { return this._steps >= this.minSteps; }

  /** Serialize state for PSSU persistence across sessions */
  toJSON() {
    return {
      kEma: this._kEma,
      steps: this._steps,
      priorSurplusMean: this._priorSurplusMean,
      priorComplexityMean: this._priorComplexityMean,
      windowSize: this.windowSize,
      alpha: this.alpha,
      decay: this.decay,
    };
  }

  /** Restore state — enables k(s) persistence */
  fromJSON(state) {
    this._kEma = state.kEma;
    this._steps = state.steps;
    this._priorSurplusMean = state.priorSurplusMean;
    this._priorComplexityMean = state.priorComplexityMean;
    this.windowSize = state.windowSize;
    this.alpha = state.alpha;
    this.decay = state.decay;
    this._surplusWindow = [];
    this._complexityWindow = [];
  }
}

// ─── Convenience function ─────────────────────────────────────────

function computeTCI(fTotal, fSurvival, k) {
  return new TCICalculator({ fSurvival }).compute(fTotal, k);
}

// Export for Node.js and browser
if (typeof module !== "undefined") {
  module.exports = { TCICalculator, KEstimator, computeTCI, getGrade, GRADES };
}
